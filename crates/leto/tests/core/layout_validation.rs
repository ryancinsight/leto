//! Adversarial coverage for the `Layout` construction boundary and the
//! layout-versus-buffer bound checks that the `unsafe` accessors rest on.
//!
//! Two distinct invariants are exercised, and the split matters:
//!
//! * The **self-contained** invariant — shape product and physical-offset
//!   arithmetic stay in range, and no addressed offset is negative — is what
//!   [`Layout::try_new`] validates. A `Layout` carries no pointer, so this is
//!   everything a layout can check about itself.
//! * The **layout-versus-buffer** invariant — every addressed physical offset
//!   lies inside the backing storage — cannot be expressed by a `Layout` at
//!   all. It is established where a layout meets a buffer. The second half of
//!   this file pins the accessors that previously trusted it without checking.
//!
//! Each case asserts the *typed* error variant, not merely `is_err()`.

use leto::{ArrayViewMut, Layout, LetoError};

/// Helper: assert a `Result` carries `LetoError::Overflow`.
fn assert_overflow<T: std::fmt::Debug>(result: Result<T, LetoError>, case: &str) {
    match result {
        Err(LetoError::Overflow { reason }) => {
            assert!(
                !reason.is_empty(),
                "{case}: Overflow must name the failing arithmetic"
            );
        }
        other => panic!("{case}: expected LetoError::Overflow, got {other:?}"),
    }
}

/// Helper: assert a `Result` carries `LetoError::StorageError`.
fn assert_storage_error<T: std::fmt::Debug>(result: Result<T, LetoError>, case: &str) {
    match result {
        Err(LetoError::StorageError { reason }) => {
            assert!(
                !reason.is_empty(),
                "{case}: StorageError must name the violated bound"
            );
        }
        other => panic!("{case}: expected LetoError::StorageError, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Invalid class 1: shape product overflows `usize`.
// ---------------------------------------------------------------------------

#[test]
fn shape_product_overflowing_usize_is_rejected() {
    // 2^32 * 2^32 == 2^64, one past `usize::MAX` on a 64-bit target.
    let half = 1usize << 32;
    assert_overflow(
        Layout::<2>::try_new([half, half], [1, 1], 0),
        "shape product 2^64",
    );
}

#[test]
fn maximal_shape_product_overflows_on_every_axis_ordering() {
    // The product is order-independent, so the rejection must be too.
    assert_overflow(
        Layout::<3>::try_new([usize::MAX, 2, 2], [1, 1, 1], 0),
        "leading maximal extent",
    );
    assert_overflow(
        Layout::<3>::try_new([2, 2, usize::MAX], [1, 1, 1], 0),
        "trailing maximal extent",
    );
}

// ---------------------------------------------------------------------------
// Invalid class 2: stride/extent product overflows `isize`.
// ---------------------------------------------------------------------------

#[test]
fn stride_extent_product_overflowing_isize_is_rejected() {
    // (shape - 1) * stride = 3 * (isize::MAX / 2) overflows `isize`.
    assert_overflow(
        Layout::<1>::try_new([4], [isize::MAX / 2], 0),
        "stride-extent product",
    );
}

#[test]
fn accumulated_offset_overflowing_isize_is_rejected() {
    // Each axis bound fits, but their sum does not.
    let big = isize::MAX / 2;
    assert_overflow(
        Layout::<3>::try_new([2, 2, 2], [big, big, big], 0),
        "accumulated maximum offset",
    );
}

#[test]
fn base_offset_beyond_isize_is_rejected() {
    // `offset` is a `usize`; a value past `isize::MAX` cannot be converted.
    let past_isize = (isize::MAX as usize) + 1;
    assert_overflow(
        Layout::<1>::try_new([1], [1], past_isize),
        "base offset conversion",
    );
}

// ---------------------------------------------------------------------------
// Invalid class 3: negative strides address a physical offset below zero.
// ---------------------------------------------------------------------------

#[test]
fn negative_stride_underrunning_the_base_offset_is_rejected() {
    // Walking [0..4) at stride -1 from base 2 reaches physical offset -1.
    assert_storage_error(
        Layout::<1>::try_new([4], [-1], 2),
        "negative stride underruns base",
    );
}

#[test]
fn negative_stride_within_the_base_offset_is_accepted() {
    // Positive control: the same walk from base 3 bottoms out at exactly 0,
    // which is in range — rejection must not be blanket "negative strides bad".
    let layout = Layout::<1>::try_new([4], [-1], 3).expect("reverse view over [0, 3] is valid");
    assert_eq!(layout.min_max_offsets(), (0, 3));
}

#[test]
fn mixed_sign_strides_are_evaluated_on_the_minimum_not_the_sum() {
    // Axis 0 contributes +10, axis 1 contributes -12: the sum is negative, but
    // the *minimum* addressed offset (base + all negative contributions) is
    // what matters and it underruns.
    assert_storage_error(
        Layout::<2>::try_new([2, 4], [10, -4], 5),
        "mixed-sign minimum offset",
    );
}

// ---------------------------------------------------------------------------
// Invalid class 4: zero-size edge cases must be accepted, not rejected.
// ---------------------------------------------------------------------------

#[test]
fn zero_extent_axis_collapses_the_addressed_span() {
    // An empty layout addresses no elements, so even a wild stride is
    // unreachable and must be accepted.
    let layout = Layout::<2>::try_new([0, 4], [isize::MAX, 1], 0)
        .expect("an empty layout addresses no element");
    assert_eq!(layout.size(), 0);
    assert_eq!(layout.min_max_offsets(), (0, 0));
}

#[test]
fn zero_extent_axis_with_negative_stride_is_accepted() {
    let layout =
        Layout::<2>::try_new([3, 0], [-100, -100], 0).expect("no addressable element to underrun");
    assert_eq!(layout.size(), 0);
}

#[test]
fn rank_zero_layout_is_valid() {
    let layout = Layout::<0>::try_new([], [], 7).expect("rank-0 scalar layout");
    assert_eq!(layout.size(), 1);
    assert_eq!(layout.offset(), 7);
}

#[test]
fn unit_extent_axes_ignore_their_strides() {
    // (1 - 1) * stride == 0, so a hostile stride on a unit axis is unreachable.
    let layout = Layout::<2>::try_new([1, 3], [isize::MAX, 1], 0)
        .expect("unit extent never applies its stride");
    assert_eq!(layout.min_max_offsets(), (0, 2));
}

// ---------------------------------------------------------------------------
// Construction-path closure: `try_new` and `TryFrom` agree, and are the only
// public routes in.
// ---------------------------------------------------------------------------

#[test]
fn try_from_tuple_matches_try_new() {
    let parts = ([2usize, 3], [3isize, 1], 4usize);
    let from_try_new = Layout::<2>::try_new(parts.0, parts.1, parts.2).expect("valid layout");
    let from_try_from = Layout::<2>::try_from(parts).expect("valid layout");
    assert_eq!(from_try_new, from_try_from);

    let bad = ([4usize], [-1isize], 2usize);
    assert_storage_error(Layout::<1>::try_from(bad), "TryFrom rejects underrun");
}

#[test]
fn accessors_round_trip_the_constructed_parts() {
    let layout = Layout::<3>::try_new([2, 3, 4], [12, 4, 1], 5).expect("valid layout");
    assert_eq!(layout.shape(), [2, 3, 4]);
    assert_eq!(layout.strides(), [12, 4, 1]);
    assert_eq!(layout.offset(), 5);
}

#[test]
fn deserialization_routes_through_validation() {
    // Deserialization is construction: a hostile payload must not mint a
    // layout that `try_new` would have refused.
    let hostile = r#"{"shape":[4],"strides":[-1],"offset":2}"#;
    let decoded: Result<Layout<1>, _> = serde_json::from_str(hostile);
    let message = decoded
        .expect_err("negative-underrun payload must be rejected")
        .to_string();
    assert!(
        message.contains("negative physical offset"),
        "deserialization error must name the violated bound, got: {message}"
    );

    // Positive control: a well-formed payload still round-trips.
    let sound = Layout::<1>::try_new([4], [-1], 3).expect("valid reverse layout");
    let encoded = serde_json::to_string(&sound).expect("serialize");
    let round_tripped: Layout<1> = serde_json::from_str(&encoded).expect("deserialize");
    assert_eq!(round_tripped, sound);
}

// ---------------------------------------------------------------------------
// Layout-versus-buffer: the invariant a `Layout` cannot carry.
//
// These are regression tests for a demonstrated out-of-bounds access. Before
// the fix, `ArrayViewMut`'s `Index`/`IndexMut` impls computed a physical offset
// and dereferenced it without comparing against the backing length, while their
// `get`/`get_mut` siblings did compare. A *validly constructed* layout that is
// simply too large for the buffer was enough to read and write out of bounds
// from entirely safe code.
// ---------------------------------------------------------------------------

/// The exact historical proof-of-concept: every call below is safe Rust, and
/// the layout comes from the validating `c_contiguous` constructor. Sealing
/// `Layout` alone would not have prevented this.
#[test]
#[should_panic(expected = "exceeds backing length")]
fn index_past_the_backing_buffer_panics_rather_than_reading_out_of_bounds() {
    let mut data = [0u32; 4];
    let layout = Layout::c_contiguous([1000]).expect("c-contiguous layout is self-consistent");
    let view = ArrayViewMut::<u32, 1>::new(layout, &mut data);
    let _ = view[999usize];
}

#[test]
#[should_panic(expected = "exceeds backing length")]
fn index_mut_past_the_backing_buffer_panics_rather_than_writing_out_of_bounds() {
    let mut data = [0u32; 4];
    let layout = Layout::c_contiguous([1000]).expect("c-contiguous layout is self-consistent");
    let mut view = ArrayViewMut::<u32, 1>::new(layout, &mut data);
    view[999usize] = 0xDEAD_BEEF;
}

#[test]
#[should_panic(expected = "exceeds backing length")]
fn rank_two_index_past_the_backing_buffer_panics() {
    let mut data = [0u32; 4];
    let layout = Layout::c_contiguous([100, 100]).expect("c-contiguous layout is self-consistent");
    let mut view = ArrayViewMut::<u32, 2>::new(layout, &mut data);
    view[[99, 99]] = 1;
}

/// `get`/`get_mut` already reported this as a typed error; that behavior is
/// the contract `Index` now matches, so it is pinned against regression.
#[test]
fn get_reports_an_over_long_layout_as_a_typed_error() {
    let mut data = [0u32; 4];
    let layout = Layout::c_contiguous([1000]).expect("c-contiguous layout is self-consistent");
    let mut view = ArrayViewMut::<u32, 1>::new(layout, &mut data);
    assert_storage_error(view.get([999]).copied(), "get past buffer");
    assert_storage_error(view.get_mut([999]).copied(), "get_mut past buffer");
}

#[test]
fn try_new_rejects_a_view_whose_layout_overruns_its_buffer() {
    let data = [0u32; 4];
    let layout = Layout::c_contiguous([1000]).expect("c-contiguous layout is self-consistent");
    assert_storage_error(
        leto::ArrayView::<u32, 1>::try_new(layout, &data).map(|_| ()),
        "ArrayView::try_new bound check",
    );
}

/// In-bounds indexing through the same accessors must keep working — the added
/// bound check must not reject legitimate strided access.
#[test]
fn in_bounds_strided_indexing_is_unaffected() {
    let mut data = [10u32, 11, 12, 13, 14, 15];
    let layout = Layout::<2>::try_new([2, 3], [3, 1], 0).expect("valid layout");
    let mut view = ArrayViewMut::<u32, 2>::new(layout, &mut data);
    assert_eq!(view[[0, 0]], 10);
    assert_eq!(view[[1, 2]], 15);
    view[[1, 0]] = 99;
    assert_eq!(view[[1, 0]], 99);

    // A reverse-stride view over the same buffer: logical index `i` addresses
    // physical offset `5 - i`, so index 2 aliases the element written above.
    let reverse = Layout::<1>::try_new([3], [-1], 5).expect("valid reverse layout");
    let reverse_view = ArrayViewMut::<u32, 1>::new(reverse, &mut data);
    assert_eq!(reverse_view[0usize], 15);
    assert_eq!(reverse_view[1usize], 14);
    assert_eq!(reverse_view[2usize], 99, "aliases the [1, 0] write above");
}
