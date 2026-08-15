//! Storage-bound regression coverage for the `leto-ops` entry points that
//! reach `get_unchecked` or raw pointer writes.
//!
//! Most `leto-ops` entry points establish their proof with
//! `Layout::validate_storage_len` before dispatching to an `unsafe` kernel.
//! Three did not — `trace`, `kron`'s strided branch, and `matmul`'s
//! `copy_back_to_out` — so a view whose layout is individually valid but too
//! large for its buffer reached an out-of-bounds access from safe code.
//!
//! A `Layout` cannot prevent this on its own: it carries no pointer and no
//! length, so "fits in the backing storage" is not a property it can hold. The
//! check therefore belongs at the operation boundary, which is what these tests
//! pin. Each asserts the typed `LetoError::StorageError`, not `is_err()`.

use leto::{ArrayView2, ArrayViewMut, Layout, LetoError};

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

#[test]
fn trace_rejects_a_view_whose_layout_overruns_its_buffer() {
    let data = [1.0f64; 4];
    // A 4x4 square matrix addresses 16 elements; the buffer holds 4.
    let layout = Layout::c_contiguous([4, 4]).expect("c-contiguous layout is self-consistent");
    let view = ArrayView2::<f64>::new(layout, &data);
    assert_storage_error(leto_ops::trace(&view), "trace storage bound");
}

#[test]
fn trace_still_accepts_an_in_bounds_matrix() {
    // Positive control: the added check must not reject legitimate input.
    let data = [1.0f64, 2.0, 3.0, 4.0];
    let layout = Layout::c_contiguous([2, 2]).expect("c-contiguous layout is self-consistent");
    let view = ArrayView2::<f64>::new(layout, &data);
    let trace = leto_ops::trace(&view).expect("2x2 trace is in bounds");
    assert!(
        (trace - 5.0).abs() < f64::EPSILON,
        "tr([[1,2],[3,4]]) = 1 + 4 = 5, got {trace}"
    );
}

#[test]
fn kron_rejects_a_strided_operand_whose_layout_overruns_its_buffer() {
    let a_data = [1.0f64; 2];
    let b_data = [1.0f64; 4];
    // A non-unit column stride keeps `a` off the `as_slice` fast path, so the
    // strided `get_unchecked` branch is the one exercised. The layout addresses
    // offsets up to 4 + 2 = 6 against a 2-element buffer.
    let a_layout = Layout::<2>::try_new([2, 2], [4, 2], 0).expect("valid strided layout");
    let b_layout = Layout::c_contiguous([2, 2]).expect("c-contiguous layout is self-consistent");
    let a = ArrayView2::<f64>::new(a_layout, &a_data);
    let b = ArrayView2::<f64>::new(b_layout, &b_data);
    assert_storage_error(leto_ops::kron(&a, &b).map(|_| ()), "kron storage bound");
}

#[test]
fn kron_still_computes_the_documented_product() {
    // Positive control against the mixed-product corollary's base case.
    let a_data = [2.0f64, 3.0];
    let b_data = [5.0f64, 7.0];
    let a = ArrayView2::<f64>::new(Layout::c_contiguous([1, 2]).expect("c-contiguous"), &a_data);
    let b = ArrayView2::<f64>::new(Layout::c_contiguous([2, 1]).expect("c-contiguous"), &b_data);
    let product = leto_ops::kron(&a, &b).expect("in-bounds kron");
    assert_eq!(product.shape(), [2, 2]);
}

#[test]
fn matmul_rejects_an_output_view_whose_layout_overruns_its_buffer() {
    // `copy_back_to_out` writes through `dst`'s layout, but only the scratch
    // output view was validated before the fix, so an over-long `dst` layout
    // reached a raw out-of-bounds write.
    let lhs_data = [1.0f64; 4];
    let rhs_data = [1.0f64; 4];
    let mut out_data = [0.0f64; 4];

    let square = Layout::c_contiguous([2, 2]).expect("c-contiguous layout is self-consistent");
    let lhs = ArrayView2::<f64>::new(square, &lhs_data);
    let rhs = ArrayView2::<f64>::new(square, &rhs_data);

    // Row stride 64 puts row 1 at physical offset 64, far past the 4-element
    // buffer, while the layout itself is perfectly self-consistent.
    let out_layout = Layout::<2>::try_new([2, 2], [64, 1], 0).expect("valid strided layout");
    let mut out = ArrayViewMut::<f64, 2>::new(out_layout, &mut out_data);

    assert_storage_error(leto_ops::matmul(&lhs, &rhs, &mut out), "matmul dst bound");
}

#[test]
fn matmul_still_writes_a_strided_but_in_bounds_output() {
    // Positive control: a genuinely strided destination inside its buffer must
    // still round-trip, so the added check is a bound test and not a
    // contiguity requirement.
    let lhs_data = [1.0f64, 2.0, 3.0, 4.0];
    let rhs_data = [1.0f64, 0.0, 0.0, 1.0];
    let mut out_data = [0.0f64; 8];

    let square = Layout::c_contiguous([2, 2]).expect("c-contiguous layout is self-consistent");
    let lhs = ArrayView2::<f64>::new(square, &lhs_data);
    let rhs = ArrayView2::<f64>::new(square, &rhs_data);

    // Rows at offsets 0 and 4, columns unit-stride: max addressed offset is 5.
    let out_layout = Layout::<2>::try_new([2, 2], [4, 1], 0).expect("valid strided layout");
    let mut out = ArrayViewMut::<f64, 2>::new(out_layout, &mut out_data);
    leto_ops::matmul(&lhs, &rhs, &mut out).expect("strided in-bounds matmul");

    // Multiplying by the identity reproduces `lhs` at the strided positions.
    assert_eq!(out_data[0], 1.0);
    assert_eq!(out_data[1], 2.0);
    assert_eq!(out_data[4], 3.0);
    assert_eq!(out_data[5], 4.0);
}
