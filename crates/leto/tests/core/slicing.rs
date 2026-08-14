#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array, Layout, SliceArg, VecStorage};

#[test]
fn test_legacy_slice_empty_range_has_zero_extent_without_underflow() {
    let layout = Layout::c_contiguous([4]).unwrap();
    let sliced = layout.slice(&[(2, 2, 1)]).unwrap();

    assert_eq!(sliced.shape, [0]);
    assert_eq!(sliced.strides, [1]);
    assert_eq!(sliced.offset, 2);
    assert_eq!(sliced.checked_size().unwrap(), 0);
}

#[test]
fn test_slicing() {
    let layout = Layout::c_contiguous([4, 4]).unwrap();
    let storage = VecStorage::new((0..16).collect());
    let array = Array::new(layout, storage).unwrap();

    // Slice axis 0: from index 1 to 3, step 1. Slice axis 1: from index 1 to 4, step 2.
    let sliced_view = array.slice(&[(1, 3, 1), (1, 4, 2)]).unwrap();
    assert_eq!(sliced_view.shape(), [2, 2]);
    assert_eq!(sliced_view.strides(), [4, 2]);
    assert_eq!(sliced_view.offset(), 5);

    // Check physical elements:
    // Original array:
    //  0  1  2  3
    //  4  5  6  7
    //  8  9 10 11
    // 12 13 14 15
    // Sliced view shape [2, 2] starting at offset 5 (element 5).
    // Row 0: index [0, 0] -> physical offset 5 (val 5), index [0, 1] -> physical offset 7 (val 7)
    // Row 1: index [1, 0] -> physical offset 9 (val 9), index [1, 1] -> physical offset 11 (val 11)
    assert_eq!(*sliced_view.get([0, 0]).unwrap(), 5);
    assert_eq!(*sliced_view.get([0, 1]).unwrap(), 7);
    assert_eq!(*sliced_view.get([1, 0]).unwrap(), 9);
    assert_eq!(*sliced_view.get([1, 1]).unwrap(), 11);
}

#[test]
fn test_leto_style_slice_with_negative_bounds_and_reverse_stride() {
    let layout = Layout::c_contiguous([5, 4]).unwrap();
    let storage = VecStorage::new((0..20).collect());
    let array = Array::new(layout, storage).unwrap();

    let view = array
        .slice_with::<2>(&[
            SliceArg::range(Some(-1), None, -2),
            SliceArg::range(Some(1), None, 2),
        ])
        .unwrap();

    assert_eq!(view.shape(), [3, 2]);
    assert_eq!(view.strides(), [-8, 2]);
    assert_eq!(view.offset(), 17);
    assert_eq!(*view.get([0, 0]).unwrap(), 17);
    assert_eq!(*view.get([0, 1]).unwrap(), 19);
    assert_eq!(*view.get([1, 0]).unwrap(), 9);
    assert_eq!(*view.get([2, 1]).unwrap(), 3);
}

#[test]
fn test_leto_style_slice_drops_indexed_axis_and_adds_new_axis() {
    let layout = Layout::c_contiguous([2, 3, 4]).unwrap();
    let storage = VecStorage::new((0..24).collect());
    let array = Array::new(layout, storage).unwrap();

    let view = array
        .slice_with::<2>(&[
            SliceArg::Index(-1),
            SliceArg::NewAxis,
            SliceArg::range(Some(1), None, 1),
            SliceArg::Index(2),
        ])
        .unwrap();

    assert_eq!(view.shape(), [1, 2]);
    assert_eq!(view.strides(), [0, 4]);
    assert_eq!(view.offset(), 18);
    assert_eq!(*view.get([0, 0]).unwrap(), 18);
    assert_eq!(*view.get([0, 1]).unwrap(), 22);
}

#[test]
fn test_leto_style_slice_ellipsis_and_implicit_trailing_axes() {
    let layout = Layout::c_contiguous([2, 3, 4]).unwrap();
    let storage = VecStorage::new((0..24).collect());
    let array = Array::new(layout, storage).unwrap();

    let ellipsis = array
        .slice_with::<3>(&[SliceArg::Ellipsis, SliceArg::Index(-1), SliceArg::NewAxis])
        .unwrap();
    assert_eq!(ellipsis.shape(), [2, 3, 1]);
    assert_eq!(ellipsis.strides(), [12, 4, 0]);
    assert_eq!(*ellipsis.get([1, 2, 0]).unwrap(), 23);

    let implicit = array
        .slice_with::<3>(&[SliceArg::range(Some(1), None, 1)])
        .unwrap();
    assert_eq!(implicit.shape(), [1, 3, 4]);
    assert_eq!(implicit.strides(), [0, 4, 1]);
    assert_eq!(*implicit.get([0, 2, 3]).unwrap(), 23);
}

#[test]
fn test_slicearg_all_step_reverses_axis() {
    // Regression: `SliceArg::All.step(-1)` — as produced by `s![.., ..;-1, ..]`
    // — must convert the full-axis selection into a strided (reversed) range,
    // not silently drop the stride (which previously left the axis forward and
    // C-contiguous).
    assert_eq!(
        SliceArg::All.step(-1),
        SliceArg::Range {
            start: None,
            end: None,
            step: -1,
        }
    );

    // End-to-end: a reversed full-axis slice yields a negative-stride,
    // non-C-contiguous view whose logical order is reversed.
    let layout = Layout::c_contiguous([3]).unwrap();
    let storage = VecStorage::new(vec![10.0f64, 20.0, 30.0]);
    let array = Array::new(layout, storage).unwrap();
    let reversed = array.slice_with::<1>(&[SliceArg::All.step(-1)]).unwrap();
    assert_eq!(reversed.shape(), [3]);
    assert!(
        reversed.as_slice().is_none(),
        "reversed full-axis view must be non-contiguous"
    );
    let collected: Vec<f64> = reversed.iter().copied().collect();
    assert_eq!(collected, vec![30.0, 20.0, 10.0]);
}
