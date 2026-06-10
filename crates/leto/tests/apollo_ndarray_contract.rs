#![cfg(feature = "ndarray-compat")]

use leto::{Array2, Array3, ArrayView2, ArrayViewMut2, Layout, SliceArg, Storage, VecStorage};
use ndarray::{Array2 as NdArray2, Axis};

fn assert_slice_eq<T>(left: &[T], right: &[T])
where
    T: Copy + PartialEq + core::fmt::Debug,
{
    assert_eq!(left, right);
}

#[test]
fn apollo_constructors_match_ndarray_c_order() {
    let zeros = Array2::<f64>::zeros([2, 3]);
    let nd_zeros = NdArray2::<f64>::zeros((2, 3));
    assert_eq!(zeros.shape(), [2, 3]);
    assert_eq!(zeros.strides(), [3, 1]);
    assert_slice_eq(zeros.storage().as_slice(), nd_zeros.as_slice().unwrap());

    let filled = Array2::<i32>::from_elem([2, 3], 7);
    let nd_filled = NdArray2::<i32>::from_elem((2, 3), 7);
    assert_slice_eq(filled.storage().as_slice(), nd_filled.as_slice().unwrap());

    let generated = Array3::from_shape_fn([2, 3, 4], |[x, y, z]| x * 12 + y * 4 + z);
    assert_eq!(generated.shape(), [2, 3, 4]);
    assert_eq!(generated.strides(), [12, 4, 1]);
    assert_slice_eq(
        generated.storage().as_slice(),
        &(0usize..24).collect::<Vec<_>>(),
    );
}

#[test]
fn apollo_view_transpose_and_broadcast_match_ndarray_metadata_and_values() {
    let values = (0..6).collect::<Vec<_>>();
    let leto = Array2::from_shape_vec([2, 3], values.clone()).unwrap();
    let ndarray = NdArray2::from_shape_vec((2, 3), values).unwrap();

    let leto_t = leto.view().transpose([1, 0]).unwrap();
    let ndarray_t = ndarray.view().reversed_axes();
    assert_eq!(leto_t.shape(), [3, 2]);
    assert_eq!(leto_t.strides(), [1, 3]);
    assert_eq!(ndarray_t.shape(), &[3, 2]);
    assert_eq!(ndarray_t.strides(), &[1, 3]);
    for row in 0..3 {
        for col in 0..2 {
            assert_eq!(
                *leto_t.get([row, col]).unwrap(),
                ndarray_t[[row, col]],
                "transpose mismatch at [{row}, {col}]"
            );
        }
    }

    let row = Array2::from_shape_vec([1, 3], vec![10, 20, 30]).unwrap();
    let nd_row = NdArray2::from_shape_vec((1, 3), vec![10, 20, 30]).unwrap();
    let leto_b = row.view().broadcast([2, 3]).unwrap();
    let nd_b = nd_row.broadcast((2, 3)).unwrap();
    assert_eq!(leto_b.shape(), [2, 3]);
    assert_eq!(leto_b.strides(), [0, 1]);
    assert_eq!(nd_b.strides(), &[0, 1]);
    for row in 0..2 {
        for col in 0..3 {
            assert_eq!(
                *leto_b.get([row, col]).unwrap(),
                nd_b[[row, col]],
                "broadcast mismatch at [{row}, {col}]"
            );
        }
    }
}

#[test]
fn apollo_axis_iteration_and_mutation_match_ndarray() {
    let mut leto = Array2::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let mut ndarray = NdArray2::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).unwrap();

    let leto_rows = leto
        .view()
        .axis_iter::<1>(0)
        .unwrap()
        .map(|row| {
            (0..row.shape()[0])
                .map(|col| *row.get([col]).unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let ndarray_rows = ndarray
        .axis_iter(Axis(0))
        .map(|row| row.iter().copied().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert_eq!(leto_rows, ndarray_rows);

    for mut row in leto.view_mut().axis_iter_mut::<1>(0).unwrap() {
        for col in 0..row.shape()[0] {
            *row.get_mut([col]).unwrap() *= 10;
        }
    }
    ndarray.axis_iter_mut(Axis(0)).for_each(|mut row| {
        row.iter_mut().for_each(|value| *value *= 10);
    });
    assert_slice_eq(leto.storage().as_slice(), ndarray.as_slice().unwrap());
}

#[test]
fn apollo_ndarray_roundtrip_preserves_contiguous_values_and_negative_stride_views() {
    let nd = NdArray2::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).unwrap();
    let leto_owned = Array2::from(nd.clone());
    assert_eq!(leto_owned.shape(), [2, 3]);
    assert_slice_eq(leto_owned.storage().as_slice(), nd.as_slice().unwrap());

    let nd_back = ndarray::Array::try_from(leto_owned).unwrap();
    assert_eq!(nd_back.shape(), &[2, 3]);
    assert_slice_eq(nd_back.as_slice().unwrap(), nd.as_slice().unwrap());

    let reversed = nd.slice(ndarray::s![..;-1, ..;-1]);
    let leto_reversed = ArrayView2::from(reversed);
    assert_eq!(leto_reversed.shape(), [2, 3]);
    assert_eq!(leto_reversed.strides(), [-3, -1]);
    assert_eq!(*leto_reversed.get([0, 0]).unwrap(), 6);
    assert_eq!(*leto_reversed.get([1, 2]).unwrap(), 1);
}

#[test]
fn apollo_mutable_broadcast_rejects_zero_stride_aliasing_like_ndarray_contract_boundary() {
    let mut leto = Array2::from_shape_vec([1, 3], vec![1, 2, 3]).unwrap();
    let result = leto.view_mut().broadcast_mut([2, 3]);
    assert!(matches!(
        result,
        Err(leto::LetoError::IncompatibleBroadcast { .. })
    ));
}

#[test]
fn apollo_slice_with_matches_ndarray_shape_stride_and_values() {
    let leto = Array3::from_shape_fn([2, 3, 4], |[x, y, z]| x * 12 + y * 4 + z);
    let ndarray = ndarray::Array3::from_shape_fn((2, 3, 4), |(x, y, z)| x * 12 + y * 4 + z);

    let leto_slice = leto
        .slice_with::<3>(&[
            SliceArg::range(Some(1), None, 1),
            SliceArg::range(None, None, 2),
            SliceArg::NewAxis,
            SliceArg::Index(2),
        ])
        .unwrap();
    let ndarray_slice = ndarray.slice(ndarray::s![1.., ..;2, ndarray::NewAxis, 2]);

    assert_eq!(leto_slice.shape(), [1, 2, 1]);
    assert_eq!(ndarray_slice.shape(), &[1, 2, 1]);
    assert_eq!(leto_slice.strides(), [0, 8, 0]);
    assert_eq!(ndarray_slice.strides(), &[0, 8, 0]);
    for x in 0..1 {
        for y in 0..2 {
            assert_eq!(
                *leto_slice.get([x, y, 0]).unwrap(),
                ndarray_slice[[x, y, 0]],
                "slice mismatch at [{x}, {y}, 0]"
            );
        }
    }
}

#[test]
fn apollo_reshape_and_to_contiguous_match_ndarray_value_order() {
    let values = (0..12).collect::<Vec<_>>();
    let leto = Array3::from_shape_vec([2, 3, 2], values.clone()).unwrap();
    let ndarray = ndarray::Array3::from_shape_vec((2, 3, 2), values).unwrap();

    let leto_reshaped = leto.reshape([3, 4]).unwrap();
    let ndarray_reshaped = ndarray.view().into_shape_with_order((3, 4)).unwrap();
    assert_eq!(leto_reshaped.shape(), [3, 4]);
    assert_eq!(ndarray_reshaped.shape(), &[3, 4]);
    for row in 0..3 {
        for col in 0..4 {
            assert_eq!(
                *leto_reshaped.get([row, col]).unwrap(),
                ndarray_reshaped[[row, col]],
                "reshape mismatch at [{row}, {col}]"
            );
        }
    }

    let leto_strided = leto
        .slice_with::<3>(&[
            SliceArg::All,
            SliceArg::range(Some(0), Some(3), 2),
            SliceArg::All,
        ])
        .unwrap();
    let ndarray_strided = ndarray.slice(ndarray::s![.., ..;2, ..]);
    let leto_contiguous = leto_strided.to_contiguous();
    let ndarray_contiguous = ndarray_strided.as_standard_layout().to_owned();

    assert_eq!(leto_contiguous.shape(), [2, 2, 2]);
    assert_slice_eq(
        leto_contiguous.storage().as_slice(),
        ndarray_contiguous.as_slice().unwrap(),
    );
}

#[test]
fn apollo_mutable_ndarray_view_roundtrip_updates_original_storage() {
    let mut array = Array2::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    {
        let mut nd_view = ndarray::ArrayViewMut::try_from(array.view_mut()).unwrap();
        nd_view[[1, 2]] = 60;
        let mut leto_view = ArrayViewMut2::from(nd_view);
        *leto_view.get_mut([0, 1]).unwrap() = 20;
    }
    assert_slice_eq(array.storage().as_slice(), &[1, 20, 3, 4, 5, 60]);
}

#[test]
fn apollo_rejects_layouts_that_exceed_storage_bounds() {
    let layout = Layout::new([2, 3], [3, 1], 0);
    let result = leto::Array::<i32, VecStorage<i32>, 2>::new(layout, VecStorage::new(vec![1, 2]));
    assert!(matches!(result, Err(leto::LetoError::StorageError { .. })));
}

#[cfg(feature = "mnemosyne-alloc")]
#[test]
fn apollo_mnemosyne_owned_constructors_match_ndarray_c_order() {
    let values = vec![1_i32, 2, 3, 4, 5, 6];
    let leto =
        leto::Array::<i32, leto::MnemosyneStorage<i32>, 2>::from_mnemosyne_slice([2, 3], &values)
            .unwrap();
    let ndarray = NdArray2::from_shape_vec((2, 3), values).unwrap();

    assert_eq!(leto.shape(), [2, 3]);
    assert_eq!(leto.strides(), [3, 1]);
    assert_slice_eq(leto.storage().as_slice(), ndarray.as_slice().unwrap());

    let zeros = leto::Array::<f64, leto::MnemosyneStorage<f64>, 2>::zeros_mnemosyne([2, 3]);
    let nd_zeros = NdArray2::<f64>::zeros((2, 3));
    assert_eq!(zeros.shape(), [2, 3]);
    assert_eq!(zeros.strides(), [3, 1]);
    assert_slice_eq(zeros.storage().as_slice(), nd_zeros.as_slice().unwrap());

    let result =
        leto::Array::<i32, leto::MnemosyneStorage<i32>, 2>::from_mnemosyne_slice([2, 3], &[1, 2]);
    assert!(matches!(result, Err(leto::LetoError::StorageError { .. })));
}
