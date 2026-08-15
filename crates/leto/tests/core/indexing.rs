#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{
    Array, Array1, Array2, Array4, ArrayView, ArrayView4, ArrayViewMut, ArrayViewMut4, Layout,
    LetoError, Storage, VecStorage,
};

#[test]
fn test_array_creation_and_indexing() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let storage = VecStorage::new(vec![10, 11, 12, 20, 21, 22]);
    let array = Array::new(layout, storage).unwrap();

    assert_eq!(*array.get([0, 0]).unwrap(), 10);
    assert_eq!(*array.get([0, 2]).unwrap(), 12);
    assert_eq!(*array.get([1, 1]).unwrap(), 21);
}

#[test]
fn test_view_try_new_rejects_external_out_of_bounds_layout() {
    let data = [10, 20];
    let invalid = Layout::try_new([2], [1], 1).unwrap();

    assert!(matches!(
        ArrayView::try_new(invalid, &data),
        Err(LetoError::StorageError { .. })
    ));

    let valid = ArrayView::try_new(Layout::try_new([2], [1], 0).unwrap(), &data).unwrap();
    assert_eq!(*valid.get([1]).unwrap(), 20);
}

#[test]
fn test_view_mut_try_new_rejects_external_out_of_bounds_layout() {
    let mut data = [10, 20];
    let invalid = Layout::try_new([2], [1], 1).unwrap();

    assert!(matches!(
        ArrayViewMut::try_new(invalid, &mut data),
        Err(LetoError::StorageError { .. })
    ));
}

#[test]
fn test_rank_aliases_construct_owned_arrays_and_views() {
    let vector: Array1<i32> = Array::new(
        Layout::c_contiguous([3]).unwrap(),
        VecStorage::new(vec![1, 2, 3]),
    )
    .unwrap();
    assert_eq!(*vector.view().get([2]).unwrap(), 3);

    let matrix: Array2<i32> = Array::new(
        Layout::c_contiguous([2, 2]).unwrap(),
        VecStorage::new(vec![1, 2, 3, 4]),
    )
    .unwrap();
    assert_eq!(*matrix.view().get([1, 0]).unwrap(), 3);

    let mut volume_time: Array4<i32> =
        Array::from_shape_vec([1, 2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    {
        let view: ArrayView4<'_, i32> = volume_time.view();
        assert_eq!(*view.get([0, 1, 1, 0]).unwrap(), 7);
    }
    {
        let mut view: ArrayViewMut4<'_, i32> = volume_time.view_mut();
        *view.get_mut([0, 1, 1, 1]).unwrap() = 10;
    }
    assert_eq!(*volume_time.get([0, 1, 1, 1]).unwrap(), 10);
}

#[test]
fn test_array1_supports_usize_indexing() {
    let mut vector = Array1::from_shape_vec([3], vec![1, 2, 3]).unwrap();

    vector[1] = 20;

    assert_eq!(vector[0], 1);
    assert_eq!(vector[1], 20);
    assert_eq!(vector[2], 3);
}

#[test]
fn test_owned_array_equality_checks_shape_and_values() {
    let lhs = Array2::from_shape_vec([1, 4], vec![1, 2, 3, 4]).unwrap();
    let same = Array2::from_shape_vec([1, 4], vec![1, 2, 3, 4]).unwrap();
    let different_shape = Array2::from_shape_vec([2, 2], vec![1, 2, 3, 4]).unwrap();
    let different_value = Array2::from_shape_vec([1, 4], vec![1, 2, 3, 5]).unwrap();

    assert_eq!(lhs, same);
    assert_ne!(lhs, different_shape);
    assert_ne!(lhs, different_value);
}

#[test]
fn test_axis_iter_yields_read_only_subviews() {
    let array =
        Array::<i32, VecStorage<i32>, 2>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let rows: Vec<Vec<i32>> = array
        .view()
        .axis_iter::<1>(0)
        .unwrap()
        .map(|row| {
            (0..row.shape()[0])
                .map(|col| *row.get([col]).unwrap())
                .collect()
        })
        .collect();

    assert_eq!(rows, vec![vec![1, 2, 3], vec![4, 5, 6]]);
}

#[test]
fn test_axis_iter_mut_yields_mutable_subviews() {
    let mut array =
        Array::<i32, VecStorage<i32>, 2>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();

    // Increment all values in row 1 by 10
    let mut iter = array.view_mut().axis_iter_mut::<1>(0).unwrap();
    let _row0 = iter.next().unwrap();
    let mut row1 = iter.next().unwrap();
    for col in 0..row1.shape()[0] {
        let val = row1.get_mut([col]).unwrap();
        *val += 10;
    }

    let rows: Vec<Vec<i32>> = array
        .view()
        .axis_iter::<1>(0)
        .unwrap()
        .map(|row| {
            (0..row.shape()[0])
                .map(|col| *row.get([col]).unwrap())
                .collect()
        })
        .collect();

    assert_eq!(rows, vec![vec![1, 2, 3], vec![14, 15, 16]]);
}

#[test]
fn test_view_axis_iter_methods_yield_subviews() {
    let mut array =
        Array::<i32, VecStorage<i32>, 2>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();

    let columns: Vec<Vec<i32>> = array
        .view()
        .axis_iter::<1>(1)
        .unwrap()
        .map(|column| {
            (0..column.shape()[0])
                .map(|row| *column.get([row]).unwrap())
                .collect()
        })
        .collect();
    assert_eq!(columns, vec![vec![1, 4], vec![2, 5], vec![3, 6]]);

    for mut row in array.view_mut().axis_iter_mut::<1>(0).unwrap() {
        *row.get_mut([0]).unwrap() *= 10;
    }
    assert_eq!(array.storage().as_slice(), &[10, 2, 3, 40, 5, 6]);
}

#[test]
fn test_named_rows_and_columns_yield_zero_copy_subviews() {
    let array =
        Array::<i32, VecStorage<i32>, 2>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();

    let rows: Vec<Vec<i32>> = array
        .rows()
        .unwrap()
        .map(|row| {
            (0..row.shape()[0])
                .map(|col| *row.get([col]).unwrap())
                .collect()
        })
        .collect();
    let columns: Vec<Vec<i32>> = array
        .columns()
        .unwrap()
        .map(|column| {
            (0..column.shape()[0])
                .map(|row| *column.get([row]).unwrap())
                .collect()
        })
        .collect();

    assert_eq!(rows, vec![vec![1, 2, 3], vec![4, 5, 6]]);
    assert_eq!(columns, vec![vec![1, 4], vec![2, 5], vec![3, 6]]);
}

#[test]
fn test_named_mutable_rows_and_columns_update_backing_storage() {
    let mut array =
        Array::<i32, VecStorage<i32>, 2>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();

    for mut row in array.rows_mut().unwrap() {
        *row.get_mut([0]).unwrap() *= 10;
    }
    for mut column in array.view_mut().columns_mut().unwrap() {
        *column.get_mut([1]).unwrap() += 100;
    }

    assert_eq!(array.storage().as_slice(), &[10, 2, 3, 140, 105, 106]);
}
