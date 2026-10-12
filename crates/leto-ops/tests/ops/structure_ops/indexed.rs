//! Indexed and coordinate map inplace traversal.

use super::*;

#[test]
fn test_indexed_zip_mut_with_uses_logical_index() {
    let mut out = arr([2, 2], vec![0.0; 4]);
    let a = arr([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = arr([2, 2], vec![10.0, 20.0, 30.0, 40.0]);
    let c = arr([2, 2], vec![100.0, 200.0, 300.0, 400.0]);
    let d = arr([2, 2], vec![1000.0, 2000.0, 3000.0, 4000.0]);

    indexed_zip_mut_with(
        out.view_mut(),
        (&a.view(), &b.view(), &c.view(), &d.view()),
        |[i, j], o, (&av, &bv, &cv, &dv)| {
            *o = av + bv + cv + dv + (i * 10 + j) as f64;
        },
    )
    .unwrap();

    assert_eq!(out.storage().as_slice(), &[1111.0, 2223.0, 3343.0, 4455.0]);
}

#[test]
fn test_indexed_map_inplace_uses_logical_index() {
    let mut out = arr([2, 3], vec![0.0; 6]);

    indexed_map_inplace(&mut out.view_mut(), |[i, j], value| {
        *value = (10 * i + j) as f64;
    })
    .unwrap();

    assert_eq!(out.storage().as_slice(), &[0.0, 1.0, 2.0, 10.0, 11.0, 12.0]);
}

#[test]
fn test_indexed_map4_inplace_fills_multiple_outputs() {
    let mut a = arr([2, 3], vec![0.0; 6]);
    let mut b = arr([2, 3], vec![0.0; 6]);
    let mut c = arr([2, 3], vec![0.0; 6]);
    let mut d = arr([2, 3], vec![0.0; 6]);

    indexed_map4_inplace(
        &mut a.view_mut(),
        &mut b.view_mut(),
        &mut c.view_mut(),
        &mut d.view_mut(),
        |[i, j], av, bv, cv, dv| {
            let index = (10 * i + j) as f64;
            *av = index;
            *bv = index + 1.0;
            *cv = index + 2.0;
            *dv = index + 3.0;
        },
    )
    .unwrap();

    assert_eq!(a.storage().as_slice(), &[0.0, 1.0, 2.0, 10.0, 11.0, 12.0]);
    assert_eq!(b.storage().as_slice(), &[1.0, 2.0, 3.0, 11.0, 12.0, 13.0]);
    assert_eq!(c.storage().as_slice(), &[2.0, 3.0, 4.0, 12.0, 13.0, 14.0]);
    assert_eq!(d.storage().as_slice(), &[3.0, 4.0, 5.0, 13.0, 14.0, 15.0]);
}

#[test]
fn test_coordinate_map_inplace_visits_sparse_coordinates_in_order() {
    let mut out = arr([2, 3], vec![0.0; 6]);
    let coordinates = [[1, 2], [0, 1], [1, 2]];

    coordinate_map_inplace(
        &mut out.view_mut(),
        &coordinates,
        |ordinal, [i, j], value| {
            *value += (100 * ordinal + 10 * i + j) as f64;
        },
    )
    .unwrap();

    assert_eq!(
        out.storage().as_slice(),
        &[0.0, 101.0, 0.0, 0.0, 0.0, 224.0]
    );
}

#[test]
fn test_coordinate_map_plan_visits_sparse_coordinates_in_order() {
    let mut out = arr([2, 3], vec![0.0; 6]);
    let coordinates = [[1, 2], [0, 1], [1, 2]];
    let plan = coordinate_map_plan(&out.view_mut(), &coordinates).unwrap();

    assert_eq!(plan.len(), 3);
    assert_eq!(*plan.layout(), out.layout());
    coordinate_map_plan_inplace(&mut out.view_mut(), &plan, |ordinal, [i, j], value| {
        *value += (100 * ordinal + 10 * i + j) as f64;
    })
    .unwrap();

    assert_eq!(
        out.storage().as_slice(),
        &[0.0, 101.0, 0.0, 0.0, 0.0, 224.0]
    );
}

#[test]
fn test_coordinate_map_plan_rejects_layout_mismatch() {
    let mut source = arr([2, 3], vec![0.0; 6]);
    let coordinates = [[1, 2], [0, 1]];
    let plan = coordinate_map_plan(&source.view_mut(), &coordinates).unwrap();
    let mut target = Array::new(
        Layout::f_contiguous([2, 3]).unwrap(),
        VecStorage::new(vec![0.0; 6]),
    )
    .unwrap();

    let err = coordinate_map_plan_inplace(&mut target.view_mut(), &plan, |_, _, value| {
        *value = 1.0;
    })
    .unwrap_err();

    assert_eq!(
        err,
        LetoError::StorageError {
            reason: "coordinate map plan target layout differs from planned layout".to_string(),
        }
    );
    assert_eq!(target.storage().as_slice(), &[0.0; 6]);
}

#[test]
fn test_coordinate_map_inplace_rejects_out_of_bounds_coordinate() {
    let mut out = arr([2, 3], vec![0.0; 6]);
    let err = coordinate_map_inplace(&mut out.view_mut(), &[[0, 0], [2, 1]], |_, _, value| {
        *value = 1.0;
    })
    .unwrap_err();

    assert_eq!(
        err,
        LetoError::OutOfBounds {
            index: vec![2, 1],
            shape: vec![2, 3],
        }
    );
    assert_eq!(out.storage().as_slice(), &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}
