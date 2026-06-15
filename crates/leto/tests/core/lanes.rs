//! Lane iteration: 1-D views along an axis (count, content, strided, mutable).

use leto::{Array1, Array2};

fn leto2(shape: [usize; 2], data: Vec<i32>) -> Array2<i32> {
    Array2::from_shape_vec(shape, data).unwrap()
}

/// Collect a lane's elements in order.
fn lane_elems(l: &leto::ArrayView1<'_, i32>) -> Vec<i32> {
    l.iter().copied().collect()
}

#[test]
fn lanes_count_and_content() {
    // [[1,2,3],[4,5,6]]: lanes along axis 0 vary the row → columns;
    // lanes along axis 1 vary the column → rows.
    let a = leto2([2, 3], vec![1, 2, 3, 4, 5, 6]);

    let cols: Vec<Vec<i32>> = a.lanes::<1>(0).unwrap().map(|l| lane_elems(&l)).collect();
    assert_eq!(cols, vec![vec![1, 4], vec![2, 5], vec![3, 6]]);
    assert_eq!(a.lanes::<1>(0).unwrap().len(), 3);

    let rows: Vec<Vec<i32>> = a.lanes::<1>(1).unwrap().map(|l| lane_elems(&l)).collect();
    assert_eq!(rows, vec![vec![1, 2, 3], vec![4, 5, 6]]);
    assert_eq!(a.lanes::<1>(1).unwrap().len(), 2);
}

#[test]
fn lanes_dual_to_rows_columns() {
    // lanes(axis=1) == rows(); lanes(axis=0) == columns().
    let a = leto2([2, 3], vec![1, 2, 3, 4, 5, 6]);
    let lane_rows: Vec<Vec<i32>> = a.lanes::<1>(1).unwrap().map(|l| lane_elems(&l)).collect();
    let rows: Vec<Vec<i32>> = a.rows().unwrap().map(|r| lane_elems(&r)).collect();
    assert_eq!(lane_rows, rows);

    let lane_cols: Vec<Vec<i32>> = a.lanes::<1>(0).unwrap().map(|l| lane_elems(&l)).collect();
    let cols: Vec<Vec<i32>> = a.columns().unwrap().map(|c| lane_elems(&c)).collect();
    assert_eq!(lane_cols, cols);
}

#[test]
fn lanes_over_transposed_view() {
    // Transposed [[1,2,3],[4,5,6]] is logical 3×2 [[1,4],[2,5],[3,6]].
    // lanes(axis=1) over it = its rows: [1,4],[2,5],[3,6].
    let a = leto2([2, 3], vec![1, 2, 3, 4, 5, 6]);
    let t = a.transpose([1, 0]).unwrap();
    let lanes: Vec<Vec<i32>> = t.lanes::<1>(1).unwrap().map(|l| lane_elems(&l)).collect();
    assert_eq!(lanes, vec![vec![1, 4], vec![2, 5], vec![3, 6]]);
}

#[test]
fn lanes_double_ended() {
    let a = leto2([2, 3], vec![1, 2, 3, 4, 5, 6]);
    // lanes(axis=0) = columns [1,4],[2,5],[3,6].
    let mut it = a.lanes::<1>(0).unwrap();
    assert_eq!(lane_elems(&it.next().unwrap()), vec![1, 4]);
    assert_eq!(lane_elems(&it.next_back().unwrap()), vec![3, 6]);
    assert_eq!(lane_elems(&it.next().unwrap()), vec![2, 5]);
    assert!(it.next().is_none());
    assert!(it.next_back().is_none());
}

#[test]
fn lanes_rank1_yields_single_whole_lane() {
    let a = Array1::from_shape_vec([4], vec![7, 8, 9, 10]).unwrap();
    let lanes: Vec<Vec<i32>> = a.lanes::<0>(0).unwrap().map(|l| lane_elems(&l)).collect();
    assert_eq!(lanes, vec![vec![7, 8, 9, 10]]);
}

#[test]
fn lanes_axis_out_of_bounds_errors() {
    let a = leto2([2, 3], vec![1, 2, 3, 4, 5, 6]);
    assert!(a.lanes::<1>(2).is_err());
}

#[test]
fn lanes_mut_partitions_and_writes() {
    // Zero array; write element k of lane r (row lane) to r*10 + k.
    let mut a = leto2([2, 3], vec![0; 6]);
    for (r, mut lane) in a.lanes_mut::<1>(1).unwrap().enumerate() {
        for k in 0..3usize {
            *lane.get_mut([k]).unwrap() = (r * 10 + k) as i32;
        }
    }
    // Row 0 → [0,1,2]; row 1 → [10,11,12].
    let contents: Vec<i32> = a.iter().copied().collect();
    assert_eq!(contents, vec![0, 1, 2, 10, 11, 12]);
}

#[test]
fn lanes_mut_along_axis0_writes_columns_disjointly() {
    // Lanes along axis 0 are columns; writing each lane independently must not
    // corrupt neighbours (disjointness).
    let mut a = leto2([3, 2], vec![0; 6]);
    for (c, mut lane) in a.lanes_mut::<1>(0).unwrap().enumerate() {
        // lane length 3 (the rows of column c).
        for k in 0..3usize {
            *lane.get_mut([k]).unwrap() = (100 * c + k) as i32;
        }
    }
    // Column 0 down rows: 0,1,2; column 1: 100,101,102. Row-major storage:
    // [[0,100],[1,101],[2,102]].
    let contents: Vec<i32> = a.iter().copied().collect();
    assert_eq!(contents, vec![0, 100, 1, 101, 2, 102]);
}
