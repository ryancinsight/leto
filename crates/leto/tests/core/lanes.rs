//! Lane iteration: 1-D views along an axis (count, content, strided, mutable).

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

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

#[test]
fn test_lanes_mut_collect_aliasing() {
    let mut a = leto2([2, 3], vec![0; 6]);
    let mut lanes: Vec<_> = a.lanes_mut::<1>(0).unwrap().collect();
    *lanes[0].get_mut([0]).unwrap() = 1;
    *lanes[1].get_mut([0]).unwrap() = 2;
    *lanes[0].get_mut([1]).unwrap() = 3;
    assert_eq!(
        a.iter().copied().collect::<Vec<_>>(),
        vec![1, 2, 0, 3, 0, 0]
    );
}

#[test]
fn lanes_mut_rejects_non_injective_layout() {
    // Shape [2, 2], strides [1, 1]: zero-stride-free yet non-injective —
    // logical (0, 1) and (1, 0) share physical offset 1, so distinct lanes
    // would write one element. The mutable iterator must reject it.
    let layout = leto::Layout::try_new([2, 2], [1, 1], 0).unwrap();
    let mut data = vec![0i32; 4];
    let view = leto::ArrayViewMut::try_new(layout, data.as_mut_slice()).unwrap();
    assert!(view.lanes_mut::<1>(0).is_err());

    let layout = leto::Layout::try_new([2, 2], [1, 1], 0).unwrap();
    let view = leto::ArrayViewMut::try_new(layout, data.as_mut_slice()).unwrap();
    assert!(view.axis_iter_mut::<1>(0).is_err());
}

#[test]
fn interleaved_lane_window_state_and_element_paths() {
    // Column lanes of a C-order matrix have stride 3: each yielded window
    // spans other lanes' elements, so it is not exclusively owned; row lanes
    // (stride 1) own their window and keep whole-window slice access.
    let mut a = leto2([2, 3], vec![1, 2, 3, 4, 5, 6]);
    {
        let mut cols = a.lanes_mut::<1>(0).unwrap();
        let col0 = cols.next().unwrap();
        assert!(!col0.has_exclusive_window());
        // Element access, fill, assign, and to_contiguous stay available.
        let materialized = col0.to_contiguous();
        assert_eq!(materialized.iter().copied().collect::<Vec<_>>(), vec![1, 4]);
    }
    {
        let mut col0 = a.lanes_mut::<1>(0).unwrap().next().unwrap();
        col0.fill(9);
        let rhs_store = [7i32, 8];
        let rhs_layout = leto::Layout::try_new([2], [1], 0).unwrap();
        let rhs = leto::ArrayView::try_new(rhs_layout, rhs_store.as_slice()).unwrap();
        col0.try_assign(&rhs).unwrap();
    }
    assert_eq!(
        a.iter().copied().collect::<Vec<_>>(),
        vec![7, 2, 3, 8, 5, 6]
    );

    let mut row0 = a.lanes_mut::<1>(1).unwrap().next().unwrap();
    assert!(row0.has_exclusive_window());
    assert_eq!(*row0.as_mut_slice().unwrap(), [7, 2, 3]);
}

#[test]
#[should_panic(expected = "window is shared with sibling lane/axis views")]
fn interleaved_lane_data_mut_panics() {
    let mut a = leto2([2, 3], vec![0; 6]);
    let mut col0 = a.lanes_mut::<1>(0).unwrap().next().unwrap();
    let _ = col0.data_mut();
}

#[test]
#[should_panic(expected = "window is shared with sibling lane/axis views")]
fn interleaved_lane_into_slice_panics() {
    let mut a = leto2([2, 3], vec![0; 6]);
    let col0 = a.lanes_mut::<1>(0).unwrap().next().unwrap();
    let _ = col0.into_slice();
}

#[test]
#[should_panic(expected = "window is shared with sibling lane/axis views")]
fn interleaved_lane_as_view_panics() {
    let mut a = leto2([2, 3], vec![0; 6]);
    let col0 = a.lanes_mut::<1>(0).unwrap().next().unwrap();
    let _ = col0.as_view();
}
