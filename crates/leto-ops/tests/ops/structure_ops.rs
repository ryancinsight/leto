use leto::{Array, Layout, LetoError, SliceArg, Storage, VecStorage};
use leto_ops::{
    batched_matmul, coordinate_map_inplace, coordinate_map_plan, coordinate_map_plan_inplace,
    cumsum, indexed_fold, indexed_fold_fortran, indexed_map4_inplace, indexed_map_inplace,
    indexed_zip_mut_with, max as reduce_max, min as reduce_min, normal_with_seed, scan_axis,
    uniform_with_seed, zip_fold, zip_mut_with, CumProdOp, ScanDirection,
};

fn arr<const N: usize>(shape: [usize; N], data: Vec<f64>) -> Array<f64, VecStorage<f64>, N> {
    Array::new(Layout::c_contiguous(shape).unwrap(), VecStorage::new(data)).unwrap()
}

#[test]
fn test_batched_matmul_two_batches() {
    // batch 0: [[1,2],[3,4]] x [[1,0],[0,1]] = identity-mul -> same
    // batch 1: [[1,1],[1,1]] x [[2,0],[0,2]] = [[2,2],[2,2]]
    let lhs = arr([2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0]);
    let rhs = arr([2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
    let mut out = arr([2, 2, 2], vec![0.0; 8]);
    batched_matmul(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();
    assert_eq!(
        out.storage().as_slice(),
        &[1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0]
    );
}

/// An empty output matrix (`M == 0`) has no work and must not panic in the
/// disjointness/span computation; it routes to the sequential loop.
#[test]
fn test_batched_matmul_empty_output_matrix_is_noop() {
    let lhs = arr([2, 0, 3], vec![]);
    let rhs = arr([2, 3, 2], (0..12).map(|x| x as f64).collect());
    let mut out = arr([2, 0, 2], vec![]);
    batched_matmul(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();
    assert_eq!(out.storage().as_slice().len(), 0);
}

/// An interleaved-batch output view (batch stride < one matrix's physical span)
/// cannot give parallel tasks disjoint `&mut` slices, so `batched_matmul` routes
/// it through the unconditionally-sound sequential path. Pins the disjointness
/// guard: the result must equal the C-contiguous reference element-for-element.
#[test]
fn test_batched_matmul_interleaved_output_matches_contiguous_reference() {
    let lhs = arr([2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let rhs = arr([2, 2, 2], vec![1.0, 0.0, 2.0, 1.0, 0.0, 3.0, 1.0, 2.0]);

    // Reference: C-contiguous [B, M, N] output (disjoint per-batch blocks).
    let mut out_ref = arr([2, 2, 2], vec![0.0; 8]);
    batched_matmul(&lhs.view(), &rhs.view(), &mut out_ref.view_mut()).unwrap();

    // Interleaved output: permute a [M, N, B] buffer to [B, M, N] so the batch
    // axis carries stride 1 (< the per-matrix span) — `batches_disjoint` is
    // false and the sequential fallback must run.
    let mut base = arr([2, 2, 2], vec![0.0; 8]);
    {
        let mut out = base.transpose_mut([2, 0, 1]).unwrap();
        assert_eq!(
            out.strides()[0],
            1,
            "batch axis must be the interleaved (stride-1) axis"
        );
        batched_matmul(&lhs.view(), &rhs.view(), &mut out).unwrap();
        for b in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(
                        *out.get([b, i, j]).unwrap(),
                        *out_ref.get([b, i, j]).unwrap(),
                        "interleaved-output mismatch at batch {b} ({i},{j})"
                    );
                }
            }
        }
    }
}

#[test]
fn test_batched_matmul_broadcasts_rhs_batch() {
    // rhs batch dim is 1, broadcast across both lhs batches.
    let lhs = arr([2, 1, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = arr([1, 2, 1], vec![1.0, 1.0]);
    let mut out = arr([2, 1, 1], vec![0.0; 2]);
    batched_matmul(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();
    // [1,2]·[1,1]=3 ; [3,4]·[1,1]=7
    assert_eq!(out.storage().as_slice(), &[3.0, 7.0]);
}

#[test]
fn test_batched_matmul_rejects_shape_mismatch() {
    let lhs = arr([2, 2, 3], vec![0.0; 12]);
    let rhs = arr([2, 2, 2], vec![0.0; 8]);
    let mut out = arr([2, 2, 2], vec![0.0; 8]);
    assert!(batched_matmul(&lhs.view(), &rhs.view(), &mut out.view_mut()).is_err());
}

#[test]
fn test_cumsum_forward_axis1() {
    let a = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = cumsum(&a.view(), 1).unwrap();
    assert_eq!(out.storage().as_slice(), &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
}

#[test]
fn test_cumsum_forward_axis0() {
    let a = arr([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = cumsum(&a.view(), 0).unwrap();
    assert_eq!(out.storage().as_slice(), &[1.0, 2.0, 4.0, 6.0, 9.0, 12.0]);
}

#[test]
fn test_scan_reverse_and_cumprod() {
    let a = arr([4], vec![1.0, 2.0, 3.0, 4.0]);
    let suffix = scan_axis::<CumProdOp, _, 1>(&a.view(), 0, ScanDirection::Reverse).unwrap();
    // reverse cumulative product: [24, 24, 12, 4]
    assert_eq!(suffix.storage().as_slice(), &[24.0, 24.0, 12.0, 4.0]);
}

#[test]
fn test_zip_mut_with_fused_multiply_add() {
    // out = out + a * b, three-operand fused update.
    let mut out = arr([2, 2], vec![1.0, 1.0, 1.0, 1.0]);
    let a = arr([2, 2], vec![2.0, 3.0, 4.0, 5.0]);
    let b = arr([2, 2], vec![10.0, 10.0, 10.0, 10.0]);
    zip_mut_with(
        &mut out.view_mut(),
        (&a.view(), &b.view()),
        |o, (&x, &y)| {
            *o += x * y;
        },
    )
    .unwrap();
    assert_eq!(out.storage().as_slice(), &[21.0, 31.0, 41.0, 51.0]);
}

#[test]
fn test_zip_mut_with_strided_input() {
    // a is a transposed (strided) view; traversal must follow logical order.
    let mut out = arr([3, 2], vec![0.0; 6]);
    let a_src = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let a = a_src.transpose([1, 0]).unwrap(); // logical [[1,4],[2,5],[3,6]]
    let b = arr([3, 2], vec![100.0, 100.0, 100.0, 100.0, 100.0, 100.0]);
    zip_mut_with(&mut out.view_mut(), (&a, &b.view()), |o, (&x, &y)| {
        *o = x + y;
    })
    .unwrap();
    assert_eq!(
        out.storage().as_slice(),
        &[101.0, 104.0, 102.0, 105.0, 103.0, 106.0]
    );
}

#[test]
fn test_zip_mut_with_three_inputs() {
    let mut out = arr([2, 2], vec![0.0; 4]);
    let prev = arr([2, 2], vec![1.0, 4.0, 9.0, 16.0]);
    let curr = arr([2, 2], vec![2.0, 5.0, 10.0, 17.0]);
    let next = arr([2, 2], vec![4.0, 8.0, 14.0, 22.0]);

    zip_mut_with(
        &mut out.view_mut(),
        (&prev.view(), &curr.view(), &next.view()),
        |d, (&p0, &p1, &p2)| {
            *d = 2.0f64.mul_add(-p1, p0) + p2;
        },
    )
    .unwrap();

    assert_eq!(out.storage().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_zip_mut_with_three_strided_inputs_follow_logical_order() {
    let mut out = arr([3, 2], vec![0.0; 6]);
    let prev_src = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let curr_src = arr([2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let next_src = arr([2, 3], vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0]);
    let prev = prev_src.transpose([1, 0]).unwrap();
    let curr = curr_src.transpose([1, 0]).unwrap();
    let next = next_src.transpose([1, 0]).unwrap();

    zip_mut_with(
        &mut out.view_mut(),
        (&prev, &curr, &next),
        |d, (&a, &b, &c)| {
            *d = a + b + c;
        },
    )
    .unwrap();

    assert_eq!(
        out.storage().as_slice(),
        &[111.0, 444.0, 222.0, 555.0, 333.0, 666.0]
    );
}

#[test]
fn test_zip_mut_with_five_inputs() {
    let mut out = arr([2, 2], vec![0.0; 4]);
    let a = arr([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = arr([2, 2], vec![10.0, 20.0, 30.0, 40.0]);
    let c = arr([2, 2], vec![100.0, 200.0, 300.0, 400.0]);
    let d = arr([2, 2], vec![1.0, 1.0, 1.0, 1.0]);
    let e = arr([2, 2], vec![2.0, 2.0, 2.0, 2.0]);

    zip_mut_with(
        &mut out.view_mut(),
        (&a.view(), &b.view(), &c.view(), &d.view(), &e.view()),
        |o, (&av, &bv, &cv, &dv, &ev)| *o = av + bv - cv + dv * ev,
    )
    .unwrap();

    assert_eq!(out.storage().as_slice(), &[-87.0, -176.0, -265.0, -354.0]);
}

#[test]
fn test_zip_mut_with_five_strided_inputs_follow_logical_order() {
    let mut out = arr([3, 2], vec![0.0; 6]);
    let a_src = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b_src = arr([2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let c_src = arr([2, 3], vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0]);
    let d_src = arr([2, 3], vec![1.0; 6]);
    let e_src = arr([2, 3], vec![2.0; 6]);
    let a = a_src.transpose([1, 0]).unwrap();
    let b = b_src.transpose([1, 0]).unwrap();
    let c = c_src.transpose([1, 0]).unwrap();
    let d = d_src.transpose([1, 0]).unwrap();
    let e = e_src.transpose([1, 0]).unwrap();

    zip_mut_with(
        &mut out.view_mut(),
        (&a, &b, &c, &d, &e),
        |o, (&av, &bv, &cv, &dv, &ev)| {
            *o = av + bv - cv + dv * ev;
        },
    )
    .unwrap();

    assert_eq!(
        out.storage().as_slice(),
        &[-87.0, -354.0, -176.0, -443.0, -265.0, -532.0]
    );
}

#[test]
fn test_zip_mut_with_preserves_heterogeneous_source_types() {
    let mut out = arr([2, 2], vec![0.0; 4]);
    let integer = Array::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();
    let scale = arr([2, 2], vec![0.5, 1.5, 2.5, 3.5]);

    zip_mut_with(
        &mut out.view_mut(),
        (&integer.view(), &scale.view()),
        |value, (&integer, &scale)| *value = f64::from(integer) + scale,
    )
    .unwrap();

    assert_eq!(out.storage().as_slice(), &[1.5, 3.5, 5.5, 7.5]);
}

#[test]
fn test_indexed_zip_mut_with_uses_logical_index() {
    let mut out = arr([2, 2], vec![0.0; 4]);
    let a = arr([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = arr([2, 2], vec![10.0, 20.0, 30.0, 40.0]);
    let c = arr([2, 2], vec![100.0, 200.0, 300.0, 400.0]);
    let d = arr([2, 2], vec![1000.0, 2000.0, 3000.0, 4000.0]);

    indexed_zip_mut_with(
        &mut out.view_mut(),
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

#[test]
fn test_zip_fold_contiguous_inputs() {
    let lhs = arr([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = arr([2, 2], vec![10.0, 20.0, 30.0, 40.0]);

    let dot = zip_fold(&lhs.view(), &rhs.view(), 0.0, |acc, &x, &y| acc + x * y).unwrap();

    assert_eq!(dot, 300.0);
}

#[test]
fn test_zip_fold_strided_inputs_follow_logical_order() {
    let lhs_src = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs_src = arr([2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let lhs = lhs_src.transpose([1, 0]).unwrap();
    let rhs = rhs_src.transpose([1, 0]).unwrap();

    let weighted = zip_fold(&lhs, &rhs, 0.0, |acc, &x, &y| acc + x * y).unwrap();

    assert_eq!(weighted, 910.0);
}

#[test]
fn test_zip_fold_rejects_shape_mismatch() {
    let lhs = arr([2, 2], vec![0.0; 4]);
    let rhs = arr([1, 4], vec![0.0; 4]);

    assert!(zip_fold(&lhs.view(), &rhs.view(), 0.0, |acc, &x, &y| acc + x + y).is_err());
}

#[test]
fn test_indexed_fold_uses_logical_index() {
    let input = arr([2, 3], vec![1.0, -4.0, 2.0, 8.0, -7.0, 3.0]);

    let peak = indexed_fold(
        &input.view(),
        (0.0_f64, [0usize; 2]),
        |(best, best_index), index, &value| {
            let magnitude = value.abs();
            if magnitude > best {
                (magnitude, index)
            } else {
                (best, best_index)
            }
        },
    )
    .unwrap();

    assert_eq!(peak, (8.0, [1, 0]));
}

#[test]
fn test_indexed_fold_strided_input_follows_logical_order() {
    let input = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 9.0, 6.0]);
    let transposed = input.transpose([1, 0]).unwrap();

    let peak = indexed_fold(
        &transposed,
        (0.0_f64, [0usize; 2]),
        |(best, best_index), index, &value| {
            if value > best {
                (value, index)
            } else {
                (best, best_index)
            }
        },
    )
    .unwrap();

    assert_eq!(peak, (9.0, [1, 1]));
}

#[test]
fn test_indexed_fold_fortran_uses_column_major_logical_order() {
    let input = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let visited = indexed_fold_fortran(
        &input.view(),
        Vec::<([usize; 2], f64)>::new(),
        |mut acc, index, &value| {
            acc.push((index, value));
            acc
        },
    )
    .unwrap();

    assert_eq!(
        visited,
        vec![
            ([0, 0], 1.0),
            ([1, 0], 4.0),
            ([0, 1], 2.0),
            ([1, 1], 5.0),
            ([0, 2], 3.0),
            ([1, 2], 6.0),
        ]
    );
}

#[test]
fn test_reduce_min_max_contiguous_inputs() {
    let input = arr([2, 3], vec![3.0, -7.0, 2.0, 11.0, 0.5, -1.0]);

    assert_eq!(reduce_min(&input.view()).unwrap(), -7.0);
    assert_eq!(reduce_max(&input.view()).unwrap(), 11.0);
}

#[test]
fn test_reduce_min_max_sliced_inputs_follow_logical_view() {
    let input = arr(
        [3, 4],
        vec![
            100.0, -2.0, 9.0, 5.0, 7.0, -11.0, 3.0, 42.0, -99.0, 6.0, -4.0, 10.0,
        ],
    );
    let view = input
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(Some(1), Some(4), 2)])
        .unwrap();

    assert_eq!(reduce_min(&view).unwrap(), -11.0);
    assert_eq!(reduce_max(&view).unwrap(), 42.0);
}

#[test]
fn test_reduce_min_max_reject_empty_inputs() {
    let empty = arr([0], Vec::<f64>::new());

    let min_err = reduce_min(&empty.view()).unwrap_err();
    let max_err = reduce_max(&empty.view()).unwrap_err();

    assert_eq!(
        min_err,
        LetoError::StorageError {
            reason: "all-elements reduction requires a non-empty input".to_string()
        }
    );
    assert_eq!(
        max_err,
        LetoError::StorageError {
            reason: "all-elements reduction requires a non-empty input".to_string()
        }
    );
}

#[test]
fn test_uniform_is_deterministic_and_in_range() {
    let a = uniform_with_seed([1000], -2.0, 5.0, 42).unwrap();
    let b = uniform_with_seed([1000], -2.0, 5.0, 42).unwrap();
    assert_eq!(a.storage().as_slice(), b.storage().as_slice());
    for &v in a.storage().as_slice() {
        assert!((-2.0..5.0).contains(&v), "out of range: {v}");
    }
    // Different seed yields a different stream.
    let c = uniform_with_seed([1000], -2.0, 5.0, 43).unwrap();
    assert_ne!(a.storage().as_slice(), c.storage().as_slice());
}

#[test]
fn test_uniform_mean_matches_closed_form() {
    let n = 100_000usize;
    let a = uniform_with_seed([n], 0.0, 1.0, 7).unwrap();
    let mean: f64 = a.storage().as_slice().iter().sum::<f64>() / n as f64;
    // Closed-form mean of U(0,1) is 0.5; sampling error well under 0.02 at this n.
    assert!((mean - 0.5).abs() < 0.02, "mean {mean}");
}

#[test]
fn test_normal_mean_and_std_match_closed_form() {
    let n = 100_000usize;
    let a = normal_with_seed([n], 1.0, 2.0, 11).unwrap();
    let data = a.storage().as_slice();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    let std = var.sqrt();
    assert!((mean - 1.0).abs() < 0.05, "mean {mean}");
    assert!((std - 2.0).abs() < 0.05, "std {std}");
}
