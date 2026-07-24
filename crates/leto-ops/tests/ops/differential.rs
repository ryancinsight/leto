use leto::{concat, stack, Array, SliceArg, Storage};
use leto_ops::{
    add, batched_matmul, cumsum, mapv, matmul, max_axis, mean_axis, min_axis, scalar_map, sum,
    sum_axis, unary_map, AddOp, ExpOp, SqrtOp,
};

#[track_caller]
fn assert_close_slice(lhs: &[f32], rhs: &[f32]) {
    assert_eq!(lhs.len(), rhs.len());
    for (left, right) in lhs.iter().zip(rhs.iter()) {
        assert!(
            (*left - *right).abs() <= 1.0e-5,
            "left {left} differs from right {right}"
        );
    }
}

/// Reference matmul: C[i,j] = sum_k A[i,k] * B[k,j] (replaces external oracle).
fn ref_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Reference elementwise add with broadcasting for [a,1] + [1,b] -> [a,b].
fn ref_broadcast_add(a: &[f32], b: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = a[i] + b[j];
        }
    }
    out
}

/// Reference mapv: apply f to each element.
fn ref_mapv<T, F>(data: &[T], f: F) -> Vec<T>
where
    T: Copy,
    F: Fn(T) -> T,
{
    data.iter().map(|&x| f(x)).collect()
}

/// Reference sum along axis (0 = columns reduce rows, 1 = rows reduce cols).
fn ref_sum_axis(data: &[f32], rows: usize, cols: usize, axis: usize) -> Vec<f32> {
    if axis == 0 {
        (0..cols)
            .map(|c| (0..rows).map(|r| data[r * cols + c]).sum())
            .collect()
    } else {
        (0..rows)
            .map(|r| (0..cols).map(|c| data[r * cols + c]).sum())
            .collect()
    }
}

/// Reference mean along axis.
fn ref_mean_axis(data: &[f32], rows: usize, cols: usize, axis: usize) -> Vec<f32> {
    if axis == 0 {
        (0..cols)
            .map(|c| (0..rows).map(|r| data[r * cols + c]).sum::<f32>() / rows as f32)
            .collect()
    } else {
        (0..rows)
            .map(|r| (0..cols).map(|c| data[r * cols + c]).sum::<f32>() / cols as f32)
            .collect()
    }
}

/// Reference min along axis.
fn ref_min_axis(data: &[f32], rows: usize, cols: usize, axis: usize) -> Vec<f32> {
    if axis == 0 {
        (0..cols)
            .map(|c| {
                (0..rows)
                    .map(|r| data[r * cols + c])
                    .fold(f32::INFINITY, f32::min)
            })
            .collect()
    } else {
        (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| data[r * cols + c])
                    .fold(f32::INFINITY, f32::min)
            })
            .collect()
    }
}

/// Reference max along axis.
fn ref_max_axis(data: &[f32], rows: usize, cols: usize, axis: usize) -> Vec<f32> {
    if axis == 0 {
        (0..cols)
            .map(|c| {
                (0..rows)
                    .map(|r| data[r * cols + c])
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect()
    } else {
        (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| data[r * cols + c])
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect()
    }
}

/// Reference reverse last axis: data[i, j] -> data[i, cols-1-j].
fn ref_reverse_last_axis(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = data[i * cols + (cols - 1 - j)];
        }
    }
    out
}

/// Reference concat along axis 0.
fn ref_concat_axis0(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = a.to_vec();
    out.extend_from_slice(b);
    out
}

/// Reference stack along new axis 1: [a_len, 2].
fn ref_stack_axis1(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len() * 2);
    for (x, y) in a.iter().zip(b.iter()) {
        out.push(*x);
        out.push(*y);
    }
    out
}

#[test]
fn test_matmul_differential_matches_reference_contiguous() {
    let lhs_values = vec![1.0f32, -2.0, 3.0, 4.5, 0.25, -6.0];
    let rhs_values = vec![7.0f32, 8.0, -9.0, 10.0, 11.0, -12.0];
    let lhs = Array::from_shape_vec([2, 3], lhs_values.clone()).unwrap();
    let rhs = Array::from_shape_vec([3, 2], rhs_values.clone()).unwrap();
    let mut out = Array::zeros([2, 2]);

    matmul(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();

    let expected = ref_matmul(&lhs_values, &rhs_values, 2, 3, 2);
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn test_matmul_differential_matches_reference_transposed_views() {
    let lhs_base_values = vec![1.0f32, 4.5, -2.0, 0.25, 3.0, -6.0];
    let rhs_base_values = vec![7.0f32, -9.0, 11.0, 8.0, 10.0, -12.0];
    let lhs_base = Array::from_shape_vec([3, 2], lhs_base_values.clone()).unwrap();
    let rhs_base = Array::from_shape_vec([2, 3], rhs_base_values.clone()).unwrap();
    let lhs = lhs_base.transpose([1, 0]).unwrap();
    let rhs = rhs_base.transpose([1, 0]).unwrap();
    let mut out = Array::zeros([2, 2]);

    matmul(&lhs, &rhs, &mut out.view_mut()).unwrap();

    // Transposed: lhs_base is [3,2], transposed to [2,3]; rhs_base is [2,3], transposed to [3,2].
    // Effective matmul: [2,3] x [3,2] = [2,2].
    // lhs_base values in row-major [3,2]: [[1,4.5],[-2,0.25],[3,-6]]
    // transposed [2,3]: [[1,-2,3],[4.5,0.25,-6]]
    // rhs_base values in row-major [2,3]: [[7,-9,11],[8,10,-12]]
    // transposed [3,2]: [[7,8],[-9,10],[11,-12]]
    // matmul: [2,3] x [3,2] = [2,2]
    let expected = ref_matmul(
        &[1.0, -2.0, 3.0, 4.5, 0.25, -6.0],
        &[7.0, 8.0, -9.0, 10.0, 11.0, -12.0],
        2,
        3,
        2,
    );
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn test_map_differential_matches_reference_contiguous_and_transposed_views() {
    let values = vec![1.0f32, -2.0, 3.5, 4.25, -5.5, 6.75];
    let input = Array::from_shape_vec([2, 3], values.clone()).unwrap();

    let mapped = mapv(&input.view(), |value| value.mul_add(value, -1.0)).unwrap();

    let expected = ref_mapv(&values, |value| value.mul_add(value, -1.0));
    assert_close_slice(mapped.storage().as_slice(), &expected);

    let transposed = input.transpose([1, 0]).unwrap();
    let mapped_transposed = mapv(&transposed, |value| value + 0.5).unwrap();
    // Transposed [3,2] of [2,3] row-major: row r of transposed = col r of original.
    let transposed_values: Vec<f32> = (0..3)
        .flat_map(|r| {
            let values = &values;
            (0..2).map(move |c| values[c * 3 + r])
        })
        .collect();
    let expected_transposed = ref_mapv(&transposed_values, |value| value + 0.5);

    assert_eq!(mapped_transposed.shape(), [3, 2]);
    assert_close_slice(mapped_transposed.storage().as_slice(), &expected_transposed);
}

#[test]
fn test_binary_broadcast_differential_matches_reference() {
    let lhs_values = vec![1.0f32, 10.0];
    let rhs_values = vec![2.0f32, -3.0, 4.5];
    let lhs = Array::from_shape_vec([2, 1], lhs_values.clone()).unwrap();
    let rhs = Array::from_shape_vec([1, 3], rhs_values.clone()).unwrap();
    let mut out = Array::zeros([2, 3]);

    add(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();

    let expected = ref_broadcast_add(&lhs_values, &rhs_values, 2, 3);
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn test_row_walk_maps_negative_last_axis_stride() {
    let values = vec![1.0f32, -2.0, 3.5, 4.25, -5.5, 6.75, 7.5, -8.25];
    let input = Array::from_shape_vec([2, 4], values.clone()).unwrap();
    let reversed = input
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();

    let mapped = mapv(&reversed, |value| value.mul_add(2.0, 1.0)).unwrap();

    let reversed_values = ref_reverse_last_axis(&values, 2, 4);
    let expected = ref_mapv(&reversed_values, |value| value.mul_add(2.0, 1.0));
    assert_eq!(mapped.shape(), [2, 4]);
    assert_close_slice(mapped.storage().as_slice(), &expected);
}

#[test]
fn test_row_walk_binary_map_negative_last_axis_stride() {
    let lhs_values = vec![1.0f32, -2.0, 3.5, 4.25, -5.5, 6.75, 7.5, -8.25];
    let rhs_values = vec![0.5f32, 1.5, -3.0, 2.0, 4.0, -1.0, 6.0, 8.0];
    let lhs_base = Array::from_shape_vec([2, 4], lhs_values.clone()).unwrap();
    let rhs_base = Array::from_shape_vec([2, 4], rhs_values.clone()).unwrap();
    let lhs = lhs_base
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();
    let rhs = rhs_base
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();
    let mut out = Array::zeros([2, 4]);

    add(&lhs, &rhs, &mut out.view_mut()).unwrap();

    let lhs_rev = ref_reverse_last_axis(&lhs_values, 2, 4);
    let rhs_rev = ref_reverse_last_axis(&rhs_values, 2, 4);
    let expected: Vec<f32> = lhs_rev.iter().zip(&rhs_rev).map(|(a, b)| a + b).collect();
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn test_row_walk_sum_negative_last_axis_stride() {
    let values = vec![1.0f32, -2.0, 3.5, 4.25, -5.5, 6.75, 7.5, -8.25];
    let input = Array::from_shape_vec([2, 4], values.clone()).unwrap();
    let reversed = input
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();

    let actual = sum(&reversed);

    let reversed_values = ref_reverse_last_axis(&values, 2, 4);
    let expected: f32 = reversed_values.iter().sum();
    assert!(
        (actual - expected).abs() <= 1.0e-5,
        "actual {actual} expected {expected}"
    );
}

#[test]
fn test_axis_reductions_differential_match_reference_keepdim() {
    let values = vec![1.0f32, -2.0, 3.5, 4.25, -5.5, 6.75];
    let input = Array::from_shape_vec([2, 3], values.clone()).unwrap();

    let row_sum = sum_axis(&input.view(), 1).unwrap();
    let row_mean = mean_axis(&input.view(), 1).unwrap();
    let row_min = min_axis(&input.view(), 1).unwrap();
    let row_max = max_axis(&input.view(), 1).unwrap();

    let expected_row_sum = ref_sum_axis(&values, 2, 3, 1);
    let expected_row_mean = ref_mean_axis(&values, 2, 3, 1);
    let expected_row_min = ref_min_axis(&values, 2, 3, 1);
    let expected_row_max = ref_max_axis(&values, 2, 3, 1);

    assert_eq!(row_sum.shape(), [2, 1]);
    assert_close_slice(row_sum.storage().as_slice(), &expected_row_sum);
    assert_close_slice(row_mean.storage().as_slice(), &expected_row_mean);
    assert_close_slice(row_min.storage().as_slice(), &expected_row_min);
    assert_close_slice(row_max.storage().as_slice(), &expected_row_max);

    let col_sum = sum_axis(&input.view(), 0).unwrap();
    let expected_col_sum = ref_sum_axis(&values, 2, 3, 0);
    assert_eq!(col_sum.shape(), [1, 3]);
    assert_close_slice(col_sum.storage().as_slice(), &expected_col_sum);
}

#[test]
fn test_unary_map_differential_matches_reference() {
    let values = vec![0.25f32, 1.0, 2.5, 4.0, 0.5, 9.0];
    let input = Array::from_shape_vec([2, 3], values.clone()).unwrap();

    let exp = unary_map(ExpOp, &input.view()).unwrap();
    let expected_exp = ref_mapv(&values, f32::exp);
    assert_close_slice(exp.storage().as_slice(), &expected_exp);

    let sqrt = unary_map(SqrtOp, &input.view()).unwrap();
    let expected_sqrt = ref_mapv(&values, f32::sqrt);
    assert_close_slice(sqrt.storage().as_slice(), &expected_sqrt);
}

#[test]
fn test_scalar_map_differential_matches_reference() {
    let values = vec![1.0f32, -2.0, 3.5, 4.25];
    let input = Array::from_shape_vec([2, 2], values.clone()).unwrap();

    let shifted = scalar_map::<AddOp, _, 2>(&input.view(), 10.0).unwrap();
    let expected = ref_mapv(&values, |v| v + 10.0);
    assert_close_slice(shifted.storage().as_slice(), &expected);
}

#[test]
fn test_concat_differential_matches_reference() {
    let a_values = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_values = vec![5.0f32, 6.0];
    let a = Array::from_shape_vec([2, 2], a_values.clone()).unwrap();
    let b = Array::from_shape_vec([1, 2], b_values.clone()).unwrap();

    let out = concat(&[a.view(), b.view()], 0).unwrap();

    let expected = ref_concat_axis0(&a_values, &b_values);
    assert_eq!(out.shape(), [3, 2]);
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn test_stack_differential_matches_reference() {
    let a_values = vec![1.0f32, 2.0, 3.0];
    let b_values = vec![4.0f32, 5.0, 6.0];
    let a = Array::from_shape_vec([3], a_values.clone()).unwrap();
    let b = Array::from_shape_vec([3], b_values.clone()).unwrap();

    let out = stack::<f32, 1, 2>(&[a.view(), b.view()], 1).unwrap();

    let expected = ref_stack_axis1(&a_values, &b_values);
    assert_eq!(out.shape(), [3, 2]);
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn test_batched_matmul_differential_matches_reference_per_batch() {
    let lhs_values: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let rhs_values: Vec<f32> = (1..=12).map(|x| (x as f32) * 0.5).collect();
    // [2, 2, 3] x [2, 3, 2] -> [2, 2, 2]
    let lhs = Array::from_shape_vec([2, 2, 3], lhs_values.clone()).unwrap();
    let rhs = Array::from_shape_vec([2, 3, 2], rhs_values.clone()).unwrap();
    let mut out = Array::zeros([2, 2, 2]);

    batched_matmul(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();

    let mut expected = Vec::with_capacity(8);
    for b in 0..2 {
        let a_slice = &lhs_values[b * 6..(b + 1) * 6];
        let b_slice = &rhs_values[b * 6..(b + 1) * 6];
        expected.extend(ref_matmul(a_slice, b_slice, 2, 3, 2));
    }
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn test_cumsum_differential_matches_reference_accumulate() {
    let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Array::from_shape_vec([2, 3], values.clone()).unwrap();

    let out = cumsum(&input.view(), 1).unwrap();

    // Reference: running sum along axis 1 of each row.
    let mut expected = vec![0.0f32; 6];
    for row in 0..2 {
        let mut acc = 0.0f32;
        for col in 0..3 {
            acc += values[row * 3 + col];
            expected[row * 3 + col] = acc;
        }
    }
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn test_axis_reductions_differential_match_reference_transposed_views() {
    let values = vec![1.0f32, 4.25, -2.0, -5.5, 3.5, 6.75];
    let input = Array::from_shape_vec([3, 2], values.clone()).unwrap();
    let transposed = input.transpose([1, 0]).unwrap();
    let reduced = sum_axis(&transposed, 1).unwrap();

    // transposed is [2,3] (row-major), sum along axis 1 -> [2,1]
    // The transposed data in row-major: each row of transposed is a column of input.
    // transposed row 0 = input col 0 = [1.0, -2.0, 3.5], sum = 2.5
    // transposed row 1 = input col 1 = [4.25, -5.5, 6.75], sum = 5.5
    let expected = vec![2.5f32, 5.5];
    assert_eq!(reduced.shape(), [2, 1]);
    assert_close_slice(reduced.storage().as_slice(), &expected);
}
