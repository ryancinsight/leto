//! Batched matmul, cumsum and scan structure.

use super::*;

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
