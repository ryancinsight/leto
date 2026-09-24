//! Allocation census of `SymmetricEigenWorkspace` reuse, for every input
//! layout: after one decomposition at an order, further decompositions at that
//! order allocate nothing.

use super::measured;
use leto::{Array2, SliceArg, Storage};
use leto_ops::SymmetricEigenWorkspace;

#[test]
fn symmetric_eigen_workspace_reuse_allocates_nothing_for_any_layout() {
    let n = 12;
    let values: Vec<f64> = (0..4 * n * n)
        .map(|i| {
            let (row, col): (usize, usize) = (i / (2 * n), i % (2 * n));
            1.0 / (1.0 + row.abs_diff(col) as f64) + if row == col { 3.0 } else { 0.0 }
        })
        .collect();
    let big = Array2::from_shape_vec([2 * n, 2 * n], values).expect("invariant: (2n)² entries");
    let contiguous = Array2::from_shape_vec([n, n], big.storage().as_slice()[..n * n].to_vec())
        .expect("invariant: n² entries");
    let strided = big
        .view()
        .slice_with::<2>(&[
            SliceArg::range(Some(0), None, 2),
            SliceArg::range(Some(0), None, 2),
        ])
        .expect("invariant: every other row and column of a 2n × 2n matrix");
    let transposed = contiguous
        .view()
        .transpose([1, 0])
        .expect("invariant: a rank-2 permutation");

    let mut workspace = SymmetricEigenWorkspace::new();
    workspace
        .decompose(&contiguous.view())
        .expect("invariant: a finite square matrix decomposes");
    for (layout, view) in [
        ("contiguous", contiguous.view()),
        ("strided", strided),
        ("transposed", transposed),
    ] {
        let (result, allocations, reallocations) = measured(|| workspace.decompose(&view));
        result.expect("invariant: a finite square matrix decomposes");
        assert_eq!(
            (allocations, reallocations),
            (0, 0),
            "{layout} layout allocated"
        );
    }
}
