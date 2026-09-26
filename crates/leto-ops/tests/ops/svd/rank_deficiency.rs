//! Exactly rank-deficient SVD input, at every precision, plus the pinv
//! Overflow regression. Split out of `super` (`svd.rs`) to keep each file
//! near the 500-line target.
#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::super::format::scale;
use eunomia::RealField;
use leto::{Array2, Storage};
use leto_ops::{pinv, singular_values, svd_decompose, RealScalar};

// ── Exactly rank-deficient input, at every precision ────────────────────────
//
// Rank deficiency has two structurally different forms, and only one of them is
// covered by testing a *near*-deficient matrix. When the deficiency is exact, the
// Householder bidiagonalization can produce an **exact zero on the diagonal** of
// `B`; a shifted QR step cannot deflate that (the implicit `BᵀB` is singular, the
// Wilkinson shift takes its nonzero eigenvalue, and the sweep reaches a fixed
// point at `d = 0`, `e ≠ 0` that the deflation test never accepts). It needs the
// zero-diagonal chase instead.
//
// Whether the zero comes out exact depends on the rounding of the
// bidiagonalization, so it depends on the precision: `[[1,2],[2,4],[3,6]]` gives
// an exact `d[1] = 0` at `f32` and a `8.9e-8` residue at `f64`. Testing rank
// deficiency at `f64` and genericity only on full-rank input therefore leaves the
// case uncovered — which is how a non-convergence error reached the GPU
// consumers. These cases run every shape at **both** precisions.

/// Backward-error bound for a Golub–Reinsch SVD of an `m × n` matrix:
/// `A + E = Û Σ̂ V̂ᵀ` with `‖E‖₂ ≤ p·ε·‖A‖₂` and `‖ÛᵀÛ − I‖₂ ≤ p·ε`, `p` modest in
/// the dimensions (Golub & Van Loan, *Matrix Computations* 4th ed., §8.6.3: the
/// factorization is a product of Householder reflectors and plane rotations, each
/// contributing `O(ε)`, with `O(max(m,n))` of them touching any one entry).
/// `p = 8·max(m, n)` throughout. Weyl's theorem carries `‖E‖₂` to each
/// `|σ̂ᵢ − σᵢ|`, including the `σᵢ` whose exact value is `0` — so the same bound
/// is what a rank-deficient direction must satisfy.
///
/// Returns `(absolute, relative)`: the `‖A‖₂ ≈ σ₁`-scaled bound for singular
/// values and reconstruction, and the bare relative bound for orthonormality.
fn error_bounds<T: RealScalar + RealField>(rows: usize, cols: usize, norm: f64) -> (f64, f64) {
    let order = u32::try_from(rows.max(cols)).expect("invariant: test dimensions fit in u32");
    let relative = 8.0 * f64::from(order) * <T as RealField>::EPSILON.to_f64();
    (relative * norm, relative)
}

/// Decompose `entries` (`rows × cols`, exactly representable in binary) at
/// precision `T` and assert full value semantics against the analytic `expected`
/// spectrum: singular values, reconstruction `A = U Σ Vᵀ`, and orthonormal
/// columns of both `U` and `V`.
///
/// Orthonormality is asserted on **both** factors and at deficient rank
/// specifically: the deleted one-sided Jacobi path returned a non-orthonormal `U`
/// exactly here (`‖UᵀU − I‖ = 1.0` on a null-space column), and that was a
/// load-bearing reason for keeping this path. Nothing may regress it.
fn assert_rank_deficient_svd<T: RealScalar + RealField>(
    rows: usize,
    cols: usize,
    entries: &[f64],
    expected: &[f64],
) {
    let values: Vec<T> = entries.iter().map(|&x| T::from_f64(x)).collect();
    let matrix = Array2::from_shape_vec([rows, cols], values).unwrap();
    let svd = svd_decompose(&matrix.view()).unwrap();
    let rank = rows.min(cols);
    let (absolute, relative) = error_bounds::<T>(rows, cols, expected[0]);

    assert_eq!(svd.singular_values.len(), rank);
    assert_eq!(expected.len(), rank);
    let sigma: Vec<f64> = svd.singular_values.iter().map(|x| x.to_f64()).collect();
    for window in sigma.windows(2) {
        assert!(
            window[0] >= window[1],
            "σ must be descending, got {sigma:?}"
        );
    }
    for (got, want) in sigma.iter().zip(expected) {
        assert!(
            (got - want).abs() <= absolute,
            "{rows}x{cols}: σ {got} vs {want}, bound {absolute:e}"
        );
    }

    let u = svd.left_singular_vectors.storage().as_slice();
    let v = svd.right_singular_vectors.storage().as_slice();
    for row in 0..rows {
        for col in 0..cols {
            let value: f64 = (0..rank)
                .map(|i| u[row * rank + i].to_f64() * sigma[i] * v[col * rank + i].to_f64())
                .sum();
            let target = entries[row * cols + col];
            assert!(
                (value - target).abs() <= absolute,
                "{rows}x{cols}: A[{row}][{col}] reconstructs as {value} not {target}"
            );
        }
    }

    for (name, factor, height) in [("U", u, rows), ("V", v, cols)] {
        for a in 0..rank {
            for b in 0..rank {
                let dot: f64 = (0..height)
                    .map(|r| factor[r * rank + a].to_f64() * factor[r * rank + b].to_f64())
                    .sum();
                let target = f64::from(u8::from(a == b));
                assert!(
                    (dot - target).abs() <= relative,
                    "{rows}x{cols}: {name}ᵀ{name}[{a}][{b}] = {dot} not {target}"
                );
            }
        }
    }
}

/// Exactly rank-1 tall input — the downstream reproducer.
///
/// `A = [1,2,3]ᵀ [1,2]`: column 2 is exactly twice column 1, so the rank is 1 and
/// `σ = (‖[1,2,3]‖·‖[1,2]‖, 0) = (√70, 0)`. At `f32` its bidiagonal factor is
/// `d = [−3.7416573, 0]`, `e = [7.4833145]` — the exact zero diagonal.
#[test]
fn svd_decompose_reveals_exact_rank_deficiency_tall() {
    let entries = [1.0, 2.0, 2.0, 4.0, 3.0, 6.0];
    let expected = [70.0f64.sqrt(), 0.0];
    assert_rank_deficient_svd::<f32>(3, 2, &entries, &expected);
    assert_rank_deficient_svd::<f64>(3, 2, &entries, &expected);
}

/// Exactly rank-1 wide input: the transpose of the tall reproducer, which takes
/// the `m < n` branch (SVD of `Aᵀ` with `U` and `V` swapped), so the zero lands in
/// the other factor.
#[test]
fn svd_decompose_reveals_exact_rank_deficiency_wide() {
    let entries = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0];
    let expected = [70.0f64.sqrt(), 0.0];
    assert_rank_deficient_svd::<f32>(2, 3, &entries, &expected);
    assert_rank_deficient_svd::<f64>(2, 3, &entries, &expected);
}

/// Exactly rank-2 square input, built as a sum of two orthogonal outer products
/// so the spectrum is analytic: `A = u₁v₁ᵀ + u₂v₂ᵀ` with `u₁ ⊥ u₂`, `v₁ ⊥ v₂`
/// gives `σ = (‖u₂‖‖v₂‖, ‖u₁‖‖v₁‖, 0) = (2√6, √6, 0)` for
/// `u₁ = (1,1,1), v₁ = (1,0,−1), u₂ = (1,0,−1), v₂ = (2,2,2)`.
#[test]
fn svd_decompose_reveals_exact_rank_deficiency_square() {
    let entries = [3.0, 2.0, 1.0, 1.0, 0.0, -1.0, -1.0, -2.0, -3.0];
    let expected = [2.0 * 6.0f64.sqrt(), 6.0f64.sqrt(), 0.0];
    assert_rank_deficient_svd::<f32>(3, 3, &entries, &expected);
    assert_rank_deficient_svd::<f64>(3, 3, &entries, &expected);
}

/// Rank 2 of 4: **two** deficient directions, so the iteration must chase more
/// than one zero out of the same matrix, and the trailing block is entirely zero.
/// `u₁ = (1,1,1,1), v₁ = (1,0,1,0), u₂ = (1,−1,1,−1), v₂ = (0,2,0,2)` (mutually
/// orthogonal) give `σ = (4√2, 2√2, 0, 0)`.
#[test]
fn svd_decompose_reveals_rank_two_of_four() {
    let entries = [
        1.0, 2.0, 1.0, 2.0, //
        1.0, -2.0, 1.0, -2.0, //
        1.0, 2.0, 1.0, 2.0, //
        1.0, -2.0, 1.0, -2.0,
    ];
    let expected = [4.0 * 2.0f64.sqrt(), 2.0 * 2.0f64.sqrt(), 0.0, 0.0];
    assert_rank_deficient_svd::<f32>(4, 4, &entries, &expected);
    assert_rank_deficient_svd::<f64>(4, 4, &entries, &expected);
}

/// Rank 2 of 3 in a non-square tall shape (`4 × 3`) and its wide transpose, so
/// the deficient case is exercised where `m ≠ n` on both branches.
/// `u₁ = (1,1,1,1), v₁ = (1,0,1)` and `u₂ = (1,−1,1,−1), v₂ = (0,2,0)` give
/// `σ = (4, 2√2, 0)`.
#[test]
fn svd_decompose_reveals_rank_deficiency_in_rectangular_shapes() {
    let tall = [1.0, 2.0, 1.0, 1.0, -2.0, 1.0, 1.0, 2.0, 1.0, 1.0, -2.0, 1.0];
    let wide = [1.0, 1.0, 1.0, 1.0, 2.0, -2.0, 2.0, -2.0, 1.0, 1.0, 1.0, 1.0];
    let expected = [4.0, 2.0 * 2.0f64.sqrt(), 0.0];
    assert_rank_deficient_svd::<f32>(4, 3, &tall, &expected);
    assert_rank_deficient_svd::<f64>(4, 3, &tall, &expected);
    assert_rank_deficient_svd::<f32>(3, 4, &wide, &expected);
    assert_rank_deficient_svd::<f64>(3, 4, &wide, &expected);
}

/// The values-only path (`bidiagonal_diag_colmajor` + non-accumulating iteration)
/// and the full SVD must agree on exactly rank-deficient input at both
/// precisions. They reach the zero diagonal by different roundings — the
/// reproducer's `d[1]` is `8.9e-8` on the values-only path and exactly `0` on the
/// accumulating one — so agreement here is a genuine differential check on the
/// chase rather than a restatement of one path.
#[test]
fn singular_values_agree_with_full_svd_at_exact_rank_deficiency() {
    fn compare<T: RealScalar + RealField>(rows: usize, cols: usize, entries: &[f64], norm: f64) {
        let values: Vec<T> = entries.iter().map(|&x| T::from_f64(x)).collect();
        let a = Array2::from_shape_vec([rows, cols], values).unwrap();
        let full = svd_decompose(&a.view()).unwrap().singular_values;
        let only = singular_values(&a.view()).unwrap();
        let (absolute, _) = error_bounds::<T>(rows, cols, norm);
        assert_eq!(full.len(), only.len());
        for (l, r) in full.iter().zip(only.iter()) {
            assert!(
                (l.to_f64() - r.to_f64()).abs() <= absolute,
                "{rows}x{cols}: {} vs {}",
                l.to_f64(),
                r.to_f64()
            );
        }
    }

    let cases: [(usize, usize, Vec<f64>, f64); 3] = [
        (3, 2, vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], 70.0f64.sqrt()),
        (2, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0], 70.0f64.sqrt()),
        (
            4,
            4,
            vec![
                1.0, 2.0, 1.0, 2.0, 1.0, -2.0, 1.0, -2.0, 1.0, 2.0, 1.0, 2.0, 1.0, -2.0, 1.0, -2.0,
            ],
            4.0 * 2.0f64.sqrt(),
        ),
    ];
    for (rows, cols, entries, norm) in cases {
        compare::<f32>(rows, cols, &entries, norm);
        compare::<f64>(rows, cols, &entries, norm);
    }
}

/// Finding F: `pinv` computed `1/sigma` for every retained singular value
/// with no range check, returning a wrong `Ok([inf, NaN, ...])` when `sigma`
/// is small enough that its reciprocal exceeds the scalar's range while
/// still above the rank cutoff. Both the F16 and f64 cases below hit exactly
/// that: the matrix is full rank (every singular value above the `1e-12`
/// relative cutoff), but the smallest one's reciprocal overflows.
#[test]
fn pinv_reports_overflow_for_a_non_finite_reciprocal() {
    use eunomia::F16;
    use leto::LetoError;

    // GENERAL from tests/ops/scale_range.rs, scaled so its smallest singular
    // value (~0.146, see `singular_values_match_diagonal_closed_form`-style
    // battery) times 2^-17 underflows toward F16's subnormal floor while the
    // largest entry (4.0) stays representable, so this is genuinely full
    // rank in F16, not rejected by the rank cutoff.
    let general: [f32; 9] = [4.0, 1.0, 0.5, 1.0, 3.0, 1.0, 0.25, 1.0, 2.0];
    let f16: Vec<F16> = general
        .iter()
        .map(|&v| F16::from_f64(scale(f64::from(v), -17)))
        .collect();
    let m = Array2::from_shape_vec([3, 3], f16).unwrap();
    match pinv(&m.view()) {
        Err(LetoError::Overflow { .. }) => {}
        other => panic!("expected Overflow, got {other:?}"),
    }

    let f64_general: Vec<f64> = general
        .iter()
        .map(|&v| scale(f64::from(v), -1030))
        .collect();
    assert!(
        f64_general.iter().all(|&v| v != 0.0),
        "input must stay subnormal, not zero"
    );
    let m = Array2::from_shape_vec([3, 3], f64_general).unwrap();
    match pinv(&m.view()) {
        Err(LetoError::Overflow { .. }) => {}
        other => panic!("expected Overflow, got {other:?}"),
    }
}
