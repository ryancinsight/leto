use super::rotation::TransposedFactors;
use super::sweep::qr_iterate;
use super::RealScalar;
use eunomia::RealField;

/// Backward-error bound for one bidiagonal QR reduction of order `k`:
/// `‖E‖₂ ≤ p·ε·‖B‖₂` and `‖UᵀU − I‖₂ ≤ p·ε`, with `p` modest in the order
/// (Golub & Van Loan, *Matrix Computations* 4th ed., §8.6.3 — the sweep is a
/// product of plane rotations, each contributing `O(ε)`, and `O(k)` of them
/// touch any one entry). `p = 8k` is used throughout; Weyl's theorem carries
/// the same bound to every `|σ̂ᵢ − σᵢ|`, and to a `σ` whose exact value is 0.
///
/// Returns `(absolute, relative)`: the absolute bound scaled by `‖B‖₂ ≈ σ₁`
/// for singular values and reconstruction, the bare relative one for
/// orthonormality (`UᵀU − I` is already dimensionless).
fn error_bounds<T: RealScalar + RealField>(k: usize, norm: f64) -> (f64, f64) {
    let order = u32::try_from(k).expect("invariant: test orders fit in u32");
    let relative = 8.0 * f64::from(order) * <T as RealField>::EPSILON.to_f64();
    (relative * norm, relative)
}

/// Run the iteration on the bidiagonal `(diag, superdiag)` with `U`/`V`
/// initialized to the identity, then assert the singular values against
/// `expected` (analytic, descending, trailing zeros included), the
/// reconstruction `B = U Σ Vᵀ` of the **input** bidiagonal, and orthonormality
/// of both accumulated factors.
///
/// Reconstructing the input is what makes this a test of the rotations rather
/// than only of the deflation: a chase that zeroes the right entries but
/// accumulates into the wrong plane, the wrong row pair, or with the wrong
/// sign leaves the singular values correct and the reconstruction wrong.
fn check_bidiagonal<T: RealScalar + RealField>(diag: &[f64], superdiag: &[f64], expected: &[f64]) {
    let k = diag.len();
    let mut d: Vec<T> = diag.iter().map(|&x| T::from_f64(x)).collect();
    let mut e: Vec<T> = (0..k)
        .map(|i| T::from_f64(superdiag.get(i).copied().unwrap_or(0.0)))
        .collect();
    let mut ut = vec![T::ZERO; k * k];
    let mut vt = vec![T::ZERO; k * k];
    for i in 0..k {
        ut[i * k + i] = T::ONE;
        vt[i * k + i] = T::ONE;
    }

    qr_iterate::<T, true>(
        &mut d,
        &mut e,
        k,
        &mut TransposedFactors::new(&mut ut, k, &mut vt, k),
    )
    .expect("the bidiagonal iteration converges on a zero diagonal");

    let (absolute, relative) = error_bounds::<T>(k, expected[0]);

    let mut sigma: Vec<f64> = d.iter().map(|x| x.abs().to_f64()).collect();
    sigma.sort_by(|a, b| b.total_cmp(a));
    for (got, want) in sigma.iter().zip(expected) {
        assert!(
            (got - want).abs() <= absolute,
            "σ {got} vs {want} exceeds {absolute:e}"
        );
    }

    // Column `i` of `U`/`V` is row `i` of `ut`/`vt` (see `rotate_row_pair`).
    for row in 0..k {
        for col in 0..k {
            let value: f64 = (0..k)
                .map(|i| ut[i * k + row].to_f64() * d[i].to_f64() * vt[i * k + col].to_f64())
                .sum();
            let target = if col == row {
                diag[row]
            } else if col == row + 1 {
                superdiag[row]
            } else {
                0.0
            };
            assert!(
                (value - target).abs() <= absolute,
                "B[{row}][{col}] reconstructs as {value} not {target}"
            );
        }
    }

    for (name, factor) in [("U", &ut), ("V", &vt)] {
        for a in 0..k {
            for b in 0..k {
                let dot: f64 = (0..k)
                    .map(|r| factor[a * k + r].to_f64() * factor[b * k + r].to_f64())
                    .sum();
                let target = f64::from(u8::from(a == b));
                assert!(
                    (dot - target).abs() <= relative,
                    "{name}ᵀ{name}[{a}][{b}] = {dot} not {target}"
                );
            }
        }
    }
}

/// A **trailing** zero diagonal: `B = [[3, 4], [0, 0]]`, exactly rank 1.
///
/// This is the reduced form of the downstream reproducer — `[[1,2],[2,4],[3,6]]`
/// bidiagonalizes at `f32` to `d = [−3.7416573, 0]`, `e = [7.4833145]`, an
/// *exact* zero. Shifted QR alone cannot deflate it: the implicit `BᵀB` is
/// singular, the Wilkinson shift takes its nonzero eigenvalue, and the sweep
/// converges to the fixed point `d = (0, 0)` with `|e|` preserved, which the
/// `scale + |e| == scale` test never accepts. Measured before the trailing
/// column chase existed: `d[0]` decayed `2.4e-7 → 2.8e-14 → … → 6e-45` (the
/// smallest subnormal) while `|e[0]|` stayed pinned at `8.3666`, spinning to
/// the 4000-iteration cap and returning a non-convergence error.
///
/// `σ = (‖(3, 4)‖, 0) = (5, 0)`: row 0 is the only nonzero row, so `σ₁` is its
/// 2-norm and the second singular value is exactly zero.
#[test]
fn trailing_zero_diagonal_deflates_to_the_row_norm() {
    check_bidiagonal::<f32>(&[3.0, 0.0], &[4.0], &[5.0, 0.0]);
    check_bidiagonal::<f64>(&[3.0, 0.0], &[4.0], &[5.0, 0.0]);
}

/// An **interior** zero diagonal: `B = [[0,5,0],[0,3,6],[0,0,4]]` — `d[0] = 0`
/// with `p = 0 < q = 2`, which takes the left-rotation row chase rather than
/// the trailing-column one. Both branches are therefore covered.
///
/// Oracle: column 0 is zero, so `σ₃ = 0` and the remaining singular values are
/// those of `M = [[5,0],[3,6],[0,4]]`, i.e. `√λ` for the eigenvalues of the
/// 2×2 `MᵀM = [[34, 18], [18, 52]]` — a closed form independent of this code.
#[test]
fn interior_zero_diagonal_chases_out_of_the_block() {
    let (trace, det) = (86.0f64, 1444.0f64);
    let discriminant = trace.mul_add(trace, -4.0 * det).sqrt();
    let expected = [
        ((trace + discriminant) / 2.0).sqrt(),
        ((trace - discriminant) / 2.0).sqrt(),
        0.0,
    ];
    check_bidiagonal::<f32>(&[0.0, 3.0, 4.0], &[5.0, 6.0], &expected);
    check_bidiagonal::<f64>(&[0.0, 3.0, 4.0], &[5.0, 6.0], &expected);
}

/// The chase writes `T::ZERO` into the deflated diagonal and nothing touches
/// it afterwards, so a zero reached that way is bit-exact rather than a
/// rounding residue. Pinning it distinguishes the structural fix from the
/// prohibited alternative of widening `scale + |e| == scale`, which would
/// leave the deficient direction at `O(ε‖B‖)` instead.
#[test]
fn chased_zero_singular_value_is_bit_exact() {
    for (diag, superdiag) in [
        (vec![3.0f64, 0.0], vec![4.0]),           // trailing-column chase
        (vec![0.0f64, 3.0, 4.0], vec![5.0, 6.0]), // interior row chase
    ] {
        let k = diag.len();
        let mut d: Vec<f32> = diag.iter().map(|&x| x as f32).collect();
        let mut e: Vec<f32> = (0..k)
            .map(|i| superdiag.get(i).copied().unwrap_or(0.0) as f32)
            .collect();
        qr_iterate::<f32, false>(&mut d, &mut e, k, &mut TransposedFactors::none()).unwrap();
        assert!(
            d.contains(&0.0),
            "the chased direction must be exactly zero, got {d:?}"
        );
    }
}

/// `dbdsqr`'s forward test splits at `|eᵢ| ≤ tol·μ` with
/// `tol = tolmul·ε`, `tolmul = min(100, ε^(−1/8)) ≈ 90.5` in `f64`: in
/// `d = (1, 1, 1)`, `e = (50ε, ½)` the recurrence starts at
/// `μ = |d₀| = 1`, so `e₀` splits before any rotation, and row 0 of both
/// accumulated factors stays exactly `e₀ᵀ` (the `½` couples only the
/// trailing pair). With `tol = ε` the block stays whole and the sweep's
/// rotations mix row 0 by `O(e₀)`.
#[test]
fn forward_test_splits_at_tolmul_eps() {
    let mut d = vec![1.0f64, 1.0, 1.0];
    let mut e = vec![50.0 * f64::EPSILON, 0.5, 0.0];
    let identity = |n: usize| -> Vec<f64> {
        (0..n * n)
            .map(|i| if i % (n + 1) == 0 { 1.0 } else { 0.0 })
            .collect()
    };
    let (mut u, mut v) = (identity(3), identity(3));
    qr_iterate::<f64, true>(
        &mut d,
        &mut e,
        3,
        &mut TransposedFactors::new(&mut u, 3, &mut v, 3),
    )
    .unwrap();
    assert_eq!(&u[..3], &[1.0, 0.0, 0.0], "U row 0: {u:?}");
    assert_eq!(&v[..3], &[1.0, 0.0, 0.0], "V row 0: {v:?}");
}

/// The SVD gate keeps the joint floor `√k·safmin` within `ε·‖A‖_F`:
/// F16, `k = 64`, entries `0.3` and `0.225` (`‖A‖_F/‖A‖_max = 1.25`,
/// `r = 1`, `l = 0`); the floor end `2^(3 − l)·smlnum = 0.5` moves `0.3`
/// up, where crediting `r` would leave it at the root end `0.25`.
#[test]
fn gate_keeps_the_deflation_floor_below_epsilon_times_the_norm() {
    use super::svd_bound;
    use crate::application::linalg::scaling;
    use eunomia::{FloatElement, NumericElement, F16};
    let k = 64;
    let mut values = vec![F16::from_f64(0.0); k * k];
    values[0] = F16::from_f64(0.3);
    values[1] = F16::from_f64(0.225);
    let frobenius = |scale: i32| {
        values
            .iter()
            .map(|v| v.scale_binary(scale).to_f64().powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let (safmin, eps) = (2.0_f64.powi(-14), 2.0_f64.powi(-10));
    let joint_floor = (k as f64).sqrt() * safmin;
    assert!(joint_floor > eps * frobenius(0), "the input must violate");
    let exponent = scaling::gate_exponent(&values, 2, svd_bound(k, k))
        .expect("the range is non-empty")
        .expect("0.3 is below the floor end");
    assert!(joint_floor <= eps * frobenius(-exponent), "{exponent}");
    // `0.3·ones(64)`: `l = 5` credits `‖A‖_F = 64·‖A‖_max`, the floor end
    // falls to `2⁻⁶` below the root end `¼`, and the input stays in place;
    // without the credit the floor end `0.5` would move it.
    let ones = vec![F16::from_f64(0.3); k * k];
    let unmoved = scaling::gate_exponent(&ones, 2, svd_bound(k, k)).expect("non-empty");
    assert_eq!(unmoved, None);
}

/// `dbdsqr`'s second zero-shift test: on `d = (1, 1, 10⁻⁹)`,
/// `e = (½, 10⁻⁹)` the Wilkinson shift is `≈ 10⁻¹⁸`, below `ε·d₀²`, so
/// the step is exactly the zero-shift sweep (a shifted step with that
/// shift rounds differently).
#[test]
fn negligible_shift_takes_the_zero_shift_sweep() {
    use super::sweep::{qr_step, SweepWindows};
    use super::zero_shift::zero_shift_sweep;
    let windows = SweepWindows::<f64>::new();
    let (mut d, mut e) = (vec![1.0, 1.0, 1e-9], vec![0.5, 1e-9, 0.0]);
    let (mut d0, mut e0) = (d.clone(), e.clone());
    qr_step::<f64, false>(
        &mut d,
        &mut e,
        0,
        2,
        &mut TransposedFactors::none(),
        windows,
    );
    zero_shift_sweep::<f64, false>(
        &mut d0,
        &mut e0,
        0,
        2,
        &mut TransposedFactors::none(),
        windows.rotation,
    );
    assert_eq!((d, e), (d0, e0));
}

/// `dbdsqr`'s first zero-shift test scales with the order: `k = 3`,
/// `d = (1, 1, x)`, `e = 0` gives `σ̃_min/σ_max = x` and `tol ≈ 90.5ε` in
/// `f64`, so the test fires for `x ≤ 1/(3·90.5) ≈ 1/272`. At `x = 1/150`
/// it must not fire (without the order factor it would, `90.5/150 < 1`);
/// at `x = 1/400` it must.
#[test]
fn zero_shift_test_scales_with_the_order() {
    use super::deflation::Deflation;
    for (x, fires) in [(1.0 / 150.0, false), (1.0 / 400.0, true)] {
        let (d, e) = (vec![1.0f64, 1.0, x], vec![0.0, 0.0, 0.0]);
        let deflation = Deflation::new(&d, &e, 3);
        assert_eq!(
            deflation.shift_ruins_accuracy(&d, &e, 0, 2, 3),
            fires,
            "x = {x}"
        );
    }
}
