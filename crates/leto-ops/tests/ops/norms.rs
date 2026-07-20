use leto::{Array, Layout, SliceArg, Storage, VecStorage};
use leto_ops::{norm_l1, norm_l2, norm_max};

const EPS: f64 = 1e-12;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= EPS * expected.abs().max(1.0),
        "actual {actual} expected {expected}"
    );
}

#[test]
fn vector_norms_match_closed_form_definitions() {
    let array = Array::from_shape_vec([5], vec![3.0f64, -4.0, 12.0, -0.5, 2.25]).unwrap();

    assert_close(norm_l2(&array.view()).unwrap(), (174.3125f64).sqrt());
    assert_close(norm_l1(&array.view()).unwrap(), 21.75);
    assert_close(norm_max(&array.view()).unwrap(), 12.0);
}

#[test]
fn frobenius_norm_matches_closed_form_rank2() {
    let array = Array::from_shape_vec([2, 3], vec![1.0f64, -2.0, 3.5, 4.25, -5.5, 6.75]).unwrap();

    // norm_l2 over rank-2 is the Frobenius norm; one generic entry point.
    assert_close(norm_l2(&array.view()).unwrap(), (111.125f64).sqrt());
}

#[test]
fn norms_are_layout_independent_on_strided_views() {
    // A transposed view must produce the same norms as its source: the
    // elementwise norms are traversal-order independent and the strided
    // fallback visits each logical element exactly once.
    let values = vec![1.0f64, -2.0, 3.0, -4.0, 5.0, -6.0];
    let array = Array::from_shape_vec([2, 3], values).unwrap();
    let transposed = array.transpose([1, 0]).unwrap();

    assert_close(
        norm_l2(&transposed).unwrap(),
        norm_l2(&array.view()).unwrap(),
    );
    assert_close(
        norm_l1(&transposed).unwrap(),
        norm_l1(&array.view()).unwrap(),
    );
    assert_close(
        norm_max(&transposed).unwrap(),
        norm_max(&array.view()).unwrap(),
    );

    // Every-other-column strided slice: norms over the logical selection only.
    let strided = array
        .view()
        .slice_with::<2>(&[leto::SliceArg::All, leto::SliceArg::range(Some(0), None, 2)])
        .unwrap();
    // columns 0 and 2: values 1, 3, -4, -6 -> L1 = 14, max = 6
    assert_close(norm_l1(&strided).unwrap(), 14.0);
    assert_close(norm_max(&strided).unwrap(), 6.0);
    assert_close(
        norm_l2(&strided).unwrap(),
        (1.0f64 + 9.0 + 16.0 + 36.0).sqrt(),
    );

    let reversed = array
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();
    assert_close(norm_l2(&reversed).unwrap(), norm_l2(&array.view()).unwrap());
    assert_close(norm_l1(&reversed).unwrap(), norm_l1(&array.view()).unwrap());
    assert_close(
        norm_max(&reversed).unwrap(),
        norm_max(&array.view()).unwrap(),
    );
}

#[test]
fn empty_view_norms_are_zero() {
    let array: Array<f64, VecStorage<f64>, 1> =
        Array::new(Layout::c_contiguous([0]).unwrap(), VecStorage::new(vec![])).unwrap();
    assert_eq!(norm_l1(&array.view()).unwrap(), 0.0);
    assert_eq!(norm_l2(&array.view()).unwrap(), 0.0);
    assert_eq!(norm_max(&array.view()).unwrap(), 0.0);
}

#[test]
fn norms_run_at_reduced_precision() {
    use eunomia::F16;
    let values: Vec<F16> = [3.0f32, 4.0]
        .iter()
        .map(|&value| F16::from_f32(value))
        .collect();
    let array = Array::from_shape_vec([2], values).unwrap();
    // The 3-4-5 triangle is exactly representable in binary16.
    assert_eq!(norm_l2(&array.view()).unwrap(), F16::from_f32(5.0));
    assert_eq!(norm_l1(&array.view()).unwrap(), F16::from_f32(7.0));
    assert_eq!(norm_max(&array.view()).unwrap(), F16::from_f32(4.0));
}

#[test]
fn test_l2_normalize_generic() {
    // 1D test
    let array = Array::from_shape_vec([3], vec![3.0f64, 0.0, 4.0]).unwrap();
    let mut out = Array::from_elem([3], 0.0f64);
    leto_ops::l2_normalize_into(&array.view(), &mut out.view_mut(), 0.0).unwrap();
    assert_close(*out.get([0]).unwrap(), 0.6);
    assert_close(*out.get([1]).unwrap(), 0.0);
    assert_close(*out.get([2]).unwrap(), 0.8);

    let owned = leto_ops::l2_normalize(&array.view(), 0.0).unwrap();
    assert_close(*owned.get([0]).unwrap(), 0.6);
    assert_close(*owned.get([1]).unwrap(), 0.0);
    assert_close(*owned.get([2]).unwrap(), 0.8);

    // Epsilon stability test
    let mut out_eps = Array::from_elem([3], 0.0f64);
    leto_ops::l2_normalize_into(&array.view(), &mut out_eps.view_mut(), 5.0).unwrap();
    // l2 norm is 5.0. denom is 5.0 + 5.0 = 10.0.
    // values should be 3/10 = 0.3, 0.0, 4/10 = 0.4.
    assert_close(*out_eps.get([0]).unwrap(), 0.3);
    assert_close(*out_eps.get([1]).unwrap(), 0.0);
    assert_close(*out_eps.get([2]).unwrap(), 0.4);
}

#[test]
fn test_random_into() {
    let mut out1 = Array::from_elem([10], 0.0f64);
    leto_ops::uniform_with_seed_into(&mut out1.view_mut(), -1.0, 1.0, 42).unwrap();
    for &val in out1.storage().as_slice() {
        assert!((-1.0..1.0).contains(&val));
    }

    let mut out2 = Array::from_elem([10], 0.0f64);
    leto_ops::normal_with_seed_into(&mut out2.view_mut(), 5.0, 2.0, 42).unwrap();
    // deterministic seed should produce same output as normal_with_seed
    let normal_owned = leto_ops::normal_with_seed([10], 5.0, 2.0, 42).unwrap();
    assert_eq!(out2.storage().as_slice(), normal_owned.storage().as_slice());
}

/// A seed yields the same normal sequence regardless of the output view's
/// layout. One PRNG drives the shared Ziggurat sampler across both the
/// contiguous (`as_mut_slice`) and strided (`RowMajorTraversal`) fill paths,
/// consuming draws in row-major order either way — so element `(i, j)` gets the
/// same draw whether the destination is C-dense or transposed-strided. Were one
/// path to diverge (e.g. a future optimization applied to only the contiguous
/// branch), the two fills would desynchronize and this test would fail.
#[test]
fn normal_seed_sequence_is_layout_independent() {
    let seed = 7u64;

    // Contiguous [3, 4], filled through the `as_mut_slice` fast path.
    let mut contiguous = Array::from_elem([3, 4], 0.0f64);
    leto_ops::normal_with_seed_into(&mut contiguous.view_mut(), 0.0, 1.0, seed).unwrap();

    // The same logical [3, 4] as the transpose of a contiguous [4, 3]: the view
    // is not C-dense, so `normal_with_seed_into` takes the strided path.
    let mut backing = Array::from_elem([4, 3], 0.0f64);
    {
        let mut strided = backing.view_mut().transpose_mut([1, 0]).unwrap();
        assert!(
            !strided.is_c_dense(),
            "transposed view must be strided to exercise the non-contiguous path"
        );
        leto_ops::normal_with_seed_into(&mut strided, 0.0, 1.0, seed).unwrap();
    }
    let strided_logical = backing.view().transpose([1, 0]).unwrap();

    for i in 0..3 {
        for j in 0..4 {
            assert_eq!(
                contiguous.get([i, j]).unwrap(),
                strided_logical.get([i, j]).unwrap(),
                "layout-independent draw at ({i}, {j})"
            );
        }
    }
}

/// Standard-normal CDF via the Abramowitz & Stegun 26.2.17 rational
/// approximation (absolute error < 7.5e-8) — self-contained so the
/// goodness-of-fit check needs no external `erf`.
fn normal_cdf(x: f64) -> f64 {
    const B: [f64; 5] = [
        0.319_381_530,
        -0.356_563_782,
        1.781_477_937,
        -1.821_255_978,
        1.330_274_429,
    ];
    const P: f64 = 0.231_641_9;
    let t = 1.0 / (1.0 + P * x.abs());
    let phi = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let poly = t * (B[0] + t * (B[1] + t * (B[2] + t * (B[3] + t * B[4]))));
    let upper_tail = phi * poly; // ≈ 1 − Φ(|x|)
    if x >= 0.0 {
        1.0 - upper_tail
    } else {
        upper_tail
    }
}

/// The Ziggurat sampler must reproduce `N(0, 1)`, not merely pass a mean/std
/// check a broken tail or table could sneak past. Over 10M standard normals this
/// verifies the first four moments, the tail exceedance probabilities
/// `P(|Z| > k)`, and a 200-bin chi-squared goodness-of-fit against the analytic
/// normal. Each moment/tail tolerance is that estimator's standard error at this
/// `N` widened to 6σ, so a genuine distribution error fails while sampling noise
/// does not; the chi-squared bound (df ≈ 150, correct sampler ~df) separates a
/// correct sampler from a wrong one (which produces thousands).
#[test]
fn ziggurat_normal_matches_analytical_distribution() {
    let n = 10_000_000usize;
    let samples = leto_ops::normal_with_seed([n], 0.0f64, 1.0, 0x5eed_1234_abcd).unwrap();
    let data = samples.storage().as_slice();

    let (mut s1, mut s2, mut s3, mut s4) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut tail = [0u64; 4];
    const BINS: usize = 200;
    const LO: f64 = -6.0;
    const HI: f64 = 6.0;
    let width = (HI - LO) / BINS as f64;
    let mut hist = [0u64; BINS];
    for &z in data {
        s1 += z;
        s2 += z * z;
        s3 += z * z * z;
        s4 += z * z * z * z;
        let az = z.abs();
        for (k, t) in [1.0, 2.0, 3.0, 4.0].into_iter().enumerate() {
            if az > t {
                tail[k] += 1;
            }
        }
        if (LO..HI).contains(&z) {
            hist[((z - LO) / width) as usize] += 1;
        }
    }

    let nf = n as f64;
    let mean = s1 / nf;
    let (e2, e3, e4) = (s2 / nf, s3 / nf, s4 / nf);
    let m2 = e2 - mean * mean;
    let m3 = e3 - 3.0 * mean * e2 + 2.0 * mean.powi(3);
    let m4 = e4 - 4.0 * mean * e3 + 6.0 * mean * mean * e2 - 3.0 * mean.powi(4);
    let skew = m3 / m2.powf(1.5);
    let kurt = m4 / (m2 * m2);

    // Moment-estimator standard errors at N: 1/√N, √(2/N), √(6/N), √(24/N).
    assert!(mean.abs() < 6.0 * (1.0 / nf).sqrt(), "mean {mean}");
    assert!((m2 - 1.0).abs() < 6.0 * (2.0 / nf).sqrt(), "variance {m2}");
    assert!(skew.abs() < 6.0 * (6.0 / nf).sqrt(), "skewness {skew}");
    assert!(
        (kurt - 3.0).abs() < 6.0 * (24.0 / nf).sqrt(),
        "kurtosis {kurt}"
    );

    // Tail exceedance probabilities P(|Z| > k), k = 1..4.
    let expected_tail = [
        0.317_310_507_862_914_2,
        0.045_500_263_896_358_42,
        0.002_699_796_063_260_207,
        0.000_063_342_483_666_24,
    ];
    for k in 0..4 {
        let p_hat = tail[k] as f64 / nf;
        let se = (expected_tail[k] * (1.0 - expected_tail[k]) / nf).sqrt();
        assert!(
            (p_hat - expected_tail[k]).abs() < 6.0 * se,
            "P(|Z|>{}) = {p_hat}, expected {} (6σ = {})",
            k + 1,
            expected_tail[k],
            6.0 * se
        );
    }

    // Binned chi-squared goodness-of-fit over bins with expected count ≥ 5.
    let mut chi2 = 0.0f64;
    let mut df = 0usize;
    for (b, &observed) in hist.iter().enumerate() {
        let lo = LO + b as f64 * width;
        let expected = nf * (normal_cdf(lo + width) - normal_cdf(lo));
        if expected >= 5.0 {
            let diff = observed as f64 - expected;
            chi2 += diff * diff / expected;
            df += 1;
        }
    }
    assert!(
        chi2 < 400.0,
        "chi-squared {chi2} over {df} bins exceeds 400 (df ≈ 150; a correct sampler sits near df)"
    );
}

#[test]
fn test_solvers_into() {
    // LU solve_into
    let a_mat = Array::from_shape_vec([2, 2], vec![4.0f64, 3.0, 6.0, 3.0]).unwrap();
    let b_vec = Array::from_shape_vec([2], vec![10.0f64, 9.0]).unwrap();
    let lu = leto_ops::lu_decompose(&a_mat.view()).unwrap();
    let mut x_lu = Array::from_elem([2], 0.0f64);
    lu.solve_into(&b_vec.view(), &mut x_lu.view_mut()).unwrap();
    // Verification: A * x should equal b.
    // 4*(-0.5) + 3*4 = -2 + 12 = 10.
    // 6*(-0.5) + 3*4 = -3 + 12 = 9.
    assert_close(*x_lu.get([0]).unwrap(), -0.5);
    assert_close(*x_lu.get([1]).unwrap(), 4.0);

    // Cholesky solve_into (requires SPD matrix)
    let spd = Array::from_shape_vec([2, 2], vec![2.0f64, -1.0, -1.0, 2.0]).unwrap();
    let rhs_cholesky = Array::from_shape_vec([2], vec![1.0f64, 1.0]).unwrap();
    let chol = leto_ops::cholesky_decompose(&spd.view()).unwrap();
    let mut x_chol = Array::from_elem([2], 0.0f64);
    chol.solve_into(&rhs_cholesky.view(), &mut x_chol.view_mut())
        .unwrap();
    // A * x = [2*1 - 1, -1*1 + 2*1] = [1, 1]. x = [1, 1].
    assert_close(*x_chol.get([0]).unwrap(), 1.0);
    assert_close(*x_chol.get([1]).unwrap(), 1.0);

    // QR solve_least_squares_into
    let qr = leto_ops::qr_decompose(&a_mat.view()).unwrap();
    let mut x_qr = Array::from_elem([2], 0.0f64);
    qr.solve_least_squares_into(&b_vec.view(), &mut x_qr.view_mut())
        .unwrap();
    assert_close(*x_qr.get([0]).unwrap(), -0.5);
    assert_close(*x_qr.get([1]).unwrap(), 4.0);
}
