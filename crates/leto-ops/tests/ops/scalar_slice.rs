//! `Scalar::axpy_pair_slice` value contract: a fused dual-output update is
//! value-identical to the two separate `axpy_slice` updates it replaces,
//! across SIMD widths, both machine precisions, and a plain integral type
//! that takes the scalar default.

use leto_ops::Scalar;

fn ramp<T: Copy + core::ops::AddAssign>(start: T, step: T, len: usize) -> Vec<T> {
    (0..len)
        .scan(start, |state, _| {
            let value = *state;
            *state += step;
            Some(value)
        })
        .collect()
}

fn pair_matches_two_axpy<T>(
    alpha_a: T,
    alpha_b: T,
    len: usize,
    x_a: (T, T),
    x_b: (T, T),
    out_a: (T, T),
    out_b: (T, T),
) where
    T: Scalar + Copy + PartialEq + core::fmt::Debug + core::ops::AddAssign,
{
    let x_a = ramp(x_a.0, x_a.1, len);
    let x_b = ramp(x_b.0, x_b.1, len);
    let mut expected_a = ramp(out_a.0, out_a.1, len);
    let mut expected_b = ramp(out_b.0, out_b.1, len);
    let mut out_a = expected_a.clone();
    let mut out_b = expected_b.clone();
    T::axpy_slice(alpha_a, &x_a, &mut expected_a);
    T::axpy_slice(alpha_b, &x_b, &mut expected_b);
    T::axpy_pair_slice(alpha_a, &x_a, &mut out_a, alpha_b, &x_b, &mut out_b);
    assert_eq!(out_a, expected_a, "out_a len {len}");
    assert_eq!(out_b, expected_b, "out_b len {len}");
}

#[test]
fn axpy_pair_slice_matches_two_axpy_slice_f64() {
    for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1027] {
        pair_matches_two_axpy::<f64>(
            1.75,
            -0.375,
            len,
            (-3.0, 0.5),
            (2.0, -0.25),
            (100.0, -1.0),
            (0.5, 1.5),
        );
    }
}

#[test]
fn axpy_pair_slice_matches_two_axpy_slice_f32() {
    for len in [0usize, 1, 9, 31, 65, 1031] {
        pair_matches_two_axpy::<f32>(
            1.0 / 3.0,
            -2.0 / 7.0,
            len,
            (0.125, 1.0),
            (0.75, -0.0625),
            (0.2, 0.5),
            (3.0, -0.3),
        );
    }
}

#[test]
fn axpy_pair_slice_matches_two_axpy_slice_i64() {
    for len in [0usize, 1, 9, 65] {
        pair_matches_two_axpy::<i64>(3, -2, len, (-5, 1), (7, -1), (100, -1), (1, 2));
    }
}

#[test]
#[should_panic(expected = "axpy_pair_slice: x_a.len() != out_a.len()")]
fn axpy_pair_slice_rejects_first_length_mismatch() {
    let mut out_a = [0.0f64; 2];
    let mut out_b = [0.0f64; 2];
    f64::axpy_pair_slice(1.0, &[1.0], &mut out_a, 1.0, &[1.0, 2.0], &mut out_b);
}

#[test]
#[should_panic(expected = "axpy_pair_slice: output length mismatch")]
fn axpy_pair_slice_rejects_output_length_mismatch() {
    let mut out_a = [0.0f64; 2];
    let mut out_b = [0.0f64; 3];
    f64::axpy_pair_slice(
        1.0,
        &[1.0, 2.0],
        &mut out_a,
        1.0,
        &[1.0, 2.0, 3.0],
        &mut out_b,
    );
}
