//! Published coefficient rationals and their validity ranges.

use super::*;

// ── Coefficients ─────────────────────────────────────────────────────────────

#[test]
fn staggered_coefficients_match_the_published_rationals() {
    // Orders 2, 4, 6, 8 from Fornberg (1988) Table 1 / Levander (1988).
    let published: [&[f64]; 4] = [
        &[1.0],
        &[9.0 / 8.0, -1.0 / 24.0],
        &[75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0],
        &[
            1225.0 / 1024.0,
            -245.0 / 3072.0,
            49.0 / 5120.0,
            -5.0 / 7168.0,
        ],
    ];
    for (index, expected) in published.iter().enumerate() {
        let half_order = index + 1;
        let derived = staggered_first_derivative_coefficients::<f64>(half_order).unwrap();
        assert_eq!(derived.half_order(), half_order);
        assert_eq!(derived.order(), 2 * half_order);
        for (&derived, &expected) in derived.taps().iter().zip(expected.iter()) {
            let relative = (derived - expected).abs() / expected.abs();
            assert!(
                relative < 1e-13,
                "half-order {half_order}: derived {derived} vs published {expected} \
                 (relative {relative:e})"
            );
        }
    }
}

#[test]
fn collocated_coefficients_match_the_published_rationals() {
    let published: [&[f64]; 3] = [
        &[0.5],
        &[2.0 / 3.0, -1.0 / 12.0],
        &[3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0],
    ];
    for (index, expected) in published.iter().enumerate() {
        let derived = central_first_derivative_coefficients::<f64>(index + 1).unwrap();
        for (&derived, &expected) in derived.taps().iter().zip(expected.iter()) {
            let relative = (derived - expected).abs() / expected.abs();
            assert!(
                relative < 1e-13,
                "derived {derived} vs published {expected}"
            );
        }
    }
}

#[test]
fn coefficients_reject_orders_outside_the_verified_range() {
    assert!(staggered_first_derivative_coefficients::<f64>(0).is_err());
    assert!(staggered_first_derivative_coefficients::<f64>(MAX_HALF_ORDER + 1).is_err());
    assert!(central_first_derivative_coefficients::<f64>(0).is_err());
    assert!(central_first_derivative_coefficients::<f64>(MAX_HALF_ORDER + 1).is_err());
    // Every order inside the range derives.
    for half_order in 1..=MAX_HALF_ORDER {
        assert!(staggered_first_derivative_coefficients::<f64>(half_order).is_ok());
        assert!(central_first_derivative_coefficients::<f64>(half_order).is_ok());
    }
}
