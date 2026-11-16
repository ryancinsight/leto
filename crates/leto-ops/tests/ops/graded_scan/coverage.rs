//! Which factor sites the derived bounds cover in the scan, per format.
//!
//! A factor assertion runs only where its a-priori bound is below `1`
//! (`backward_error::informative`); elsewhere only the a-posteriori
//! certificates apply. [`Coverage`] counts, per format and site, the cases
//! asserted and the cases not, and requires every case not asserted to
//! derive, from its own rounding counts alone, that its bound is at least
//! `1`: a site is recorded vacuous by derivation, never by absence.
//!
//! # Derivation
//!
//! The probabilistic bounds are `γ̃_N` (`backward_error::gamma_tilde`) at the
//! entry's rounding count `N`, composed with nonnegative deterministic
//! factors. Write `E(N) = λ√N·u + N·u²/(1−u)`, so `γ̃_N = exp(E(N)) − 1`,
//! `u = ε/2`. With `d = √m·γ̃_{N_v}` for the accumulated vectors (`m` the
//! factor order) and `1 + η ≥ exp(E(N_r))` for the values:
//!
//! - orthogonality `2d + d² = (1 + d)² − 1`, so it is `≥ 1` exactly when
//!   `2·ln(1 + d) ≥ ln 2`;
//! - the residual `η + (2d + d²)(1 + η) = (1 + η)(1 + d)² − 1` is `≥ 1` when
//!   `E(N_r) + 2·ln(1 + d) ≥ ln 2`, dropping the nonnegative deterministic
//!   factors of `1 + η`.
//!
//! The test below pins `E` against `gamma_tilde` and the count at which `E`
//! crosses `ln 2`, so the log-domain derivation here is cross-checked
//! against the evaluation the oracle asserts with. The tabulated rows are a
//! regression pin of the scan's seeds; the per-case derivation is the
//! oracle.

use super::super::backward_error::PROBABILISTIC_LAMBDA;
use eunomia::FloatElement;
use std::collections::BTreeMap;

/// A factor site: the factorization and the quantity asserted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Site {
    SvdResidual,
    SvdOrthogonality,
    SchurResidual,
    SchurOrthogonality,
}

impl Site {
    fn is_residual(self) -> bool {
        matches!(self, Self::SvdResidual | Self::SchurResidual)
    }
}

/// The cases of one (format, site): asserted, and not asserted with their
/// bound derived to be at least `1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Row {
    pub(super) asserted: usize,
    pub(super) vacuous: usize,
}

impl Row {
    pub(super) const fn new(asserted: usize, vacuous: usize) -> Self {
        Self { asserted, vacuous }
    }
}

/// One case's per-entry rounding counts: `value` along the value path,
/// `vectors` along the accumulated factors, and the factor order `order`.
#[derive(Clone, Copy)]
pub(super) struct Counts {
    pub(super) value: f64,
    pub(super) vectors: f64,
    pub(super) order: usize,
}

/// `E(N) = ln(1 + γ̃_N)`: the log-domain growth of `N` roundings at
/// `u = ε/2`, `λ = λ₇`.
fn growth(roundings: f64, eps: f64) -> f64 {
    let u = eps / 2.0;
    PROBABILISTIC_LAMBDA * roundings.sqrt() * u + roundings * u * u / (1.0 - u)
}

/// Whether `counts` derive a bound of at least `1` at `site`
/// (module documentation).
fn derives_vacuous(site: Site, counts: Counts, eps: f64) -> bool {
    let vectors = f64::from_count(counts.order).sqrt() * growth(counts.vectors, eps).exp_m1();
    let values = if site.is_residual() {
        growth(counts.value, eps)
    } else {
        0.0
    };
    values + 2.0 * vectors.ln_1p() >= std::f64::consts::LN_2
}

/// Counts of asserted and vacuous cases, keyed by format precision and site.
#[derive(Default)]
pub(super) struct Coverage {
    rows: BTreeMap<(i32, Site), Row>,
}

impl Coverage {
    /// Records one case of the format with `precision` bits. A case whose
    /// factor assertion did not run must derive its bound `≥ 1` from `counts`.
    pub(super) fn record(
        &mut self,
        precision: i32,
        site: Site,
        asserted: bool,
        counts: Counts,
        eps: f64,
    ) {
        let row = self.rows.entry((precision, site)).or_default();
        if asserted {
            row.asserted += 1;
        } else {
            assert!(
                derives_vacuous(site, counts, eps),
                "p = {precision} {site:?}: bound not below 1, yet counts value {} vectors {} \
                 order {} derive less than 1",
                counts.value,
                counts.vectors,
                counts.order
            );
            row.vacuous += 1;
        }
    }

    /// Requires the recorded rows to equal `expected` exactly.
    pub(super) fn assert_rows(&self, expected: &[(i32, Site, Row)]) {
        let expected: BTreeMap<_, _> = expected
            .iter()
            .map(|&(precision, site, row)| ((precision, site), row))
            .collect();
        assert_eq!(self.rows, expected);
    }
}

#[cfg(test)]
mod growth_counts {
    use super::super::super::backward_error::gamma_tilde;
    use super::super::super::format::{epsilon, Format};
    use super::growth;
    use eunomia::{Bf16, F16};

    /// The round trip `ln_1p(exp_m1(E))` and the evaluations' operations
    /// (square root, product, sum, `exp_m1`, `ln_1p`) stay within this many
    /// relative `ε₆₄` of `E`: one rounding per operation, doubled for the
    /// libm functions' own error.
    const ROUND_TRIP_ULPS: f64 = 8.0;

    /// `E(N) = ln(1 + γ̃_N)` at counts spanning the scans' range.
    fn growth_matches_gamma_tilde<T: Format>() {
        let eps = epsilon::<T>();
        for roundings in [1.0, 24.0, 250.0, 624.0, 625.0, 4000.0, 40_000.0] {
            let expected = gamma_tilde(roundings, eps).ln_1p();
            let derived = growth(roundings, eps);
            assert!(
                (derived - expected).abs() <= ROUND_TRIP_ULPS * f64::EPSILON * expected,
                "p = {} N = {roundings}: E {derived:e} vs ln(1 + γ̃) {expected:e}",
                T::PRECISION
            );
        }
    }

    #[test]
    fn growth_is_the_log_of_the_probabilistic_bound_in_every_format() {
        growth_matches_gamma_tilde::<f64>();
        growth_matches_gamma_tilde::<f32>();
        growth_matches_gamma_tilde::<F16>();
        growth_matches_gamma_tilde::<Bf16>();
    }

    /// At `u = 2⁻⁸`, `λ = 7`: `γ̃_k < 1` holds for `k ≤ 624` and fails at
    /// `625`, the item's exact evaluation (`√k ≈ 25.0` at `E = ln 2`).
    #[test]
    fn growth_crosses_ln_2_at_625_roundings_for_unit_roundoff_2_pow_minus_8() {
        let eps = 2.0_f64.powi(-7);
        assert!(growth(624.0, eps) < std::f64::consts::LN_2);
        assert!(growth(625.0, eps) >= std::f64::consts::LN_2);
    }
}
