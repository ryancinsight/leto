//! Exact power-of-two scaling of dense factorization inputs, in two tiers.
//!
//! **Kernel tier.** The Francis double-shift and Golub–Kahan bidiagonal QR
//! kernels form their products scale-safely. A Givens norm, a reflector
//! norm, and the Golub–Kahan shift are formed unscaled while their local
//! magnitude lies in the kernel's representable window
//! ([`thresholds::kernel_window`](super::thresholds::kernel_window)) and, only
//! outside it, from operands rescaled by the power of two bringing the local
//! magnitude into `[1, 2)` ([`KernelWindow`]) — the LAPACK `dlartg` / `dnrm2`
//! pattern; inside the window the arithmetic is the unscaled arithmetic. The
//! Francis shift and first column (`dlahqr`) and the 2×2 standardization
//! (`dlanv2`) are scale-safe by construction: every product in them is of
//! ratios or square roots.
//!
//! **Matrix tier.** What the kernels cannot rescale locally — an intermediate
//! that is a product of entries accumulated across the whole matrix
//! (a pivoted-QR column norm, the QL chase's `e₁·eₗ`) or a degree-1 sum whose
//! bound grows with the dimension — is guarded by the gate: each routine
//! states the degree `d` and a power-of-two bound `2^f` with
//! `|intermediate| ≤ 2^f·‖A‖_max^d`, derived from its formulas at the call
//! site, and [`thresholds::homogeneous_safe_range`](super::thresholds::homogeneous_safe_range)
//! turns that into the range of `‖A‖_max` factored unscaled. An input inside
//! it is factored unscaled. Outside it the routine factors `2⁻ᵏ·A`, `k` the minimal move
//! (either sign) bringing `‖A‖_max` back inside ([`balancing_exponent`](gate::balancing_exponent)),
//! and multiplies the scale-carrying results back by `2ᵏ` ([`restore`]).
//!
//! **Exactness.** Multiplying by a power of two changes only the exponent, so
//! it is exact whenever the result stays normal. Scaling *up* is therefore
//! always exact; scaling *down* can underflow an entry far below the largest
//! one (`diag(1e300, 1e-300)` would lose `1e-300` if `f64` had to scale it
//! down). The minimal move scales down only as far as the gate's upper end
//! requires, and the upper ends are the overflow threshold itself (not the
//! LAPACK `ε/safmin` margin), so such loss occurs only for inputs within
//! `2^f` of overflowing; there the factorizations' backward-error bound, not
//! entrywise exactness, is the guarantee.

mod gate;
mod kernel_window;
mod norm_ratio;
#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test scope: a failed precondition is a test failure"
)]
mod tests;

pub(crate) use gate::{balanced, gate_exponent, restore, scale_by_power_of_two, GateBound};
pub(crate) use kernel_window::{hypot, KernelWindow};
pub(crate) use norm_ratio::{norm_ratio_floor_log2, norm_ratio_log2};
