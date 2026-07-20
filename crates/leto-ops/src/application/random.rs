use crate::application::index::RowMajorTraversal;
use crate::domain::real::RealScalar;
use crate::domain::rng::Xorshift64;
use leto::{Array, ArrayViewMut, LetoError, Result, VecStorage};
use std::sync::LazyLock;

/// Fill a caller-owned view with i.i.d. uniform samples in `[low, high)`,
/// derived deterministically from `seed`.
pub fn uniform_with_seed_into<T: RealScalar, const N: usize>(
    out: &mut ArrayViewMut<'_, T, N>,
    low: T,
    high: T,
    seed: u64,
) -> Result<()> {
    let span = high.sub(low);
    let mut rng = Xorshift64::new(seed);

    if let Some(out_slice) = out.as_mut_slice() {
        for val in out_slice.iter_mut() {
            let unit = T::from_f64(rng.next_unit_f64());
            *val = low.add(unit.mul(span));
        }
        return Ok(());
    }

    out.layout().validate_storage_len(out.data().len())?;
    if out.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "random output layout must not contain zero-stride aliasing".to_string(),
        });
    }
    let size = out.layout().checked_size()?;
    let shape = out.shape();
    let out_layout = out.layout();
    let out_data = out.data_mut();

    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let out_step = traversal.last_axis_stride(out_layout);
    for row in 0..traversal.rows() {
        let base_idx = traversal.base_index(row);
        let mut out_offset = out_layout.offset_of(base_idx)? as isize;
        for _ in 0..traversal.inner() {
            let unit = T::from_f64(rng.next_unit_f64());
            out_data[out_offset as usize] = low.add(unit.mul(span));
            out_offset += out_step;
        }
    }

    Ok(())
}

/// Fill a C-contiguous array of `shape` with i.i.d. uniform samples in
/// `[low, high)`, derived deterministically from `seed`.
///
/// Sampling runs through the native precision of `T`: a unit `f64` from the
/// PRNG is converted once to `T` (a construction-time conversion, not a
/// compute-path widen-narrow) and affinely mapped into `[low, high)`.
pub fn uniform_with_seed<T: RealScalar, const N: usize>(
    shape: [usize; N],
    low: T,
    high: T,
    seed: u64,
) -> Result<Array<T, VecStorage<T>, N>> {
    let mut out = Array::from_elem(shape, T::ZERO);
    uniform_with_seed_into(&mut out.view_mut(), low, high, seed)?;
    Ok(out)
}

/// Precomputed Ziggurat tables for the standard normal distribution (128
/// layers), per Marsaglia & Tsang, "The Ziggurat Method for Generating Random
/// Variables" (J. Stat. Software 5(8), 2000), in the reference form of Burkardt's
/// `r8_nor`. [`ZIG_R`] (the tail boundary `x₁`) and [`ZIG_V`] (the common area of
/// each equal-area region) are the published self-consistent 128-layer constants;
/// the equal-area recurrence in [`ZigguratNormal::build`] reconstructs the ladder
/// from them, and the moment / tail / goodness-of-fit tests verify the result.
///
/// Replaces the Box-Muller sampler: ~99% of samples come from one table lookup
/// and one integer comparison, with no transcendental on the fast path, where
/// Box-Muller paid `ln`/`sqrt`/`sin`/`cos` on every sample. The
/// `N(mean, std_dev)` distribution is unchanged; the exact per-seed *sequence*
/// is not — Ziggurat consumes a data-dependent number of PRNG draws per sample
/// (callers depend on the documented distribution, not on specific draw values).
struct ZigguratNormal {
    /// Fast-accept thresholds: `|hz| < kn[iz]` accepts without the wedge test.
    kn: [u64; 128],
    /// Per-layer x-scale: a sample is `hz as f64 * wn[iz]`.
    wn: [f64; 128],
    /// Density `exp(-x_i²/2)` at each layer boundary, for the wedge test.
    fx: [f64; 128],
}

/// Tail boundary `x₁` for the 128-layer standard-normal Ziggurat
/// (Marsaglia & Tsang 2000).
const ZIG_R: f64 = 3.442_619_855_899;
/// Common area of each of the 128 equal-area regions (Marsaglia & Tsang 2000).
const ZIG_V: f64 = 9.912_563_035_262_17e-3;

/// The process-wide normal Ziggurat tables, built once on first use.
static ZIGGURAT_NORMAL: LazyLock<ZigguratNormal> = LazyLock::new(ZigguratNormal::build);

impl ZigguratNormal {
    /// Reconstruct the ladder from [`ZIG_R`]/[`ZIG_V`] via the equal-area
    /// recurrence `x_i = √(−2·ln(v/x_{i+1} + e^{−x_{i+1}²/2}))`, descending from
    /// the tail `x₁ = r` to the peak. Scales are relative to `2³¹` because a
    /// sample's index and magnitude come from a signed 32-bit integer.
    fn build() -> Self {
        const M1: f64 = 2_147_483_648.0;
        let mut kn = [0u64; 128];
        let mut wn = [0.0f64; 128];
        let mut fx = [0.0f64; 128];

        let mut dn = ZIG_R;
        let mut tn = ZIG_R;
        let q = ZIG_V / (-0.5 * dn * dn).exp();
        kn[0] = ((dn / q) * M1) as u64;
        kn[1] = 0;
        wn[0] = q / M1;
        wn[127] = dn / M1;
        fx[0] = 1.0;
        fx[127] = (-0.5 * dn * dn).exp();
        for i in (1..=126).rev() {
            dn = (-2.0 * (ZIG_V / dn + (-0.5 * dn * dn).exp()).ln()).sqrt();
            kn[i + 1] = ((dn / tn) * M1) as u64;
            tn = dn;
            fx[i] = (-0.5 * dn * dn).exp();
            wn[i] = dn / M1;
        }
        Self { kn, wn, fx }
    }

    /// One standard normal `N(0, 1)`. Fast path (~99%): a 32-bit draw, an index
    /// mask, and one `|hz| < kn[iz]` comparison. Slow path: the tail rejection
    /// for the base layer (`iz == 0`) or the wedge acceptance test.
    #[inline]
    fn sample(&self, rng: &mut Xorshift64) -> f64 {
        loop {
            // Low 32 bits as a signed value: its sign is the output sign, its low
            // 7 bits select the layer, its magnitude scales the sample.
            let hz = rng.next_u64() as u32 as i32;
            let iz = (hz & 127) as usize;
            if (hz as i64).unsigned_abs() < self.kn[iz] {
                return hz as f64 * self.wn[iz];
            }
            if iz == 0 {
                // Base layer: sample the tail beyond `r` by Marsaglia's method —
                // `x = −ln(u₁)/r`, accepted when `x² ≤ −2·ln(u₂)`.
                let tail = loop {
                    let x = -rng.next_unit_f64().ln() / ZIG_R;
                    let y = -rng.next_unit_f64().ln();
                    if x * x <= y + y {
                        break x;
                    }
                };
                return if hz <= 0 { -ZIG_R - tail } else { ZIG_R + tail };
            }
            let x = hz as f64 * self.wn[iz];
            // Wedge: accept when a uniform height between the layer's lower and
            // upper density bounds falls below the true density at `x`.
            if self.fx[iz] + rng.next_unit_f64() * (self.fx[iz - 1] - self.fx[iz])
                < (-0.5 * x * x).exp()
            {
                return x;
            }
            // Wedge rejected — draw a fresh `hz` on the next iteration.
        }
    }
}

/// Fill a caller-owned view with i.i.d. normal samples of the given `mean`
/// and `std_dev`, derived deterministically from `seed`.
pub fn normal_with_seed_into<T: RealScalar, const N: usize>(
    out: &mut ArrayViewMut<'_, T, N>,
    mean: T,
    std_dev: T,
    seed: u64,
) -> Result<()> {
    let mut rng = Xorshift64::new(seed);

    if let Some(out_slice) = out.as_mut_slice() {
        for val in out_slice.iter_mut() {
            *val = mean.add(std_dev.mul(T::from_f64(ZIGGURAT_NORMAL.sample(&mut rng))));
        }
        return Ok(());
    }

    out.layout().validate_storage_len(out.data().len())?;
    if out.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "random output layout must not contain zero-stride aliasing".to_string(),
        });
    }
    let size = out.layout().checked_size()?;
    let shape = out.shape();
    let out_layout = out.layout();
    let out_data = out.data_mut();

    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let out_step = traversal.last_axis_stride(out_layout);
    for row in 0..traversal.rows() {
        let base_idx = traversal.base_index(row);
        let mut out_offset = out_layout.offset_of(base_idx)? as isize;
        for _ in 0..traversal.inner() {
            out_data[out_offset as usize] = mean.add(std_dev.mul(T::from_f64(ZIGGURAT_NORMAL.sample(&mut rng))));
            out_offset += out_step;
        }
    }

    Ok(())
}

/// Fill a C-contiguous array of `shape` with i.i.d. normal samples of the
/// given `mean` and `std_dev`, derived deterministically from `seed`.
///
/// Standard normals come from the [`ZigguratNormal`] tables (Marsaglia & Tsang);
/// the standard-normal deviate is drawn in `f64` and converted once to `T`, then
/// scaled by `std_dev` and shifted by `mean`.
pub fn normal_with_seed<T: RealScalar, const N: usize>(
    shape: [usize; N],
    mean: T,
    std_dev: T,
    seed: u64,
) -> Result<Array<T, VecStorage<T>, N>> {
    let mut out = Array::from_elem(shape, T::ZERO);
    normal_with_seed_into(&mut out.view_mut(), mean, std_dev, seed)?;
    Ok(out)
}
