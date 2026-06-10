use crate::domain::real::RealScalar;
use crate::domain::rng::Xorshift64;
use leto::{Array, Result, VecStorage};

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
    let layout = leto::Layout::c_contiguous(shape)?;
    let size = layout.size();
    let span = high.sub(low);
    let mut rng = Xorshift64::new(seed);
    let mut values = Vec::with_capacity(size);
    for _ in 0..size {
        let unit = T::from_f64(rng.next_unit_f64());
        values.push(low.add(unit.mul(span)));
    }
    Array::from_vec(shape, values)
}

/// Fill a C-contiguous array of `shape` with i.i.d. normal samples of the
/// given `mean` and `std_dev`, derived deterministically from `seed`.
///
/// Uses the Box-Muller transform. Each element consumes two uniforms; the
/// arithmetic (`ln`, `sqrt`, `cos`) runs in the native precision of `T`.
pub fn normal_with_seed<T: RealScalar, const N: usize>(
    shape: [usize; N],
    mean: T,
    std_dev: T,
    seed: u64,
) -> Result<Array<T, VecStorage<T>, N>> {
    let layout = leto::Layout::c_contiguous(shape)?;
    let size = layout.size();
    let tau = T::from_f64(std::f64::consts::TAU);
    let neg_two = T::from_f64(-2.0);
    let mut rng = Xorshift64::new(seed);
    let mut values = Vec::with_capacity(size);
    for _ in 0..size {
        // u1 in (0, 1] avoids ln(0); u2 in [0, 1) is the angle fraction.
        let u1 = T::ONE.sub(T::from_f64(rng.next_unit_f64()));
        let u2 = T::from_f64(rng.next_unit_f64());
        let radius = neg_two.mul(u1.ln()).sqrt();
        let z0 = radius.mul(tau.mul(u2).cos());
        values.push(mean.add(std_dev.mul(z0)));
    }
    Array::from_vec(shape, values)
}
