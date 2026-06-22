use crate::application::index::RowMajorTraversal;
use crate::domain::real::RealScalar;
use crate::domain::rng::Xorshift64;
use leto::{Array, ArrayViewMut, LetoError, Result, VecStorage};

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

/// Fill a caller-owned view with i.i.d. normal samples of the given `mean`
/// and `std_dev`, derived deterministically from `seed`.
pub fn normal_with_seed_into<T: RealScalar, const N: usize>(
    out: &mut ArrayViewMut<'_, T, N>,
    mean: T,
    std_dev: T,
    seed: u64,
) -> Result<()> {
    let tau = T::from_f64(std::f64::consts::TAU);
    let neg_two = T::from_f64(-2.0);
    let mut rng = Xorshift64::new(seed);

    if let Some(out_slice) = out.as_mut_slice() {
        for val in out_slice.iter_mut() {
            let u1 = T::ONE.sub(T::from_f64(rng.next_unit_f64()));
            let u2 = T::from_f64(rng.next_unit_f64());
            let radius = neg_two.mul(u1.ln()).sqrt();
            let z0 = radius.mul(tau.mul(u2).cos());
            *val = mean.add(std_dev.mul(z0));
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
            let u1 = T::ONE.sub(T::from_f64(rng.next_unit_f64()));
            let u2 = T::from_f64(rng.next_unit_f64());
            let radius = neg_two.mul(u1.ln()).sqrt();
            let z0 = radius.mul(tau.mul(u2).cos());
            out_data[out_offset as usize] = mean.add(std_dev.mul(z0));
            out_offset += out_step;
        }
    }

    Ok(())
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
    let mut out = Array::from_elem(shape, T::ZERO);
    normal_with_seed_into(&mut out.view_mut(), mean, std_dev, seed)?;
    Ok(out)
}
