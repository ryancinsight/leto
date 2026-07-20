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

/// Box-Muller standard-normal sampler that emits *both* normals from each
/// uniform pair.
///
/// The basic transform maps two uniforms `(u1, u2)` to two independent standard
/// normals: with `radius = √(-2·ln u1)` and `θ = τ·u2`, they are `radius·cos θ`
/// and `radius·sin θ`. Yielding only the cosine half — the historical behavior —
/// discarded the sine half yet still paid the `ln`/`sqrt`/trig cost for every
/// element. Caching the sine normal halves the transcendental work per sample.
/// The distribution is identical; the exact per-seed *sequence* is not (callers
/// depend on `N(mean, std_dev)` reproducibility, not on specific draw values).
///
/// One generator drives both the contiguous and strided output paths, so a seed
/// yields the same sequence regardless of the destination layout.
struct StandardNormals<T: RealScalar> {
    rng: Xorshift64,
    /// The sine-half normal of the most recent pair, awaiting emission.
    cached: Option<T>,
}

impl<T: RealScalar> StandardNormals<T> {
    #[inline]
    fn new(seed: u64) -> Self {
        Self {
            rng: Xorshift64::new(seed),
            cached: None,
        }
    }

    /// The next standard normal `N(0, 1)`.
    #[inline]
    fn next_standard(&mut self) -> T {
        if let Some(sine_half) = self.cached.take() {
            return sine_half;
        }
        // `1 - u` flips the PRNG's `[0, 1)` onto `(0, 1]` so `ln u1` stays finite;
        // `u2 ∈ [0, 1)` is the fraction of a full turn `τ`.
        let u1 = T::ONE.sub(T::from_f64(self.rng.next_unit_f64()));
        let u2 = T::from_f64(self.rng.next_unit_f64());
        let radius = T::from_f64(-2.0).mul(u1.ln()).sqrt();
        let angle = T::from_f64(std::f64::consts::TAU).mul(u2);
        self.cached = Some(radius.mul(angle.sin()));
        radius.mul(angle.cos())
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
    let mut normals = StandardNormals::<T>::new(seed);

    if let Some(out_slice) = out.as_mut_slice() {
        for val in out_slice.iter_mut() {
            *val = mean.add(std_dev.mul(normals.next_standard()));
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
            out_data[out_offset as usize] = mean.add(std_dev.mul(normals.next_standard()));
            out_offset += out_step;
        }
    }

    Ok(())
}

/// Fill a C-contiguous array of `shape` with i.i.d. normal samples of the
/// given `mean` and `std_dev`, derived deterministically from `seed`.
///
/// Uses the Box-Muller transform via the internal `StandardNormals` iterator,
/// which yields both normals of each `(u1, u2)` pair, so two output elements
/// share one `ln`/`sqrt`/`sin`/`cos` evaluation. The arithmetic runs in the
/// native precision of `T`.
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
