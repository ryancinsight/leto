use crate::domain::scalar::Scalar;
use leto::{ArrayView, ArrayViewMut, LetoError, Result};

// Helper to convert flat 1D index to N-dimensional index
#[inline(always)]
fn index_from_flat<const N: usize>(flat: usize, shape: &[usize; N]) -> [usize; N] {
    let mut index = [0usize; N];
    let mut temp = flat;
    for i in (0..N).rev() {
        if shape[i] > 0 {
            index[i] = temp % shape[i];
            temp /= shape[i];
        }
    }
    index
}

/// Element-wise array addition: `out = lhs + rhs`
pub fn add<T: Scalar, const N: usize>(
    lhs: &ArrayView<'_, T, N>,
    rhs: &ArrayView<'_, T, N>,
    out: &mut ArrayViewMut<'_, T, N>,
) -> Result<()> {
    if lhs.shape() != rhs.shape() || lhs.shape() != out.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }

    // Contiguous layout fast path
    if let (Some(lhs_slice), Some(rhs_slice), Some(out_slice)) =
        (lhs.as_slice(), rhs.as_slice(), out.as_mut_slice())
    {
        #[cfg(feature = "parallel")]
        {
            if lhs_slice.len() >= 8192 {
                let numel = lhs_slice.len();
                let lhs_ptr = lhs_slice.as_ptr() as usize;
                let rhs_ptr = rhs_slice.as_ptr() as usize;
                let out_ptr = out_slice.as_mut_ptr() as usize;

                crate::infrastructure::parallel::parallel_for(0, numel, move |i| unsafe {
                    let lhs_val = *(lhs_ptr as *const T).add(i);
                    let rhs_val = *(rhs_ptr as *const T).add(i);
                    *(out_ptr as *mut T).add(i) = lhs_val.add(rhs_val);
                });
                return Ok(());
            }
        }

        T::add_slice(lhs_slice, rhs_slice, out_slice);
        return Ok(());
    }

    // Strided layout fallback
    let size = lhs.size();
    let shape = lhs.shape();
    let lhs_layout = lhs.layout();
    let rhs_layout = rhs.layout();
    let out_layout = out.layout();

    let lhs_data = lhs.data();
    let rhs_data = rhs.data();
    let out_data = out.data_mut();

    #[cfg(feature = "parallel")]
    {
        if size >= 8192 {
            let lhs_ptr = lhs_data.as_ptr() as usize;
            let rhs_ptr = rhs_data.as_ptr() as usize;
            let out_ptr = out_data.as_mut_ptr() as usize;

            crate::infrastructure::parallel::parallel_for(0, size, move |flat_idx| unsafe {
                let multi_idx = index_from_flat(flat_idx, &shape);
                let lhs_off = lhs_layout.offset_of(multi_idx).unwrap();
                let rhs_off = rhs_layout.offset_of(multi_idx).unwrap();
                let out_off = out_layout.offset_of(multi_idx).unwrap();

                let lhs_val = *(lhs_ptr as *const T).add(lhs_off);
                let rhs_val = *(rhs_ptr as *const T).add(rhs_off);
                *(out_ptr as *mut T).add(out_off) = lhs_val.add(rhs_val);
            });
            return Ok(());
        }
    }

    for flat_idx in 0..size {
        let multi_idx = index_from_flat(flat_idx, &shape);
        let lhs_off = lhs_layout.offset_of(multi_idx)?;
        let rhs_off = rhs_layout.offset_of(multi_idx)?;
        let out_off = out_layout.offset_of(multi_idx)?;
        out_data[out_off] = lhs_data[lhs_off].add(rhs_data[rhs_off]);
    }

    Ok(())
}

/// Element-wise array subtraction: `out = lhs - rhs`
pub fn sub<T: Scalar, const N: usize>(
    lhs: &ArrayView<'_, T, N>,
    rhs: &ArrayView<'_, T, N>,
    out: &mut ArrayViewMut<'_, T, N>,
) -> Result<()> {
    if lhs.shape() != rhs.shape() || lhs.shape() != out.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }

    if let (Some(lhs_slice), Some(rhs_slice), Some(out_slice)) =
        (lhs.as_slice(), rhs.as_slice(), out.as_mut_slice())
    {
        #[cfg(feature = "parallel")]
        {
            if lhs_slice.len() >= 8192 {
                let numel = lhs_slice.len();
                let lhs_ptr = lhs_slice.as_ptr() as usize;
                let rhs_ptr = rhs_slice.as_ptr() as usize;
                let out_ptr = out_slice.as_mut_ptr() as usize;

                crate::infrastructure::parallel::parallel_for(0, numel, move |i| unsafe {
                    let lhs_val = *(lhs_ptr as *const T).add(i);
                    let rhs_val = *(rhs_ptr as *const T).add(i);
                    *(out_ptr as *mut T).add(i) = lhs_val.sub(rhs_val);
                });
                return Ok(());
            }
        }

        T::sub_slice(lhs_slice, rhs_slice, out_slice);
        return Ok(());
    }

    let size = lhs.size();
    let shape = lhs.shape();
    let lhs_layout = lhs.layout();
    let rhs_layout = rhs.layout();
    let out_layout = out.layout();

    let lhs_data = lhs.data();
    let rhs_data = rhs.data();
    let out_data = out.data_mut();

    #[cfg(feature = "parallel")]
    {
        if size >= 8192 {
            let lhs_ptr = lhs_data.as_ptr() as usize;
            let rhs_ptr = rhs_data.as_ptr() as usize;
            let out_ptr = out_data.as_mut_ptr() as usize;

            crate::infrastructure::parallel::parallel_for(0, size, move |flat_idx| unsafe {
                let multi_idx = index_from_flat(flat_idx, &shape);
                let lhs_off = lhs_layout.offset_of(multi_idx).unwrap();
                let rhs_off = rhs_layout.offset_of(multi_idx).unwrap();
                let out_off = out_layout.offset_of(multi_idx).unwrap();

                let lhs_val = *(lhs_ptr as *const T).add(lhs_off);
                let rhs_val = *(rhs_ptr as *const T).add(rhs_off);
                *(out_ptr as *mut T).add(out_off) = lhs_val.sub(rhs_val);
            });
            return Ok(());
        }
    }

    for flat_idx in 0..size {
        let multi_idx = index_from_flat(flat_idx, &shape);
        let lhs_off = lhs_layout.offset_of(multi_idx)?;
        let rhs_off = rhs_layout.offset_of(multi_idx)?;
        let out_off = out_layout.offset_of(multi_idx)?;
        out_data[out_off] = lhs_data[lhs_off].sub(rhs_data[rhs_off]);
    }

    Ok(())
}

/// Element-wise array multiplication: `out = lhs * rhs`
pub fn mul<T: Scalar, const N: usize>(
    lhs: &ArrayView<'_, T, N>,
    rhs: &ArrayView<'_, T, N>,
    out: &mut ArrayViewMut<'_, T, N>,
) -> Result<()> {
    if lhs.shape() != rhs.shape() || lhs.shape() != out.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }

    if let (Some(lhs_slice), Some(rhs_slice), Some(out_slice)) =
        (lhs.as_slice(), rhs.as_slice(), out.as_mut_slice())
    {
        #[cfg(feature = "parallel")]
        {
            if lhs_slice.len() >= 8192 {
                let numel = lhs_slice.len();
                let lhs_ptr = lhs_slice.as_ptr() as usize;
                let rhs_ptr = rhs_slice.as_ptr() as usize;
                let out_ptr = out_slice.as_mut_ptr() as usize;

                crate::infrastructure::parallel::parallel_for(0, numel, move |i| unsafe {
                    let lhs_val = *(lhs_ptr as *const T).add(i);
                    let rhs_val = *(rhs_ptr as *const T).add(i);
                    *(out_ptr as *mut T).add(i) = lhs_val.mul(rhs_val);
                });
                return Ok(());
            }
        }

        T::mul_slice(lhs_slice, rhs_slice, out_slice);
        return Ok(());
    }

    let size = lhs.size();
    let shape = lhs.shape();
    let lhs_layout = lhs.layout();
    let rhs_layout = rhs.layout();
    let out_layout = out.layout();

    let lhs_data = lhs.data();
    let rhs_data = rhs.data();
    let out_data = out.data_mut();

    #[cfg(feature = "parallel")]
    {
        if size >= 8192 {
            let lhs_ptr = lhs_data.as_ptr() as usize;
            let rhs_ptr = rhs_data.as_ptr() as usize;
            let out_ptr = out_data.as_mut_ptr() as usize;

            crate::infrastructure::parallel::parallel_for(0, size, move |flat_idx| unsafe {
                let multi_idx = index_from_flat(flat_idx, &shape);
                let lhs_off = lhs_layout.offset_of(multi_idx).unwrap();
                let rhs_off = rhs_layout.offset_of(multi_idx).unwrap();
                let out_off = out_layout.offset_of(multi_idx).unwrap();

                let lhs_val = *(lhs_ptr as *const T).add(lhs_off);
                let rhs_val = *(rhs_ptr as *const T).add(rhs_off);
                *(out_ptr as *mut T).add(out_off) = lhs_val.mul(rhs_val);
            });
            return Ok(());
        }
    }

    for flat_idx in 0..size {
        let multi_idx = index_from_flat(flat_idx, &shape);
        let lhs_off = lhs_layout.offset_of(multi_idx)?;
        let rhs_off = rhs_layout.offset_of(multi_idx)?;
        let out_off = out_layout.offset_of(multi_idx)?;
        out_data[out_off] = lhs_data[lhs_off].mul(rhs_data[rhs_off]);
    }

    Ok(())
}

/// Element-wise array division: `out = lhs / rhs`
pub fn div<T: Scalar, const N: usize>(
    lhs: &ArrayView<'_, T, N>,
    rhs: &ArrayView<'_, T, N>,
    out: &mut ArrayViewMut<'_, T, N>,
) -> Result<()> {
    if lhs.shape() != rhs.shape() || lhs.shape() != out.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }

    if let (Some(lhs_slice), Some(rhs_slice), Some(out_slice)) =
        (lhs.as_slice(), rhs.as_slice(), out.as_mut_slice())
    {
        #[cfg(feature = "parallel")]
        {
            if lhs_slice.len() >= 8192 {
                let numel = lhs_slice.len();
                let lhs_ptr = lhs_slice.as_ptr() as usize;
                let rhs_ptr = rhs_slice.as_ptr() as usize;
                let out_ptr = out_slice.as_mut_ptr() as usize;

                crate::infrastructure::parallel::parallel_for(0, numel, move |i| unsafe {
                    let lhs_val = *(lhs_ptr as *const T).add(i);
                    let rhs_val = *(rhs_ptr as *const T).add(i);
                    *(out_ptr as *mut T).add(i) = lhs_val.div(rhs_val);
                });
                return Ok(());
            }
        }

        T::div_slice(lhs_slice, rhs_slice, out_slice);
        return Ok(());
    }

    let size = lhs.size();
    let shape = lhs.shape();
    let lhs_layout = lhs.layout();
    let rhs_layout = rhs.layout();
    let out_layout = out.layout();

    let lhs_data = lhs.data();
    let rhs_data = rhs.data();
    let out_data = out.data_mut();

    #[cfg(feature = "parallel")]
    {
        if size >= 8192 {
            let lhs_ptr = lhs_data.as_ptr() as usize;
            let rhs_ptr = rhs_data.as_ptr() as usize;
            let out_ptr = out_data.as_mut_ptr() as usize;

            crate::infrastructure::parallel::parallel_for(0, size, move |flat_idx| unsafe {
                let multi_idx = index_from_flat(flat_idx, &shape);
                let lhs_off = lhs_layout.offset_of(multi_idx).unwrap();
                let rhs_off = rhs_layout.offset_of(multi_idx).unwrap();
                let out_off = out_layout.offset_of(multi_idx).unwrap();

                let lhs_val = *(lhs_ptr as *const T).add(lhs_off);
                let rhs_val = *(rhs_ptr as *const T).add(rhs_off);
                *(out_ptr as *mut T).add(out_off) = lhs_val.div(rhs_val);
            });
            return Ok(());
        }
    }

    for flat_idx in 0..size {
        let multi_idx = index_from_flat(flat_idx, &shape);
        let lhs_off = lhs_layout.offset_of(multi_idx)?;
        let rhs_off = rhs_layout.offset_of(multi_idx)?;
        let out_off = out_layout.offset_of(multi_idx)?;
        out_data[out_off] = lhs_data[lhs_off].div(rhs_data[rhs_off]);
    }

    Ok(())
}

// ── Reductions ──

/// Sum reduction over all elements of the view.
pub fn sum<T: Scalar, const N: usize>(arr: &ArrayView<'_, T, N>) -> T {
    if let Some(slice) = arr.as_slice() {
        return T::sum_slice(slice);
    }

    let size = arr.size();
    let shape = arr.shape();
    let layout = arr.layout();
    let data = arr.data();

    let mut total = T::ZERO;
    for flat_idx in 0..size {
        let multi_idx = index_from_flat(flat_idx, &shape);
        if let Ok(off) = layout.offset_of(multi_idx) {
            total = total.add(data[off]);
        }
    }
    total
}

// ── Matrix Multiplication ──

/// Perform matrix multiplication `out = lhs * rhs` for 2D views.
/// Uses cache-friendly loop ordering and optional row-wise Moirai parallelization.
pub fn matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
) -> Result<()> {
    let [m, k1] = lhs.shape();
    let [k2, n] = rhs.shape();
    let [out_m, out_n] = out.shape();

    if k1 != k2 || m != out_m || n != out_n {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }

    // Zero out initial output view
    for r in 0..m {
        for c in 0..n {
            *out.get_mut([r, c])? = T::ZERO;
        }
    }

    #[cfg(feature = "parallel")]
    {
        if m >= 16 {
            let lhs_ptr = lhs.data().as_ptr() as usize;
            let rhs_ptr = rhs.data().as_ptr() as usize;
            let out_ptr = out.data_mut().as_mut_ptr() as usize;

            let lhs_layout = lhs.layout();
            let rhs_layout = rhs.layout();
            let out_layout = out.layout();

            crate::infrastructure::parallel::parallel_for(0, m, move |i| unsafe {
                for k in 0..k1 {
                    let lhs_off = lhs_layout.offset_of([i, k]).unwrap();
                    let lhs_val = *(lhs_ptr as *const T).add(lhs_off);
                    if lhs_val == T::ZERO {
                        continue;
                    }
                    for j in 0..n {
                        let rhs_off = rhs_layout.offset_of([k, j]).unwrap();
                        let rhs_val = *(rhs_ptr as *const T).add(rhs_off);
                        let out_off = out_layout.offset_of([i, j]).unwrap();
                        let out_ref = &mut *(out_ptr as *mut T).add(out_off);
                        *out_ref = out_ref.add(lhs_val.mul(rhs_val));
                    }
                }
            });
            return Ok(());
        }
    }

    // Cache-efficient sequential loop ordering
    for i in 0..m {
        for k in 0..k1 {
            let lhs_val = *lhs.get([i, k])?;
            if lhs_val == T::ZERO {
                continue;
            }
            for j in 0..n {
                let rhs_val = *rhs.get([k, j])?;
                let out_ref = out.get_mut([i, j])?;
                *out_ref = out_ref.add(lhs_val.mul(rhs_val));
            }
        }
    }

    Ok(())
}
