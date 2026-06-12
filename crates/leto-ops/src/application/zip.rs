use crate::application::index::{line_elements, RowMajorTraversal, TileGeometry};
use leto::{ArrayView, ArrayViewMut, LetoError, Result};

#[inline]
fn validate_zip_storage<T, U, const N: usize>(
    lhs: &ArrayViewMut<'_, T, N>,
    rhs: &ArrayView<'_, U, N>,
) -> Result<()> {
    lhs.layout().validate_storage_len(lhs.data().len())?;
    rhs.layout().validate_storage_len(rhs.data().len())?;
    if lhs.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "zip mutable output layout must not contain zero-stride aliasing".to_string(),
        });
    }
    Ok(())
}

/// Mutably zip-map elements of a view with elements from another view in place.
///
/// `lhs` owns mutation, `rhs` is read-only, and both views must have identical
/// logical shapes. Every logical element pair is visited exactly once, so the
/// resulting array values are independent of the backing storage layout; the
/// traversal *order* is unspecified (column-walk layouts run cache-line
/// tiled), so a stateful closure must not rely on row-major visitation —
/// use [`indexed_zip_mut_with`] when the logical index matters.
pub fn zip_mut_with<T, U, F, const N: usize>(
    lhs: &mut ArrayViewMut<'_, T, N>,
    rhs: &ArrayView<'_, U, N>,
    mut f: F,
) -> Result<()>
where
    F: FnMut(&mut T, &U),
{
    if lhs.shape() != rhs.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }

    validate_zip_storage(lhs, rhs)?;

    if let (Some(lhs_slice), Some(rhs_slice)) = (lhs.as_mut_slice(), rhs.as_slice()) {
        for (left, right) in lhs_slice.iter_mut().zip(rhs_slice.iter()) {
            f(left, right);
        }
        return Ok(());
    }

    let size = lhs.layout().checked_size()?;
    let shape = lhs.shape();
    let lhs_layout = lhs.layout();
    let rhs_layout = rhs.layout();
    let lhs_data = lhs.data_mut();
    let rhs_data = rhs.data();

    // Row-walk traversal: one offset computation per innermost row, then a
    // stride-increment walk (shared RowMajorTraversal policy; see binary_map
    // for rationale and the recorded baselines).
    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let lhs_step = traversal.last_axis_stride(lhs_layout);
    let rhs_step = traversal.last_axis_stride(rhs_layout);

    // Cache-line micro-tiling, mirroring binary_map: pays exactly when some
    // operand's last-axis walk skips whole lines. Mixed element sizes choose
    // the smaller elements-per-line count so both operands stay line-resident
    // inside a tile.
    let tile = line_elements::<T>().min(line_elements::<U>());
    let column_walk = lhs_step.unsigned_abs() >= tile || rhs_step.unsigned_abs() >= tile;
    if column_walk && N >= 2 {
        if let Some(geometry) = TileGeometry::new(size, shape, tile) {
            let (lhs_rs, rhs_rs) = (lhs_layout.strides[N - 2], rhs_layout.strides[N - 2]);
            for slab in 0..geometry.slabs() {
                let base_idx = geometry.slab_base_index(slab);
                let lhs_base = lhs_layout.offset_of(base_idx)? as isize;
                let rhs_base = rhs_layout.offset_of(base_idx)? as isize;
                let mut rb = 0;
                while rb < geometry.height() {
                    let rend = (rb + geometry.tile()).min(geometry.height());
                    let mut cb = 0;
                    while cb < geometry.width() {
                        let cend = (cb + geometry.tile()).min(geometry.width());
                        for r in rb..rend {
                            let r = r as isize;
                            let c0 = cb as isize;
                            let mut lhs_off = lhs_base + r * lhs_rs + c0 * lhs_step;
                            let mut rhs_off = rhs_base + r * rhs_rs + c0 * rhs_step;
                            for _ in cb..cend {
                                f(
                                    &mut lhs_data[lhs_off as usize],
                                    &rhs_data[rhs_off as usize],
                                );
                                lhs_off += lhs_step;
                                rhs_off += rhs_step;
                            }
                        }
                        cb = cend;
                    }
                    rb = rend;
                }
            }
            return Ok(());
        }
    }

    for row in 0..traversal.rows() {
        let base = traversal.base_index(row);
        let mut lhs_offset = lhs_layout.offset_of(base)? as isize;
        let mut rhs_offset = rhs_layout.offset_of(base)? as isize;
        for _ in 0..traversal.inner() {
            f(
                &mut lhs_data[lhs_offset as usize],
                &rhs_data[rhs_offset as usize],
            );
            lhs_offset += lhs_step;
            rhs_offset += rhs_step;
        }
    }

    Ok(())
}

/// Mutably zip-map elements in place with the logical row-major index.
///
/// This is the indexed analogue of [`zip_mut_with`] (`ndarray`'s
/// `Zip::indexed`). The closure receives the logical index before the mutable
/// and read-only operands, so Apollo/Coeus call sites can derive position-aware
/// scaling, phase, or layout metadata without allocating an index array.
pub fn indexed_zip_mut_with<T, U, F, const N: usize>(
    lhs: &mut ArrayViewMut<'_, T, N>,
    rhs: &ArrayView<'_, U, N>,
    mut f: F,
) -> Result<()>
where
    F: FnMut([usize; N], &mut T, &U),
{
    if lhs.shape() != rhs.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }

    validate_zip_storage(lhs, rhs)?;

    let size = lhs.layout().checked_size()?;
    let shape = lhs.shape();
    let lhs_layout = lhs.layout();
    let rhs_layout = rhs.layout();
    let lhs_data = lhs.data_mut();
    let rhs_data = rhs.data();

    // Row-walk with an incrementally updated last coordinate: the closure
    // still receives the exact logical index, but the per-element div/mod
    // decomposition and offset products are gone.
    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let lhs_step = traversal.last_axis_stride(lhs_layout);
    let rhs_step = traversal.last_axis_stride(rhs_layout);
    for row in 0..traversal.rows() {
        let mut index = traversal.base_index(row);
        let mut lhs_offset = lhs_layout.offset_of(index)? as isize;
        let mut rhs_offset = rhs_layout.offset_of(index)? as isize;
        for k in 0..traversal.inner() {
            if N > 0 {
                index[N - 1] = k;
            }
            f(
                index,
                &mut lhs_data[lhs_offset as usize],
                &rhs_data[rhs_offset as usize],
            );
            lhs_offset += lhs_step;
            rhs_offset += rhs_step;
        }
    }

    Ok(())
}

/// Mutably zip-map a view with elements from two read-only views in place.
///
/// The three-operand analogue of [`zip_mut_with`] (`ndarray`'s
/// `Zip::from(out).and(a).and(b)`). `lhs` owns mutation; `a` and `b` are
/// read-only. All three views must share the same logical shape. Strided inputs
/// are traversed by logical row-major index, independent of backing layout.
pub fn zip2_mut_with<T, A, B, F, const N: usize>(
    lhs: &mut ArrayViewMut<'_, T, N>,
    a: &ArrayView<'_, A, N>,
    b: &ArrayView<'_, B, N>,
    mut f: F,
) -> Result<()>
where
    F: FnMut(&mut T, &A, &B),
{
    if lhs.shape() != a.shape() || lhs.shape() != b.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: a.shape().to_vec(),
        });
    }

    lhs.layout().validate_storage_len(lhs.data().len())?;
    a.layout().validate_storage_len(a.data().len())?;
    b.layout().validate_storage_len(b.data().len())?;
    if lhs.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "zip mutable output layout must not contain zero-stride aliasing".to_string(),
        });
    }

    if let (Some(lhs_slice), Some(a_slice), Some(b_slice)) =
        (lhs.as_mut_slice(), a.as_slice(), b.as_slice())
    {
        for ((left, av), bv) in lhs_slice.iter_mut().zip(a_slice.iter()).zip(b_slice.iter()) {
            f(left, av, bv);
        }
        return Ok(());
    }

    let size = lhs.layout().checked_size()?;
    let shape = lhs.shape();
    let lhs_layout = lhs.layout();
    let a_layout = a.layout();
    let b_layout = b.layout();
    let lhs_data = lhs.data_mut();
    let a_data = a.data();
    let b_data = b.data();

    // Row-walk traversal over all three layouts (see zip_mut_with).
    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let lhs_step = traversal.last_axis_stride(lhs_layout);
    let a_step = traversal.last_axis_stride(a_layout);
    let b_step = traversal.last_axis_stride(b_layout);
    for row in 0..traversal.rows() {
        let base = traversal.base_index(row);
        let mut lhs_offset = lhs_layout.offset_of(base)? as isize;
        let mut a_offset = a_layout.offset_of(base)? as isize;
        let mut b_offset = b_layout.offset_of(base)? as isize;
        for _ in 0..traversal.inner() {
            f(
                &mut lhs_data[lhs_offset as usize],
                &a_data[a_offset as usize],
                &b_data[b_offset as usize],
            );
            lhs_offset += lhs_step;
            a_offset += a_step;
            b_offset += b_step;
        }
    }

    Ok(())
}

/// Mutably zip-map a view with two read-only operands and the logical index.
///
/// This combines [`zip2_mut_with`] with `Zip::indexed`-style logical coordinate
/// access while preserving the same shape and storage validation contract.
pub fn indexed_zip2_mut_with<T, A, B, F, const N: usize>(
    lhs: &mut ArrayViewMut<'_, T, N>,
    a: &ArrayView<'_, A, N>,
    b: &ArrayView<'_, B, N>,
    mut f: F,
) -> Result<()>
where
    F: FnMut([usize; N], &mut T, &A, &B),
{
    if lhs.shape() != a.shape() || lhs.shape() != b.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: a.shape().to_vec(),
        });
    }

    lhs.layout().validate_storage_len(lhs.data().len())?;
    a.layout().validate_storage_len(a.data().len())?;
    b.layout().validate_storage_len(b.data().len())?;
    if lhs.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "zip mutable output layout must not contain zero-stride aliasing".to_string(),
        });
    }

    let size = lhs.layout().checked_size()?;
    let shape = lhs.shape();
    let lhs_layout = lhs.layout();
    let a_layout = a.layout();
    let b_layout = b.layout();
    let lhs_data = lhs.data_mut();
    let a_data = a.data();
    let b_data = b.data();

    // Row-walk with an incrementally updated last coordinate (see
    // indexed_zip_mut_with).
    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let lhs_step = traversal.last_axis_stride(lhs_layout);
    let a_step = traversal.last_axis_stride(a_layout);
    let b_step = traversal.last_axis_stride(b_layout);
    for row in 0..traversal.rows() {
        let mut index = traversal.base_index(row);
        let mut lhs_offset = lhs_layout.offset_of(index)? as isize;
        let mut a_offset = a_layout.offset_of(index)? as isize;
        let mut b_offset = b_layout.offset_of(index)? as isize;
        for k in 0..traversal.inner() {
            if N > 0 {
                index[N - 1] = k;
            }
            f(
                index,
                &mut lhs_data[lhs_offset as usize],
                &a_data[a_offset as usize],
                &b_data[b_offset as usize],
            );
            lhs_offset += lhs_step;
            a_offset += a_step;
            b_offset += b_step;
        }
    }

    Ok(())
}
