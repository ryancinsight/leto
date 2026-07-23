use crate::application::index::{line_elements, RowMajorTraversal, TileGeometry};
use leto::{ArrayView, ArrayViewMut, Layout, LetoError, Result};

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
                                f(&mut lhs_data[lhs_off as usize], &rhs_data[rhs_off as usize]);
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

/// Fold two read-only views into one accumulator.
///
/// This is the reduction analogue of [`zip_mut_with`]. Both views must have the
/// same logical shape, and every logical pair is visited exactly once. Traversal
/// order is unspecified for strided layouts, so callers should use associative
/// or order-insensitive reductions when bitwise replay is required across
/// layouts.
pub fn zip_fold<T, U, Acc, F, const N: usize>(
    lhs: &ArrayView<'_, T, N>,
    rhs: &ArrayView<'_, U, N>,
    init: Acc,
    mut f: F,
) -> Result<Acc>
where
    F: FnMut(Acc, &T, &U) -> Acc,
{
    if lhs.shape() != rhs.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }

    lhs.layout().validate_storage_len(lhs.data().len())?;
    rhs.layout().validate_storage_len(rhs.data().len())?;

    if let (Some(lhs_slice), Some(rhs_slice)) = (lhs.as_slice(), rhs.as_slice()) {
        let mut acc = init;
        for (left, right) in lhs_slice.iter().zip(rhs_slice.iter()) {
            acc = f(acc, left, right);
        }
        return Ok(acc);
    }

    let size = lhs.layout().checked_size()?;
    let shape = lhs.shape();
    let lhs_layout = lhs.layout();
    let rhs_layout = rhs.layout();
    let lhs_data = lhs.data();
    let rhs_data = rhs.data();

    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(init);
    };
    let lhs_step = traversal.last_axis_stride(lhs_layout);
    let rhs_step = traversal.last_axis_stride(rhs_layout);
    let mut acc = init;
    for row in 0..traversal.rows() {
        let base = traversal.base_index(row);
        let mut lhs_offset = lhs_layout.offset_of(base)? as isize;
        let mut rhs_offset = rhs_layout.offset_of(base)? as isize;
        for _ in 0..traversal.inner() {
            acc = f(
                acc,
                &lhs_data[lhs_offset as usize],
                &rhs_data[rhs_offset as usize],
            );
            lhs_offset += lhs_step;
            rhs_offset += rhs_step;
        }
    }

    Ok(acc)
}

/// Fold one read-only view with the logical row-major index.
///
/// This is the indexed analogue of [`zip_fold`]. Every logical element is
/// visited exactly once, and the closure receives the logical index before the
/// read-only value. The traversal follows logical row-major order independent
/// of the backing storage layout.
pub fn indexed_fold<T, Acc, F, const N: usize>(
    view: &ArrayView<'_, T, N>,
    init: Acc,
    mut f: F,
) -> Result<Acc>
where
    F: FnMut(Acc, [usize; N], &T) -> Acc,
{
    view.layout().validate_storage_len(view.data().len())?;

    let size = view.layout().checked_size()?;
    let shape = view.shape();
    let layout = view.layout();
    let data = view.data();

    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(init);
    };
    let step = traversal.last_axis_stride(layout);
    let mut acc = init;
    for row in 0..traversal.rows() {
        let mut index = traversal.base_index(row);
        let mut offset = layout.offset_of(index)? as isize;
        for k in 0..traversal.inner() {
            if N > 0 {
                index[N - 1] = k;
            }
            acc = f(acc, index, &data[offset as usize]);
            offset += step;
        }
    }

    Ok(acc)
}

/// Fold one read-only view with the logical Fortran/column-major index order.
///
/// This is the column-major analogue of [`indexed_fold`]. It visits axis `0`
/// fastest, then axis `1`, and so on, independent of the backing storage
/// layout. Use this only when the logical visitation order is part of the
/// caller's contract.
pub fn indexed_fold_fortran<T, Acc, F, const N: usize>(
    view: &ArrayView<'_, T, N>,
    init: Acc,
    mut f: F,
) -> Result<Acc>
where
    F: FnMut(Acc, [usize; N], &T) -> Acc,
{
    view.layout().validate_storage_len(view.data().len())?;

    let size = view.layout().checked_size()?;
    if size == 0 {
        return Ok(init);
    }
    let shape = view.shape();
    let layout = view.layout();
    let data = view.data();
    let mut acc = init;
    for flat in 0..size {
        let mut index = [0usize; N];
        let mut remaining = flat;
        for axis in 0..N {
            if shape[axis] > 0 {
                index[axis] = remaining % shape[axis];
                remaining /= shape[axis];
            }
        }
        let offset = layout.offset_of(index)?;
        acc = f(acc, index, &data[offset]);
    }

    Ok(acc)
}

/// Mutably map elements in place with the logical row-major index.
///
/// This is the one-view indexed analogue of [`zip_mut_with`]. Every logical
/// element is visited exactly once, and the closure receives the logical index
/// before the mutable element.
pub fn indexed_map_inplace<T, F, const N: usize>(
    view: &mut ArrayViewMut<'_, T, N>,
    mut f: F,
) -> Result<()>
where
    F: FnMut([usize; N], &mut T),
{
    view.layout().validate_storage_len(view.data().len())?;
    if view.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "indexed mutable map layout must not contain zero-stride aliasing".to_string(),
        });
    }

    let size = view.layout().checked_size()?;
    let shape = view.shape();
    let layout = view.layout();
    let data = view.data_mut();

    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let step = traversal.last_axis_stride(layout);
    for row in 0..traversal.rows() {
        let mut index = traversal.base_index(row);
        let mut offset = layout.offset_of(index)? as isize;
        for k in 0..traversal.inner() {
            if N > 0 {
                index[N - 1] = k;
            }
            f(index, &mut data[offset as usize]);
            offset += step;
        }
    }

    Ok(())
}

/// Mutably visit a sparse list of logical coordinates in a view.
///
/// `coordinates` are interpreted in the view's logical index space. The
/// closure receives the ordinal position in the coordinate list, the logical
/// coordinate, and the mutable element at that coordinate. Repeated coordinates
/// are visited repeatedly in input order, which makes scatter-add style updates
/// explicit and deterministic.
pub fn coordinate_map_inplace<T, F, const N: usize>(
    view: &mut ArrayViewMut<'_, T, N>,
    coordinates: &[[usize; N]],
    mut f: F,
) -> Result<()>
where
    F: FnMut(usize, [usize; N], &mut T),
{
    view.layout().validate_storage_len(view.data().len())?;
    if view.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "coordinate mutable map layout must not contain zero-stride aliasing"
                .to_string(),
        });
    }

    let shape = view.shape();
    let layout = view.layout();
    let data = view.data_mut();
    for (ordinal, &index) in coordinates.iter().enumerate() {
        if index
            .iter()
            .zip(shape.iter())
            .any(|(&component, &axis)| component >= axis)
        {
            return Err(LetoError::OutOfBounds {
                index: index.to_vec(),
                shape: shape.to_vec(),
            });
        }
        let offset = layout.offset_of(index)?;
        f(ordinal, index, &mut data[offset]);
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CoordinateMapEntry<const N: usize> {
    ordinal: usize,
    index: [usize; N],
    offset: usize,
}

/// Prevalidated sparse-coordinate mutation plan for repeated view updates.
///
/// A plan binds a coordinate list to the exact logical layout of the mutable
/// view used to build it. Applying the plan then validates the target storage
/// and layout once, but does not recompute per-coordinate bounds checks or
/// physical offsets. Repeated coordinates are retained in input order, so
/// scatter-add style updates preserve the same deterministic semantics as
/// [`coordinate_map_inplace`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordinateMapPlan<const N: usize> {
    layout: Layout<N>,
    entries: Vec<CoordinateMapEntry<N>>,
}

impl<const N: usize> CoordinateMapPlan<N> {
    /// Build a sparse-coordinate plan for the exact layout of `view`.
    ///
    /// # Errors
    ///
    /// Returns a [`LetoError`] if `view` has invalid storage coverage,
    /// contains mutable zero-stride aliasing, or any coordinate is outside the
    /// view's logical shape.
    pub fn new<T>(view: &ArrayViewMut<'_, T, N>, coordinates: &[[usize; N]]) -> Result<Self> {
        view.layout().validate_storage_len(view.data().len())?;
        if view.layout().has_zero_stride_aliasing() {
            return Err(LetoError::StorageError {
                reason: "coordinate map plan layout must not contain zero-stride aliasing"
                    .to_string(),
            });
        }

        let shape = view.shape();
        let layout = view.layout();
        let mut entries = Vec::with_capacity(coordinates.len());
        for (ordinal, &index) in coordinates.iter().enumerate() {
            if index
                .iter()
                .zip(shape.iter())
                .any(|(&component, &axis)| component >= axis)
            {
                return Err(LetoError::OutOfBounds {
                    index: index.to_vec(),
                    shape: shape.to_vec(),
                });
            }
            let offset = layout.offset_of(index)?;
            entries.push(CoordinateMapEntry {
                ordinal,
                index,
                offset,
            });
        }

        Ok(Self { layout, entries })
    }

    /// Return the number of planned coordinate visits.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return true when the plan contains no coordinate visits.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the exact view layout this plan was built for.
    #[must_use]
    pub fn layout(&self) -> &Layout<N> {
        &self.layout
    }

    /// Apply the prevalidated coordinate plan to a mutable view.
    ///
    /// # Errors
    ///
    /// Returns a [`LetoError`] if `view` has invalid storage coverage, contains
    /// mutable zero-stride aliasing, or does not have the exact layout used to
    /// build this plan.
    pub fn apply<T, F>(&self, view: &mut ArrayViewMut<'_, T, N>, mut f: F) -> Result<()>
    where
        F: FnMut(usize, [usize; N], &mut T),
    {
        view.layout().validate_storage_len(view.data().len())?;
        if view.layout().has_zero_stride_aliasing() {
            return Err(LetoError::StorageError {
                reason: "coordinate map plan target layout must not contain zero-stride aliasing"
                    .to_string(),
            });
        }
        if view.layout() != self.layout {
            return Err(LetoError::StorageError {
                reason: "coordinate map plan target layout differs from planned layout".to_string(),
            });
        }

        let data = view.data_mut();
        for entry in &self.entries {
            f(entry.ordinal, entry.index, &mut data[entry.offset]);
        }
        Ok(())
    }
}

/// Build a sparse-coordinate mutation plan for repeated updates of `view`.
///
/// This is the planned companion to [`coordinate_map_inplace`].
///
/// # Errors
///
/// Returns a [`LetoError`] if `view` has invalid storage coverage, contains
/// mutable zero-stride aliasing, or any coordinate is outside the view's
/// logical shape.
pub fn coordinate_map_plan<T, const N: usize>(
    view: &ArrayViewMut<'_, T, N>,
    coordinates: &[[usize; N]],
) -> Result<CoordinateMapPlan<N>> {
    CoordinateMapPlan::new(view, coordinates)
}

/// Apply a prevalidated sparse-coordinate mutation plan to a mutable view.
///
/// # Errors
///
/// Returns a [`LetoError`] if `view` is not storage-valid, contains mutable
/// zero-stride aliasing, or has a different layout than the planned layout.
pub fn coordinate_map_plan_inplace<T, F, const N: usize>(
    view: &mut ArrayViewMut<'_, T, N>,
    plan: &CoordinateMapPlan<N>,
    f: F,
) -> Result<()>
where
    F: FnMut(usize, [usize; N], &mut T),
{
    plan.apply(view, f)
}

/// Mutably map four views in place with the logical row-major index.
///
/// This is the multi-output analogue of [`indexed_map_inplace`]. All four views
/// must share the same logical shape, and each output layout must be free of
/// zero-stride aliasing so every logical element has one mutable destination.
pub fn indexed_map4_inplace<A, B, C, D, F, const N: usize>(
    a: &mut ArrayViewMut<'_, A, N>,
    b: &mut ArrayViewMut<'_, B, N>,
    c: &mut ArrayViewMut<'_, C, N>,
    d: &mut ArrayViewMut<'_, D, N>,
    mut f: F,
) -> Result<()>
where
    F: FnMut([usize; N], &mut A, &mut B, &mut C, &mut D),
{
    if a.shape() != b.shape() || a.shape() != c.shape() || a.shape() != d.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        });
    }

    a.layout().validate_storage_len(a.data().len())?;
    b.layout().validate_storage_len(b.data().len())?;
    c.layout().validate_storage_len(c.data().len())?;
    d.layout().validate_storage_len(d.data().len())?;
    if a.layout().has_zero_stride_aliasing()
        || b.layout().has_zero_stride_aliasing()
        || c.layout().has_zero_stride_aliasing()
        || d.layout().has_zero_stride_aliasing()
    {
        return Err(LetoError::StorageError {
            reason: "indexed multi-output map layouts must not contain zero-stride aliasing"
                .to_string(),
        });
    }

    let size = a.layout().checked_size()?;
    let shape = a.shape();
    let a_layout = a.layout();
    let b_layout = b.layout();
    let c_layout = c.layout();
    let d_layout = d.layout();
    let a_data = a.data_mut();
    let b_data = b.data_mut();
    let c_data = c.data_mut();
    let d_data = d.data_mut();

    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let a_step = traversal.last_axis_stride(a_layout);
    let b_step = traversal.last_axis_stride(b_layout);
    let c_step = traversal.last_axis_stride(c_layout);
    let d_step = traversal.last_axis_stride(d_layout);
    for row in 0..traversal.rows() {
        let mut index = traversal.base_index(row);
        let mut a_offset = a_layout.offset_of(index)? as isize;
        let mut b_offset = b_layout.offset_of(index)? as isize;
        let mut c_offset = c_layout.offset_of(index)? as isize;
        let mut d_offset = d_layout.offset_of(index)? as isize;
        for k in 0..traversal.inner() {
            if N > 0 {
                index[N - 1] = k;
            }
            f(
                index,
                &mut a_data[a_offset as usize],
                &mut b_data[b_offset as usize],
                &mut c_data[c_offset as usize],
                &mut d_data[d_offset as usize],
            );
            a_offset += a_step;
            b_offset += b_step;
            c_offset += c_step;
            d_offset += d_step;
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

/// Mutably zip-map a view with elements from three read-only views in place.
///
/// This is the four-operand analogue of [`zip_mut_with`] (`ndarray`'s
/// `Zip::from(out).and(a).and(b).and(c)`). `lhs` owns mutation; `a`, `b`, and
/// `c` are read-only. All four views must share the same logical shape.
/// Strided inputs are traversed by logical row-major index, independent of
/// backing layout.
pub fn zip3_mut_with<T, A, B, C, F, const N: usize>(
    lhs: &mut ArrayViewMut<'_, T, N>,
    a: &ArrayView<'_, A, N>,
    b: &ArrayView<'_, B, N>,
    c: &ArrayView<'_, C, N>,
    mut f: F,
) -> Result<()>
where
    F: FnMut(&mut T, &A, &B, &C),
{
    if lhs.shape() != a.shape() || lhs.shape() != b.shape() || lhs.shape() != c.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: a.shape().to_vec(),
        });
    }

    lhs.layout().validate_storage_len(lhs.data().len())?;
    a.layout().validate_storage_len(a.data().len())?;
    b.layout().validate_storage_len(b.data().len())?;
    c.layout().validate_storage_len(c.data().len())?;
    if lhs.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "zip mutable output layout must not contain zero-stride aliasing".to_string(),
        });
    }

    if let (Some(lhs_slice), Some(a_slice), Some(b_slice), Some(c_slice)) =
        (lhs.as_mut_slice(), a.as_slice(), b.as_slice(), c.as_slice())
    {
        for (((left, av), bv), cv) in lhs_slice
            .iter_mut()
            .zip(a_slice.iter())
            .zip(b_slice.iter())
            .zip(c_slice.iter())
        {
            f(left, av, bv, cv);
        }
        return Ok(());
    }

    let size = lhs.layout().checked_size()?;
    let shape = lhs.shape();
    let lhs_layout = lhs.layout();
    let a_layout = a.layout();
    let b_layout = b.layout();
    let c_layout = c.layout();
    let lhs_data = lhs.data_mut();
    let a_data = a.data();
    let b_data = b.data();
    let c_data = c.data();

    // Row-walk traversal over all four layouts (see zip_mut_with).
    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let lhs_step = traversal.last_axis_stride(lhs_layout);
    let a_step = traversal.last_axis_stride(a_layout);
    let b_step = traversal.last_axis_stride(b_layout);
    let c_step = traversal.last_axis_stride(c_layout);
    for row in 0..traversal.rows() {
        let base = traversal.base_index(row);
        let mut lhs_offset = lhs_layout.offset_of(base)? as isize;
        let mut a_offset = a_layout.offset_of(base)? as isize;
        let mut b_offset = b_layout.offset_of(base)? as isize;
        let mut c_offset = c_layout.offset_of(base)? as isize;
        for _ in 0..traversal.inner() {
            f(
                &mut lhs_data[lhs_offset as usize],
                &a_data[a_offset as usize],
                &b_data[b_offset as usize],
                &c_data[c_offset as usize],
            );
            lhs_offset += lhs_step;
            a_offset += a_step;
            b_offset += b_step;
            c_offset += c_step;
        }
    }

    Ok(())
}

/// Mutably zip-map a view with elements from five read-only views in place.
///
/// This is the six-operand analogue of [`zip_mut_with`]. `lhs` owns mutation;
/// `a` through `e` are read-only. All six views must share the same logical
/// shape. Strided inputs are traversed by logical row-major index, independent
/// of backing layout.
pub fn zip5_mut_with<T, A, B, C, D, E, F, const N: usize>(
    lhs: &mut ArrayViewMut<'_, T, N>,
    a: &ArrayView<'_, A, N>,
    b: &ArrayView<'_, B, N>,
    c: &ArrayView<'_, C, N>,
    d: &ArrayView<'_, D, N>,
    e: &ArrayView<'_, E, N>,
    mut f: F,
) -> Result<()>
where
    F: FnMut(&mut T, &A, &B, &C, &D, &E),
{
    if lhs.shape() != a.shape()
        || lhs.shape() != b.shape()
        || lhs.shape() != c.shape()
        || lhs.shape() != d.shape()
        || lhs.shape() != e.shape()
    {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: a.shape().to_vec(),
        });
    }

    lhs.layout().validate_storage_len(lhs.data().len())?;
    a.layout().validate_storage_len(a.data().len())?;
    b.layout().validate_storage_len(b.data().len())?;
    c.layout().validate_storage_len(c.data().len())?;
    d.layout().validate_storage_len(d.data().len())?;
    e.layout().validate_storage_len(e.data().len())?;
    if lhs.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "zip mutable output layout must not contain zero-stride aliasing".to_string(),
        });
    }

    if let (
        Some(lhs_slice),
        Some(a_slice),
        Some(b_slice),
        Some(c_slice),
        Some(d_slice),
        Some(e_slice),
    ) = (
        lhs.as_mut_slice(),
        a.as_slice(),
        b.as_slice(),
        c.as_slice(),
        d.as_slice(),
        e.as_slice(),
    ) {
        for (((((left, av), bv), cv), dv), ev) in lhs_slice
            .iter_mut()
            .zip(a_slice.iter())
            .zip(b_slice.iter())
            .zip(c_slice.iter())
            .zip(d_slice.iter())
            .zip(e_slice.iter())
        {
            f(left, av, bv, cv, dv, ev);
        }
        return Ok(());
    }

    let size = lhs.layout().checked_size()?;
    let shape = lhs.shape();
    let lhs_layout = lhs.layout();
    let a_layout = a.layout();
    let b_layout = b.layout();
    let c_layout = c.layout();
    let d_layout = d.layout();
    let e_layout = e.layout();
    let lhs_data = lhs.data_mut();
    let a_data = a.data();
    let b_data = b.data();
    let c_data = c.data();
    let d_data = d.data();
    let e_data = e.data();

    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let lhs_step = traversal.last_axis_stride(lhs_layout);
    let a_step = traversal.last_axis_stride(a_layout);
    let b_step = traversal.last_axis_stride(b_layout);
    let c_step = traversal.last_axis_stride(c_layout);
    let d_step = traversal.last_axis_stride(d_layout);
    let e_step = traversal.last_axis_stride(e_layout);
    for row in 0..traversal.rows() {
        let base = traversal.base_index(row);
        let mut lhs_offset = lhs_layout.offset_of(base)? as isize;
        let mut a_offset = a_layout.offset_of(base)? as isize;
        let mut b_offset = b_layout.offset_of(base)? as isize;
        let mut c_offset = c_layout.offset_of(base)? as isize;
        let mut d_offset = d_layout.offset_of(base)? as isize;
        let mut e_offset = e_layout.offset_of(base)? as isize;
        for _ in 0..traversal.inner() {
            f(
                &mut lhs_data[lhs_offset as usize],
                &a_data[a_offset as usize],
                &b_data[b_offset as usize],
                &c_data[c_offset as usize],
                &d_data[d_offset as usize],
                &e_data[e_offset as usize],
            );
            lhs_offset += lhs_step;
            a_offset += a_step;
            b_offset += b_step;
            c_offset += c_step;
            d_offset += d_step;
            e_offset += e_step;
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

/// Mutably zip-map a view with four read-only operands and the logical index.
///
/// This is the indexed analogue of a five-operand mutable zip. It is intended
/// for kernels where the logical coordinate participates in the expression
/// without allocating an index array.
pub fn indexed_zip4_mut_with<T, A, B, C, D, F, const N: usize>(
    lhs: &mut ArrayViewMut<'_, T, N>,
    a: &ArrayView<'_, A, N>,
    b: &ArrayView<'_, B, N>,
    c: &ArrayView<'_, C, N>,
    d: &ArrayView<'_, D, N>,
    mut f: F,
) -> Result<()>
where
    F: FnMut([usize; N], &mut T, &A, &B, &C, &D),
{
    if lhs.shape() != a.shape()
        || lhs.shape() != b.shape()
        || lhs.shape() != c.shape()
        || lhs.shape() != d.shape()
    {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: a.shape().to_vec(),
        });
    }

    lhs.layout().validate_storage_len(lhs.data().len())?;
    a.layout().validate_storage_len(a.data().len())?;
    b.layout().validate_storage_len(b.data().len())?;
    c.layout().validate_storage_len(c.data().len())?;
    d.layout().validate_storage_len(d.data().len())?;
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
    let c_layout = c.layout();
    let d_layout = d.layout();
    let lhs_data = lhs.data_mut();
    let a_data = a.data();
    let b_data = b.data();
    let c_data = c.data();
    let d_data = d.data();

    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let lhs_step = traversal.last_axis_stride(lhs_layout);
    let a_step = traversal.last_axis_stride(a_layout);
    let b_step = traversal.last_axis_stride(b_layout);
    let c_step = traversal.last_axis_stride(c_layout);
    let d_step = traversal.last_axis_stride(d_layout);
    for row in 0..traversal.rows() {
        let mut index = traversal.base_index(row);
        let mut lhs_offset = lhs_layout.offset_of(index)? as isize;
        let mut a_offset = a_layout.offset_of(index)? as isize;
        let mut b_offset = b_layout.offset_of(index)? as isize;
        let mut c_offset = c_layout.offset_of(index)? as isize;
        let mut d_offset = d_layout.offset_of(index)? as isize;
        for k in 0..traversal.inner() {
            if N > 0 {
                index[N - 1] = k;
            }
            f(
                index,
                &mut lhs_data[lhs_offset as usize],
                &a_data[a_offset as usize],
                &b_data[b_offset as usize],
                &c_data[c_offset as usize],
                &d_data[d_offset as usize],
            );
            lhs_offset += lhs_step;
            a_offset += a_step;
            b_offset += b_step;
            c_offset += c_step;
            d_offset += d_step;
        }
    }

    Ok(())
}
