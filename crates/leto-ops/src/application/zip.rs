use crate::application::index::RowMajorTraversal;
use leto::{ArrayView, ArrayViewMut, Layout, LetoError, Result};

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

mod sealed {
    pub trait ZipSources<const N: usize> {}
    pub trait ZipMutOutputs<S, F, const N: usize> {}
    pub trait IndexedZipMutOutputs<S, F, const N: usize> {}
}

/// A statically typed set of read-only views for a multi-input zip.
///
/// Implementations are provided for read-only ArrayView references and
/// tuples containing them. Tuple values are passed to the closure as one
/// nested value, so heterogeneous element types remain statically dispatched
/// without allocating a type-erased source list.
pub trait ZipSources<const N: usize>: sealed::ZipSources<N> {
    /// Values borrowed from the source set at one logical element.
    type Values;

    /// Physical offsets for the source set at one logical element.
    type Offsets: Copy;

    /// Contiguous source slices used by the dense traversal fast path.
    type Contiguous: Copy;

    /// Validate source storage and require the expected logical shape.
    fn validate(&self, expected_shape: [usize; N]) -> Result<()>;

    /// Compute physical source offsets for one logical index.
    fn offsets_at(&self, index: [usize; N]) -> Result<Self::Offsets>;

    /// Return the last-axis stride for each source.
    fn steps(&self) -> Self::Offsets;

    /// Advance all source offsets by one logical element.
    fn advance(&self, offsets: &mut Self::Offsets, steps: Self::Offsets);

    /// Return dense row-major slices when every source is contiguous.
    fn contiguous(&self) -> Option<Self::Contiguous>;

    /// Borrow source values from dense slices at one logical position.
    fn contiguous_values(sources: Self::Contiguous, index: usize) -> Self::Values;

    /// Borrow source values from strided storage at the given offsets.
    fn values(&self, offsets: Self::Offsets) -> Self::Values;
}

#[inline]
fn zip_offset<const N: usize>(layout: Layout<N>, index: [usize; N]) -> Result<isize> {
    isize::try_from(layout.offset_of(index)?).map_err(|_| LetoError::StorageError {
        reason: "zip layout offset exceeds isize range".to_string(),
    })
}

impl<'data, T, const N: usize> sealed::ZipSources<N> for &ArrayView<'data, T, N> {}

impl<'data, T, const N: usize> ZipSources<N> for &ArrayView<'data, T, N> {
    type Values = &'data T;
    type Offsets = isize;
    type Contiguous = &'data [T];

    #[inline]
    fn validate(&self, expected_shape: [usize; N]) -> Result<()> {
        let view = *self;
        if view.shape() != expected_shape {
            return Err(LetoError::ShapeMismatch {
                lhs: expected_shape.to_vec(),
                rhs: view.shape().to_vec(),
            });
        }
        view.layout().validate_storage_len(view.data().len())
    }

    #[inline]
    fn offsets_at(&self, index: [usize; N]) -> Result<Self::Offsets> {
        zip_offset((*self).layout(), index)
    }

    #[inline]
    fn steps(&self) -> Self::Offsets {
        if N == 0 {
            0
        } else {
            (*self).layout().strides[N - 1]
        }
    }

    #[inline]
    fn advance(&self, offsets: &mut Self::Offsets, steps: Self::Offsets) {
        *offsets += steps;
    }

    #[inline]
    fn contiguous(&self) -> Option<Self::Contiguous> {
        (*self).as_slice()
    }

    #[inline]
    fn contiguous_values(sources: Self::Contiguous, index: usize) -> Self::Values {
        &sources[index]
    }

    #[inline]
    fn values(&self, offsets: Self::Offsets) -> Self::Values {
        let view = *self;
        let offset =
            usize::try_from(offsets).expect("invariant: validated zip offset is non-negative");
        &view.data()[offset]
    }
}

impl<const N: usize> sealed::ZipSources<N> for () {}

impl<const N: usize> ZipSources<N> for () {
    type Values = ();
    type Offsets = ();
    type Contiguous = ();

    #[inline]
    fn validate(&self, _expected_shape: [usize; N]) -> Result<()> {
        Ok(())
    }

    #[inline]
    fn offsets_at(&self, _index: [usize; N]) -> Result<Self::Offsets> {
        Ok(())
    }

    #[inline]
    fn steps(&self) -> Self::Offsets {}

    #[inline]
    fn advance(&self, _offsets: &mut Self::Offsets, _steps: Self::Offsets) {}

    #[inline]
    fn contiguous(&self) -> Option<Self::Contiguous> {
        Some(())
    }

    #[inline]
    fn contiguous_values(_sources: Self::Contiguous, _index: usize) -> Self::Values {}

    #[inline]
    fn values(&self, _offsets: Self::Offsets) -> Self::Values {}
}

macro_rules! impl_zip_sources_for_tuple {
    ($($source:ident : $index:tt),+ $(,)?) => {
        impl<$($source,)+ const N: usize> sealed::ZipSources<N> for ($($source,)+)
        where
            $($source: ZipSources<N>,)+
        {
        }

        impl<$($source,)+ const N: usize> ZipSources<N> for ($($source,)+)
        where
            $($source: ZipSources<N>,)+
        {
            type Values = ($($source::Values,)+);
            type Offsets = ($($source::Offsets,)+);
            type Contiguous = ($($source::Contiguous,)+);

            #[inline]
            fn validate(&self, expected_shape: [usize; N]) -> Result<()> {
                $(self.$index.validate(expected_shape)?;)+
                Ok(())
            }

            #[inline]
            fn offsets_at(&self, index: [usize; N]) -> Result<Self::Offsets> {
                Ok(($(self.$index.offsets_at(index)?,)+))
            }

            #[inline]
            fn steps(&self) -> Self::Offsets {
                ($(self.$index.steps(),)+)
            }

            #[inline]
            fn advance(&self, offsets: &mut Self::Offsets, steps: Self::Offsets) {
                $(self.$index.advance(&mut offsets.$index, steps.$index);)+
            }

            #[inline]
            fn contiguous(&self) -> Option<Self::Contiguous> {
                Some(($(self.$index.contiguous()?,)+))
            }

            #[inline]
            fn contiguous_values(sources: Self::Contiguous, index: usize) -> Self::Values {
                ($(<$source as ZipSources<N>>::contiguous_values(
                    sources.$index,
                    index,
                ),)+)
            }

            #[inline]
            fn values(&self, offsets: Self::Offsets) -> Self::Values {
                ($(self.$index.values(offsets.$index),)+)
            }
        }
    };
}

impl_zip_sources_for_tuple!(A: 0, B: 1);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10, L: 11);

#[inline]
fn validate_output<T, const N: usize>(output: &ArrayViewMut<'_, T, N>) -> Result<()> {
    output.layout().validate_storage_len(output.data().len())?;
    if output.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "zip mutable output layout must not contain zero-stride aliasing".to_string(),
        });
    }
    Ok(())
}

fn zip_one<T, S, F, const N: usize>(
    output: &mut ArrayViewMut<'_, T, N>,
    sources: S,
    mut f: F,
) -> Result<()>
where
    S: ZipSources<N>,
    F: FnMut(&mut T, S::Values),
{
    validate_output(output)?;
    let shape = output.shape();
    sources.validate(shape)?;
    if let (Some(output_slice), Some(source_slices)) = (output.as_mut_slice(), sources.contiguous())
    {
        for (index, value) in output_slice.iter_mut().enumerate() {
            f(value, S::contiguous_values(source_slices, index));
        }
        return Ok(());
    }
    let size = output.layout().checked_size()?;
    let output_layout = output.layout();
    let output_data = output.data_mut();
    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let output_step = traversal.last_axis_stride(output_layout);
    let source_steps = sources.steps();

    for row in 0..traversal.rows() {
        let base = traversal.base_index(row);
        let mut output_offset = zip_offset(output_layout, base)?;
        let mut source_offsets = sources.offsets_at(base)?;
        for _ in 0..traversal.inner() {
            let output_index = usize::try_from(output_offset)
                .expect("invariant: validated zip output offset is non-negative");
            f(
                &mut output_data[output_index],
                sources.values(source_offsets),
            );
            output_offset += output_step;
            sources.advance(&mut source_offsets, source_steps);
        }
    }
    Ok(())
}

fn indexed_zip_one<T, S, F, const N: usize>(
    output: &mut ArrayViewMut<'_, T, N>,
    sources: S,
    mut f: F,
) -> Result<()>
where
    S: ZipSources<N>,
    F: FnMut([usize; N], &mut T, S::Values),
{
    validate_output(output)?;
    let shape = output.shape();
    sources.validate(shape)?;
    let size = output.layout().checked_size()?;
    let output_layout = output.layout();
    let output_data = output.data_mut();
    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let output_step = traversal.last_axis_stride(output_layout);
    let source_steps = sources.steps();

    for row in 0..traversal.rows() {
        let mut index = traversal.base_index(row);
        let mut output_offset = zip_offset(output_layout, index)?;
        let mut source_offsets = sources.offsets_at(index)?;
        for k in 0..traversal.inner() {
            if N > 0 {
                index[N - 1] = k;
            }
            let output_index = usize::try_from(output_offset)
                .expect("invariant: validated zip output offset is non-negative");
            f(
                index,
                &mut output_data[output_index],
                sources.values(source_offsets),
            );
            output_offset += output_step;
            sources.advance(&mut source_offsets, source_steps);
        }
    }
    Ok(())
}

/// A statically typed mutable-output zip operation.
///
/// Implementations cover one, two, and three mutable views. The operation
/// trait carries the source and closure types so the closure receives concrete
/// mutable references at each callsite without a higher-ranked trait bound over
/// a generic associated lifetime.
pub trait ZipMutOutputs<S, F, const N: usize>: sealed::ZipMutOutputs<S, F, N> {
    /// Mutate the output views with the source values.
    fn zip(outputs: Self, sources: S, f: F) -> Result<()>;
}

/// Indexed counterpart to [`ZipMutOutputs`].
pub trait IndexedZipMutOutputs<S, F, const N: usize>:
    sealed::IndexedZipMutOutputs<S, F, N>
{
    /// Mutate the output views with source values and logical indices.
    fn indexed_zip(outputs: Self, sources: S, f: F) -> Result<()>;
}

impl<'data, T, S, F, const N: usize> sealed::ZipMutOutputs<S, F, N> for ArrayViewMut<'data, T, N> {}

impl<'data, T, S, F, const N: usize> ZipMutOutputs<S, F, N> for ArrayViewMut<'data, T, N>
where
    S: ZipSources<N>,
    F: FnMut(&mut T, S::Values),
{
    fn zip(mut output: Self, sources: S, f: F) -> Result<()> {
        zip_one(&mut output, sources, f)
    }
}

macro_rules! impl_zip_mut_outputs_for_tuple {
    ($($type:ident => $data:ident : $index:tt),+ $(,)?) => {
        impl<'data, $($type,)+ S, F, const N: usize>
            sealed::ZipMutOutputs<S, F, N>
            for ($(ArrayViewMut<'data, $type, N>,)+)
        {
        }

        impl<'data, $($type,)+ S, F, const N: usize>
            ZipMutOutputs<S, F, N>
            for ($(ArrayViewMut<'data, $type, N>,)+)
        where
            S: ZipSources<N>,
            F: FnMut(($(&mut $type,)+), S::Values),
        {
            fn zip(mut outputs: Self, sources: S, mut f: F) -> Result<()> {
                let shape = outputs.0.shape();
                $(
                    if outputs.$index.shape() != shape {
                        return Err(LetoError::ShapeMismatch {
                            lhs: shape.to_vec(),
                            rhs: outputs.$index.shape().to_vec(),
                        });
                    }
                    validate_output(&outputs.$index)?;
                )+
                let size = outputs.0.layout().checked_size()?;
                if let ($(Some($data),)+) = ($(outputs.$index.as_mut_slice(),)+) {
                    if let Some(source_slices) = sources.contiguous() {
                        for index in 0..size {
                            f(
                                ($(&mut $data[index],)+),
                                S::contiguous_values(source_slices, index),
                            );
                        }
                        return Ok(());
                    }
                }
                let output_layouts = ($(outputs.$index.layout(),)+);
                let ($($data,)+) = ($(outputs.$index.data_mut(),)+);
                let Some(traversal) = RowMajorTraversal::new(size, shape) else {
                    return Ok(());
                };
                let output_steps = ($(traversal.last_axis_stride(output_layouts.$index),)+);
                let source_steps = sources.steps();

                for row in 0..traversal.rows() {
                    let base = traversal.base_index(row);
                    let mut output_offsets =
                        ($(zip_offset(output_layouts.$index, base)?,)+);
                    let mut source_offsets = sources.offsets_at(base)?;
                    for _ in 0..traversal.inner() {
                        f(
                            ($(&mut $data[usize::try_from(output_offsets.$index)
                                .expect("invariant: validated zip output offset is non-negative")],)+),
                            sources.values(source_offsets),
                        );
                        $(output_offsets.$index += output_steps.$index;)+
                        sources.advance(&mut source_offsets, source_steps);
                    }
                }
                Ok(())
            }

        }
    };
}

impl_zip_mut_outputs_for_tuple!(A => a: 0, B => b: 1);
impl_zip_mut_outputs_for_tuple!(A => a: 0, B => b: 1, C => c: 2);

impl<'a, 'data, T, S, F, const N: usize> sealed::ZipMutOutputs<S, F, N>
    for &'a mut ArrayViewMut<'data, T, N>
{
}

impl<'a, 'data, T, S, F, const N: usize> ZipMutOutputs<S, F, N>
    for &'a mut ArrayViewMut<'data, T, N>
where
    S: ZipSources<N>,
    F: FnMut(&mut T, S::Values),
{
    fn zip(outputs: Self, sources: S, f: F) -> Result<()> {
        zip_one(outputs, sources, f)
    }
}

impl<'data, T, S, F, const N: usize> sealed::IndexedZipMutOutputs<S, F, N>
    for ArrayViewMut<'data, T, N>
{
}

impl<'data, T, S, F, const N: usize> IndexedZipMutOutputs<S, F, N> for ArrayViewMut<'data, T, N>
where
    S: ZipSources<N>,
    F: FnMut([usize; N], &mut T, S::Values),
{
    fn indexed_zip(mut output: Self, sources: S, f: F) -> Result<()> {
        indexed_zip_one(&mut output, sources, f)
    }
}

macro_rules! impl_indexed_zip_mut_outputs_for_tuple {
    ($($type:ident => $data:ident : $index:tt),+ $(,)?) => {
        impl<'data, $($type,)+ S, F, const N: usize>
            sealed::IndexedZipMutOutputs<S, F, N>
            for ($(ArrayViewMut<'data, $type, N>,)+)
        {
        }

        impl<'data, $($type,)+ S, F, const N: usize>
            IndexedZipMutOutputs<S, F, N>
            for ($(ArrayViewMut<'data, $type, N>,)+)
        where
            S: ZipSources<N>,
            F: FnMut([usize; N], ($(&mut $type,)+), S::Values),
        {
            fn indexed_zip(mut outputs: Self, sources: S, mut f: F) -> Result<()> {
                let shape = outputs.0.shape();
                $(
                    if outputs.$index.shape() != shape {
                        return Err(LetoError::ShapeMismatch {
                            lhs: shape.to_vec(),
                            rhs: outputs.$index.shape().to_vec(),
                        });
                    }
                    validate_output(&outputs.$index)?;
                )+
                let size = outputs.0.layout().checked_size()?;
                let output_layouts = ($(outputs.$index.layout(),)+);
                let ($($data,)+) = ($(outputs.$index.data_mut(),)+);
                let Some(traversal) = RowMajorTraversal::new(size, shape) else {
                    return Ok(());
                };
                let output_steps = ($(traversal.last_axis_stride(output_layouts.$index),)+);
                let source_steps = sources.steps();

                for row in 0..traversal.rows() {
                    let mut index = traversal.base_index(row);
                    let mut output_offsets = ($(zip_offset(output_layouts.$index, index)?,)+);
                    let mut source_offsets = sources.offsets_at(index)?;
                    for k in 0..traversal.inner() {
                        if N > 0 {
                            index[N - 1] = k;
                        }
                        f(
                            index,
                            ($(&mut $data[usize::try_from(output_offsets.$index)
                                .expect("invariant: validated zip output offset is non-negative")],)+),
                            sources.values(source_offsets),
                        );
                        $(output_offsets.$index += output_steps.$index;)+
                        sources.advance(&mut source_offsets, source_steps);
                    }
                }
                Ok(())
            }
        }
    };
}

impl_indexed_zip_mut_outputs_for_tuple!(A => a: 0, B => b: 1);
impl_indexed_zip_mut_outputs_for_tuple!(A => a: 0, B => b: 1, C => c: 2);

impl<'a, 'data, T, S, F, const N: usize> sealed::IndexedZipMutOutputs<S, F, N>
    for &'a mut ArrayViewMut<'data, T, N>
{
}

impl<'a, 'data, T, S, F, const N: usize> IndexedZipMutOutputs<S, F, N>
    for &'a mut ArrayViewMut<'data, T, N>
where
    S: ZipSources<N>,
    F: FnMut([usize; N], &mut T, S::Values),
{
    fn indexed_zip(outputs: Self, sources: S, f: F) -> Result<()> {
        indexed_zip_one(outputs, sources, f)
    }
}

/// Mutably zip-map one or more output views with zero or more read-only views.
///
/// A single output is passed as `&mut T`; multiple outputs are passed as a
/// tuple. Read-only sources follow the same rule: one source is passed directly
/// and multiple sources are passed as a tuple. All arities are statically
/// dispatched, so the output/source family and closure monomorphize together.
pub fn zip_mut_with<O, S, F, const N: usize>(outputs: O, sources: S, f: F) -> Result<()>
where
    O: ZipMutOutputs<S, F, N>,
{
    O::zip(outputs, sources, f)
}

/// Mutably zip-map output views with zero or more read-only views and logical
/// row-major indices.
pub fn indexed_zip_mut_with<O, S, F, const N: usize>(outputs: O, sources: S, f: F) -> Result<()>
where
    O: IndexedZipMutOutputs<S, F, N>,
{
    O::indexed_zip(outputs, sources, f)
}
