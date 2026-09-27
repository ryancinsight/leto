use crate::application::index::validate_mutable_output;
use leto::{ArrayViewMut, Layout, LetoError, Result};

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
    validate_mutable_output(view, "coordinate mutable map")?;

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
        validate_mutable_output(view, "coordinate map plan")?;

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
        validate_mutable_output(view, "coordinate map plan target")?;
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
