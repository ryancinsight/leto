//! Shared window and destination validation for the 3D difference passes.

use core::ops::Range;

use leto::{LetoError, Result};

use super::leapfrog::Axis;
use super::window::PlaneWindow;

/// Runtime dst-shape check (errors on mismatch, not debug-only).
/// The refusal for a fused pass given no destination.
pub(super) fn no_destination() -> LetoError {
    LetoError::InvalidInput(
        "FiniteDifference3D: a fused pass needs at least one destination".into(),
    )
}

/// Planes a fourth-order derivative reaches on either side of its own.
pub(super) const FOURTH_ORDER_REACH: usize = 2;

/// Rejects windows that cannot serve a windowed pass over `planes`.
///
/// Every window must lie within the grid and share the destinations' second
/// and third extents. The destinations, the pointwise inputs and the y- and
/// z-derivative fields must hold every plane written; an x-derivative field
/// must also hold the planes its stencil reaches from them. An empty
/// `planes` writes nothing and reads nothing, so it asks only that the
/// windows be shaped consistently.
pub(super) fn check_windows<T>(
    grid_planes: usize,
    planes: &Range<usize>,
    (held, held_shape): &(Range<usize>, [usize; 3]),
    terms: &[(Axis, PlaneWindow<'_, T>)],
    pointwise: &[PlaneWindow<'_, T>],
) -> Result<()> {
    if planes.start > planes.end || planes.end > grid_planes {
        return Err(LetoError::InvalidInput(format!(
            "FiniteDifference3D: planes {planes:?} do not lie within 0..{grid_planes}"
        )));
    }
    let lanes = [held_shape[1], held_shape[2]];
    let reached = planes.start.saturating_sub(FOURTH_ORDER_REACH)
        ..planes
            .end
            .saturating_add(FOURTH_ORDER_REACH)
            .min(grid_planes);
    let windows = core::iter::once(("destination", held.clone(), *held_shape, planes.clone()))
        .chain(pointwise.iter().map(|window| {
            (
                "pointwise input",
                window.planes(),
                window.shape(),
                planes.clone(),
            )
        }))
        .chain(terms.iter().map(|(axis, window)| {
            let needed = match axis {
                Axis::X => reached.clone(),
                Axis::Y | Axis::Z => planes.clone(),
            };
            ("derivative field", window.planes(), window.shape(), needed)
        }));
    for (role, held, shape, needed) in windows {
        if held.end > grid_planes {
            return Err(LetoError::InvalidInput(format!(
                "FiniteDifference3D: a {role} holds planes {held:?}, past the grid's \
                 0..{grid_planes}"
            )));
        }
        if [shape[1], shape[2]] != lanes {
            return Err(LetoError::InvalidInput(format!(
                "FiniteDifference3D: a {role} has lanes {:?}, not the destination's {lanes:?}",
                [shape[1], shape[2]]
            )));
        }
        if !planes.is_empty() && (needed.start < held.start || needed.end > held.end) {
            return Err(LetoError::InvalidInput(format!(
                "FiniteDifference3D: a {role} holds planes {held:?} but writing {planes:?} \
                 needs {needed:?}"
            )));
        }
    }
    Ok(())
}

#[inline]
pub(super) fn assert_dst_shape(actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual != expected {
        return Err(LetoError::InvalidInput(format!(
            "FiniteDifference3D: dst shape {actual:?} does not match expected {expected:?}"
        )));
    }
    Ok(())
}
