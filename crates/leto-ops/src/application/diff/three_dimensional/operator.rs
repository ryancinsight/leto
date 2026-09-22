//! The 3-D finite-difference operator: its spacings, and the dispatch from a
//! stencil scheme to the stencil that implements it.
//!
//! The stencils live in [`super::central`], [`super::fourth_order`] and
//! [`super::staggered`]; this
//! drives them and checks the destination shape.

use core::ops::Range;

use eunomia::{FloatElement, NumericElement, RealField};
use leto::{ArrayView3, ArrayViewMut3, LetoError, Result};

use super::central::{
    central2_x_into, central2_y_into, central2_z_into, central6_x_into, central6_y_into,
    central6_z_into,
};
use super::fourth_order::{central4_divergence_into, central4_into, central4_map_into};
use super::leapfrog::Axis;
use super::staggered::{
    staggered_backward_x_into, staggered_backward_y_into, staggered_backward_z_into,
    staggered_forward_x_into, staggered_forward_y_into, staggered_forward_z_into,
};
use super::window::{PlaneWindow, PlaneWindowMut};
use super::FiniteDifference3DScheme;

/// Generic 3-D finite-difference operator acting on `u(x, y, z)`.
#[derive(Debug, Clone, Copy)]
pub struct FiniteDifference3D<T> {
    scheme: FiniteDifference3DScheme,
    dx: T,
    dy: T,
    dz: T,
}

impl<T: RealField + FloatElement + Copy> FiniteDifference3D<T> {
    /// Construct an operator from a stencil scheme and per-axis spacings.
    ///
    /// # Errors
    /// - [`LetoError::InvalidInput`] when any of `dx`, `dy`, `dz` is not
    ///   strictly positive.
    pub fn new(scheme: FiniteDifference3DScheme, dx: T, dy: T, dz: T) -> Result<Self> {
        let zero = <T as NumericElement>::ZERO;
        if dx <= zero || dy <= zero || dz <= zero {
            return Err(LetoError::InvalidInput(
                "FiniteDifference3D: dx, dy, dz must all be strictly positive".into(),
            ));
        }
        Ok(Self { scheme, dx, dy, dz })
    }

    /// 2nd-order central-difference scheme.
    pub fn central_second_order(dx: T, dy: T, dz: T) -> Result<Self> {
        Self::new(FiniteDifference3DScheme::CentralSecondOrder, dx, dy, dz)
    }

    /// 4th-order central-difference scheme.
    pub fn central_fourth_order(dx: T, dy: T, dz: T) -> Result<Self> {
        Self::new(FiniteDifference3DScheme::CentralFourthOrder, dx, dy, dz)
    }

    /// 6th-order central-difference scheme.
    pub fn central_sixth_order(dx: T, dy: T, dz: T) -> Result<Self> {
        Self::new(FiniteDifference3DScheme::CentralSixthOrder, dx, dy, dz)
    }

    /// Yee staggered forward scheme.
    pub fn staggered_forward(dx: T, dy: T, dz: T) -> Result<Self> {
        Self::new(FiniteDifference3DScheme::StaggeredForward, dx, dy, dz)
    }

    /// Yee staggered backward scheme (kwavers-side convention).
    pub fn staggered_backward(dx: T, dy: T, dz: T) -> Result<Self> {
        Self::new(FiniteDifference3DScheme::StaggeredBackward, dx, dy, dz)
    }

    /// Returns the configured stencil scheme.
    #[must_use]
    pub fn scheme(&self) -> FiniteDifference3DScheme {
        self.scheme
    }

    /// Returns the per-axis grid spacing `(dx, dy, dz)`.
    #[must_use]
    pub fn spacing(&self) -> (T, T, T) {
        (self.dx, self.dy, self.dz)
    }

    /// Stencil width = number of grid points used by the interior kernel.
    #[must_use]
    pub fn stencil_width(&self) -> usize {
        match self.scheme {
            FiniteDifference3DScheme::CentralSecondOrder => 3,
            FiniteDifference3DScheme::CentralFourthOrder => 5,
            FiniteDifference3DScheme::CentralSixthOrder => 7,
            FiniteDifference3DScheme::StaggeredForward
            | FiniteDifference3DScheme::StaggeredBackward => 2,
        }
    }

    /// Apply ∂/∂x into a pre-allocated destination.
    ///
    /// # Errors
    /// - [`LetoError::InvalidInput`] when the diff axis has fewer than the
    ///   minimum required points for the chosen scheme, or when the dst shape
    ///   does not match the scheme's documented contract.
    pub fn apply_x_into(&self, field: ArrayView3<T>, dst: &mut ArrayViewMut3<'_, T>) -> Result<()> {
        let [nx, ny, nz] = field.shape();
        match self.scheme {
            FiniteDifference3DScheme::CentralSecondOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central2_x_into(field, dst, nx, ny, nz, self.dx)
            }
            FiniteDifference3DScheme::CentralFourthOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central4_into(field, dst, Axis::X, self.dx)
            }
            FiniteDifference3DScheme::CentralSixthOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central6_x_into(field, dst, nx, ny, nz, self.dx)
            }
            FiniteDifference3DScheme::StaggeredForward => {
                assert_dst_shape(&dst.shape(), &[nx - 1, ny, nz])?;
                staggered_forward_x_into(field, dst, nx, ny, nz, self.dx)
            }
            FiniteDifference3DScheme::StaggeredBackward => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                staggered_backward_x_into(field, dst, nx, ny, nz, self.dx)
            }
        }
    }

    /// Sum the three axis derivatives of three fields into one destination:
    /// `dst = ∂fields[0]/∂x + ∂fields[1]/∂y + ∂fields[2]/∂z`.
    ///
    /// The fused form reads each field once per output lane instead of
    /// writing three per-axis buffers a caller then adds, which is where a
    /// divergence spends its time once the stencils are lane-swept: at 64
    /// cubed it moves 8 MB against the composed form's 20 MB. Values are
    /// bit-identical to applying each axis separately and summing in x, y, z
    /// order.
    ///
    /// # Errors
    /// - [`LetoError::InvalidInput`] when the fields and `dst` do not share
    ///   one shape, or when the scheme is not `CentralFourthOrder`. The other
    ///   schemes have no fused kernel yet; compose [`Self::apply_x_into`] and
    ///   its siblings and sum, or extend `fourth_order` the same way.
    pub fn divergence_into(
        &self,
        fields: [ArrayView3<'_, T>; 3],
        dst: &mut ArrayViewMut3<'_, T>,
    ) -> Result<()> {
        let shape = fields[0].shape();
        for field in &fields[1..] {
            assert_dst_shape(&field.shape(), &shape)?;
        }
        assert_dst_shape(&dst.shape(), &shape)?;
        match self.scheme {
            FiniteDifference3DScheme::CentralFourthOrder => {
                central4_divergence_into(fields, dst, [self.dx, self.dy, self.dz])
            }
            other => Err(LetoError::InvalidInput(format!(
                "divergence_into has no fused kernel for {other:?}; apply each \
                 axis separately and sum"
            ))),
        }
    }

    /// Combine several axis derivatives and pointwise fields into one
    /// destination in a single pass: `dst[i] = combine(derivatives, values)`,
    /// where `derivatives[j]` is `∂terms[j].1/∂terms[j].0` at that lane and
    /// `values[k]` is `pointwise[k]` there.
    ///
    /// [`Self::divergence_into`] is the sum of three axis derivatives; this
    /// serves the combinations that are not a sum. An elastic shear stress
    /// `μ (∂u/∂b + ∂v/∂a)` is `N = 2, M = 1`: it reads two fields and the
    /// scale once per lane, where sweeping each axis into a buffer and
    /// scaling the sum moves twice the traffic. Each derivative is the value
    /// the corresponding [`Self::apply_x_into`] sweep would have written.
    ///
    /// Terms name distinct axes in the cases this serves; a repeated axis is
    /// read twice and both values are handed to `combine`.
    ///
    /// # Errors
    /// - [`LetoError::InvalidInput`] when the fields, the pointwise inputs
    ///   and `dst` do not share one shape, or when the scheme is not
    ///   `CentralFourthOrder`. The other schemes have no fused kernel yet;
    ///   sweep each axis separately and combine, or extend `fourth_order` the
    ///   same way.
    pub fn map_axis_derivatives<const N: usize, const M: usize, F>(
        &self,
        terms: [(Axis, ArrayView3<'_, T>); N],
        pointwise: [ArrayView3<'_, T>; M],
        dst: &mut ArrayViewMut3<'_, T>,
        combine: F,
    ) -> Result<()>
    where
        F: Fn([T; N], [T; M]) -> T + Send + Sync,
    {
        self.map_axis_derivatives_many(terms, pointwise, [dst], |derivatives, values| {
            [combine(derivatives, values)]
        })
    }

    /// [`Self::map_axis_derivatives`] writing `K` destinations from one
    /// derivative pass: `combine` returns the value for each.
    ///
    /// Where several results read the same axis derivatives -- the diagonal
    /// of an elastic stress tensor reads the same three normal strains, and
    /// all six stresses read the nine displacement gradients -- this sweeps
    /// the stencils once and writes every result, where a call per
    /// destination sweeps them again and re-reads every field. At 64 cubed
    /// the diagonal alone is 16 MB of traffic this way against 36 MB for
    /// three single-destination calls. Any `K` is one parallel region.
    ///
    /// # Errors
    /// - [`LetoError::InvalidInput`] when the fields, the pointwise inputs
    ///   and the destinations do not share one shape, when `K` is zero, or
    ///   when the scheme is not `CentralFourthOrder`. The other schemes have
    ///   no fused kernel yet; sweep each axis separately and combine, or
    ///   extend `fourth_order` the same way.
    pub fn map_axis_derivatives_many<const N: usize, const M: usize, const K: usize, F>(
        &self,
        terms: [(Axis, ArrayView3<'_, T>); N],
        pointwise: [ArrayView3<'_, T>; M],
        dst: [&mut ArrayViewMut3<'_, T>; K],
        combine: F,
    ) -> Result<()>
    where
        F: Fn([T; N], [T; M]) -> [T; K] + Send + Sync,
    {
        let Some(first) = dst.first() else {
            return Err(no_destination());
        };
        let shape = first.shape();
        for destination in &dst[1..] {
            assert_dst_shape(&destination.shape(), &shape)?;
        }
        for (_, field) in &terms {
            assert_dst_shape(&field.shape(), &shape)?;
        }
        for field in &pointwise {
            assert_dst_shape(&field.shape(), &shape)?;
        }
        self.map_axis_derivatives_in_windows(
            shape[0],
            0..shape[0],
            terms.map(|(axis, field)| (axis, PlaneWindow::whole(field))),
            pointwise.map(PlaneWindow::whole),
            dst.map(PlaneWindowMut::whole),
            combine,
        )
    }

    /// [`Self::map_axis_derivatives_many`] over the grid planes `planes` of a
    /// grid of `grid_planes` x-planes, where each field and each destination
    /// hold only a window of those planes. The destinations hold the same
    /// planes.
    ///
    /// A derivative at grid plane `x` reads the planes its stencil reaches
    /// and takes the stencil the grid's extent gives `x`, whatever window
    /// holds the field, so each plane written is bit-identical to the same
    /// plane of a whole-grid call on the same values. Only `planes` is
    /// written.
    ///
    /// That is what lets a chain of passes run a slab of planes at a time
    /// through buffers the size of a slab. A consumer that writes
    /// intermediates and reads them back -- stresses, then their divergence
    /// -- keeps the intermediates in a window it reuses for every slab, so
    /// they stay in cache instead of making a grid-sized round trip through
    /// DRAM; see [`PlaneWindow`].
    ///
    /// # Errors
    /// [`LetoError::InvalidInput`] when
    /// - `K` is zero, or the destinations do not hold the same planes in one
    ///   shape;
    /// - `planes` runs backwards or past `grid_planes`, or a window reaches
    ///   past `grid_planes`;
    /// - a field or pointwise input differs from the destinations in its
    ///   second or third extent;
    /// - a destination, a pointwise input, or a y- or z-derivative field does
    ///   not hold every plane in `planes`, or an x-derivative field does not
    ///   hold every plane its stencil reaches from them;
    /// - the scheme is not `CentralFourthOrder`.
    pub fn map_axis_derivatives_in_windows<const N: usize, const M: usize, const K: usize, F>(
        &self,
        grid_planes: usize,
        planes: Range<usize>,
        terms: [(Axis, PlaneWindow<'_, T>); N],
        pointwise: [PlaneWindow<'_, T>; M],
        dst: [PlaneWindowMut<'_, '_, T>; K],
        combine: F,
    ) -> Result<()>
    where
        F: Fn([T; N], [T; M]) -> [T; K] + Send + Sync,
    {
        let destinations = dst
            .each_ref()
            .map(|window| (window.planes(), window.shape()));
        let Some(held) = destinations.first() else {
            return Err(no_destination());
        };
        if destinations.iter().any(|destination| destination != held) {
            return Err(LetoError::InvalidInput(format!(
                "FiniteDifference3D: every destination must hold the same planes in one \
                 shape, not {destinations:?}"
            )));
        }
        check_windows(grid_planes, &planes, held, &terms, &pointwise)?;
        match self.scheme {
            FiniteDifference3DScheme::CentralFourthOrder => central4_map_into(
                terms,
                pointwise,
                dst,
                grid_planes,
                planes,
                [self.dx, self.dy, self.dz],
                combine,
            ),
            other => Err(LetoError::InvalidInput(format!(
                "map_axis_derivatives has no fused kernel for {other:?}; sweep \
                 each axis separately and combine"
            ))),
        }
    }

    /// Apply ∂/∂y into a pre-allocated destination.
    /// # Errors
    /// See [`Self::apply_x_into`].
    pub fn apply_y_into(&self, field: ArrayView3<T>, dst: &mut ArrayViewMut3<'_, T>) -> Result<()> {
        let [nx, ny, nz] = field.shape();
        match self.scheme {
            FiniteDifference3DScheme::CentralSecondOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central2_y_into(field, dst, nx, ny, nz, self.dy)
            }
            FiniteDifference3DScheme::CentralFourthOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central4_into(field, dst, Axis::Y, self.dy)
            }
            FiniteDifference3DScheme::CentralSixthOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central6_y_into(field, dst, nx, ny, nz, self.dy)
            }
            FiniteDifference3DScheme::StaggeredForward => {
                assert_dst_shape(&dst.shape(), &[nx, ny - 1, nz])?;
                staggered_forward_y_into(field, dst, nx, ny, nz, self.dy)
            }
            FiniteDifference3DScheme::StaggeredBackward => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                staggered_backward_y_into(field, dst, nx, ny, nz, self.dy)
            }
        }
    }

    /// Apply ∂/∂z into a pre-allocated destination.
    /// # Errors
    /// See [`Self::apply_x_into`].
    pub fn apply_z_into(&self, field: ArrayView3<T>, dst: &mut ArrayViewMut3<'_, T>) -> Result<()> {
        let [nx, ny, nz] = field.shape();
        match self.scheme {
            FiniteDifference3DScheme::CentralSecondOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central2_z_into(field, dst, nx, ny, nz, self.dz)
            }
            FiniteDifference3DScheme::CentralFourthOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central4_into(field, dst, Axis::Z, self.dz)
            }
            FiniteDifference3DScheme::CentralSixthOrder => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                central6_z_into(field, dst, nx, ny, nz, self.dz)
            }
            FiniteDifference3DScheme::StaggeredForward => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz - 1])?;
                staggered_forward_z_into(field, dst, nx, ny, nz, self.dz)
            }
            FiniteDifference3DScheme::StaggeredBackward => {
                assert_dst_shape(&dst.shape(), &[nx, ny, nz])?;
                staggered_backward_z_into(field, dst, nx, ny, nz, self.dz)
            }
        }
    }
}

/// Runtime dst-shape check (errors on mismatch, not debug-only).
/// The refusal for a fused pass given no destination.
fn no_destination() -> LetoError {
    LetoError::InvalidInput(
        "FiniteDifference3D: a fused pass needs at least one destination".into(),
    )
}

/// Planes a fourth-order derivative reaches on either side of its own.
const FOURTH_ORDER_REACH: usize = 2;

/// Rejects windows that cannot serve a windowed pass over `planes`.
///
/// Every window must lie within the grid and share the destinations' second
/// and third extents. The destinations, the pointwise inputs and the y- and
/// z-derivative fields must hold every plane written; an x-derivative field
/// must also hold the planes its stencil reaches from them. An empty
/// `planes` writes nothing and reads nothing, so it asks only that the
/// windows be shaped consistently.
fn check_windows<T>(
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
fn assert_dst_shape(actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual != expected {
        return Err(LetoError::InvalidInput(format!(
            "FiniteDifference3D: dst shape {actual:?} does not match expected {expected:?}"
        )));
    }
    Ok(())
}
