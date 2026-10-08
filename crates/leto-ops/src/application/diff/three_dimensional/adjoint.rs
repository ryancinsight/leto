//! Transpose (adjoint) sweeps of the fixed-scheme first derivatives.
//!
//! For a forward sweep `y = A f`, the adjoint maps an upstream lane `u` to
//! `v = Aᵀ u`, the gradient of `⟨u, A f⟩` with respect to `f`. Autograd
//! backward passes are the consumer: a linear operator's pullback is its
//! transpose, applied to the output gradient.
//!
//! # Derivation shape
//!
//! Each forward row is a small linear form in `f`, so each output lane `v[j]`
//! is a sum of predicated terms — one per forward row whose stencil touches
//! `f[j]`, weighted by that row's tap for `f[j]`. A lane reads upstream
//! lanes `u[j−3..=j+3]` at most (central-6), and every predicate is a pure
//! function of `(scheme, n, j)`, so the lane formula is closed-form: no
//! iteration, no row-type dispatch per lane beyond integer comparison.
//!
//! # Bit-exactness contract with the device mirror
//!
//! Lanes accumulate predicated terms in canonical order — wall taps, then
//! second-, fourth-, then sixth-order row taps — as a left-associated sum
//! starting from zero, each term parenthesized `(c * u) * inv_scale` with an
//! exact small-integer coefficient. The device kernels evaluate the same
//! terms in the same order, so CPU and GPU agree bit for bit in `f32`.
//!
//! # What the wall terms mean
//!
//! Interior lanes of the central families recover the negated forward
//! stencil (central differences are skew-symmetric away from the walls),
//! but the one-sided wall closures transpose into multi-lane corrections:
//! row 0's forward difference contributes `−u₀/h` to lane 0 and `+u₀/h` to
//! lane 1, and symmetrically at the far wall. The staggered forward sweep is
//! rectangular (`n → n−1`), so its adjoint fans an `n−1`-lane upstream back
//! out to `n` lanes.

use eunomia::{FloatElement, NumericElement, RealField};
use leto::{ArrayView3, ArrayViewMut3, LetoError, Result};

use super::f;
use super::leapfrog::Axis;
use super::FiniteDifference3DScheme;

/// Reciprocal scales for the swept axis, in the provider parenthesization.
struct Scales<T> {
    inv_h: T,
    inv_2h: T,
    inv_12h: T,
    inv_60h: T,
}

fn scales<T: RealField + FloatElement + Copy>(h: T) -> Scales<T> {
    let one = <T as NumericElement>::ONE;
    Scales {
        inv_h: one / h,
        inv_2h: one / (f::<T>(2.0) * h),
        inv_12h: one / (f::<T>(12.0) * h),
        inv_60h: one / (f::<T>(60.0) * h),
    }
}

/// Whether forward row `i` of a central-4 sweep is a second-order row.
///
/// First match wins, exactly the forward selection: the walls are one-sided
/// first, so on a two-point axis no row is second-order.
fn is_central4_second(n: usize, i: usize) -> bool {
    i >= 1 && i + 2 <= n && (i < 2 || i + 2 >= n)
}

/// Whether forward row `i` of a central-4 sweep is a fourth-order row.
fn is_central4_fourth(n: usize, i: usize) -> bool {
    i >= 2 && i + 3 <= n
}

/// Central-2 adjoint lane: wall taps plus the interior's negated stencil.
fn adjoint2_lane<T, F>(u: F, n: usize, j: usize, scales: &Scales<T>) -> T
where
    T: RealField + FloatElement + Copy,
    F: Fn(usize) -> T,
{
    let zero = <T as NumericElement>::ZERO;
    let mut v = zero;
    if j == 0 {
        v += (-u(0)) * scales.inv_h;
    }
    if j == 1 {
        v += u(0) * scales.inv_h;
    }
    if j + 2 == n {
        v += (-u(n - 1)) * scales.inv_h;
    }
    if j + 1 == n {
        v += u(n - 1) * scales.inv_h;
    }
    if j + 3 <= n {
        v += (-u(j + 1)) * scales.inv_2h;
    }
    if j >= 2 {
        v += u(j - 1) * scales.inv_2h;
    }
    v
}

/// Central-4 adjoint lane: wall taps, second-order row taps, fourth-order
/// row taps. A singleton axis is flat, since the forward sweep is zero.
fn adjoint4_lane<T, F>(u: F, n: usize, j: usize, scales: &Scales<T>) -> T
where
    T: RealField + FloatElement + Copy,
    F: Fn(usize) -> T,
{
    let zero = <T as NumericElement>::ZERO;
    if n == 1 {
        return zero;
    }
    let eight = f::<T>(8.0);
    let mut v = zero;
    if j == 0 {
        v += (-u(0)) * scales.inv_h;
    }
    if j == 1 {
        v += u(0) * scales.inv_h;
    }
    if j + 2 == n {
        v += (-u(n - 1)) * scales.inv_h;
    }
    if j + 1 == n {
        v += u(n - 1) * scales.inv_h;
    }
    if j + 1 < n && is_central4_second(n, j + 1) {
        v += (-u(j + 1)) * scales.inv_2h;
    }
    if j >= 1 && is_central4_second(n, j - 1) {
        v += u(j - 1) * scales.inv_2h;
    }
    if j + 2 < n && is_central4_fourth(n, j + 2) {
        v += u(j + 2) * scales.inv_12h;
    }
    if j + 1 < n && is_central4_fourth(n, j + 1) {
        v += ((-eight) * u(j + 1)) * scales.inv_12h;
    }
    if j >= 1 && is_central4_fourth(n, j - 1) {
        v += (eight * u(j - 1)) * scales.inv_12h;
    }
    if j >= 2 && is_central4_fourth(n, j - 2) {
        v += (-u(j - 2)) * scales.inv_12h;
    }
    v
}

/// Central-6 adjoint lane: wall taps, then second-, fourth-, sixth-order row
/// taps in canonical order.
fn adjoint6_lane<T, F>(u: F, n: usize, j: usize, scales: &Scales<T>) -> T
where
    T: RealField + FloatElement + Copy,
    F: Fn(usize) -> T,
{
    let zero = <T as NumericElement>::ZERO;
    let eight = f::<T>(8.0);
    let nine = f::<T>(9.0);
    let forty_five = f::<T>(45.0);
    let is_second = |i: usize| i == 1 || i + 2 == n;
    let is_fourth = |i: usize| i == 2 || i + 3 == n;
    let is_sixth = |i: usize| i >= 3 && i + 4 <= n;
    let mut v = zero;
    if j == 0 {
        v += (-u(0)) * scales.inv_h;
    }
    if j == 1 {
        v += u(0) * scales.inv_h;
    }
    if j + 2 == n {
        v += (-u(n - 1)) * scales.inv_h;
    }
    if j + 1 == n {
        v += u(n - 1) * scales.inv_h;
    }
    if j + 1 < n && is_second(j + 1) {
        v += (-u(j + 1)) * scales.inv_2h;
    }
    if j >= 1 && is_second(j - 1) {
        v += u(j - 1) * scales.inv_2h;
    }
    if j + 2 < n && is_fourth(j + 2) {
        v += u(j + 2) * scales.inv_12h;
    }
    if j + 1 < n && is_fourth(j + 1) {
        v += ((-eight) * u(j + 1)) * scales.inv_12h;
    }
    if j >= 1 && is_fourth(j - 1) {
        v += (eight * u(j - 1)) * scales.inv_12h;
    }
    if j >= 2 && is_fourth(j - 2) {
        v += (-u(j - 2)) * scales.inv_12h;
    }
    if j + 3 < n && is_sixth(j + 3) {
        v += (-u(j + 3)) * scales.inv_60h;
    }
    if j + 2 < n && is_sixth(j + 2) {
        v += (nine * u(j + 2)) * scales.inv_60h;
    }
    if j + 1 < n && is_sixth(j + 1) {
        v += ((-forty_five) * u(j + 1)) * scales.inv_60h;
    }
    if j >= 1 && is_sixth(j - 1) {
        v += (forty_five * u(j - 1)) * scales.inv_60h;
    }
    if j >= 2 && is_sixth(j - 2) {
        v += ((-nine) * u(j - 2)) * scales.inv_60h;
    }
    if j >= 3 && is_sixth(j - 3) {
        v += u(j - 3) * scales.inv_60h;
    }
    v
}

/// Staggered-forward adjoint lane: the rectangular `n → n−1` sweep fans an
/// `n−1`-lane upstream back out to `n` lanes.
fn adjoint_forward_lane<T, F>(u: F, n: usize, j: usize, inv_h: T) -> T
where
    T: RealField + FloatElement + Copy,
    F: Fn(usize) -> T,
{
    let zero = <T as NumericElement>::ZERO;
    let mut v = zero;
    if j + 1 < n {
        v += (-u(j)) * inv_h;
    }
    if j >= 1 {
        v += u(j - 1) * inv_h;
    }
    v
}

/// Staggered-backward adjoint lane: the forward wall row plus the backward
/// rows' transposed taps.
fn adjoint_backward_lane<T, F>(u: F, n: usize, j: usize, inv_h: T) -> T
where
    T: RealField + FloatElement + Copy,
    F: Fn(usize) -> T,
{
    let zero = <T as NumericElement>::ZERO;
    let mut v = zero;
    if j == 0 {
        v += (-u(0)) * inv_h;
    }
    if j == 1 {
        v += u(0) * inv_h;
    }
    if j + 2 <= n {
        v += (-u(j + 1)) * inv_h;
    }
    if j >= 1 {
        v += u(j) * inv_h;
    }
    v
}

fn minimum_extent(scheme: FiniteDifference3DScheme) -> usize {
    match scheme {
        FiniteDifference3DScheme::CentralSecondOrder => 3,
        FiniteDifference3DScheme::CentralFourthOrder => 1,
        FiniteDifference3DScheme::CentralSixthOrder => 7,
        FiniteDifference3DScheme::StaggeredForward
        | FiniteDifference3DScheme::StaggeredBackward => 2,
    }
}

/// Transpose sweep of one scheme along one axis: `upstream` has the forward
/// sweep's output shape, `grad` the forward sweep's input shape.
pub(super) fn adjoint_into<T>(
    scheme: FiniteDifference3DScheme,
    axis: Axis,
    upstream: ArrayView3<T>,
    grad: &mut ArrayViewMut3<'_, T>,
    h: T,
) -> Result<()>
where
    T: RealField + FloatElement + Copy,
{
    let shape = grad.shape();
    let n = shape[axis.index()];
    let required = minimum_extent(scheme);
    if n < required {
        return Err(LetoError::InvalidInput(
            format!(
                "{scheme:?} adjoint: need at least {required} points on the differentiated axis, got {n}"
            ),
        ));
    }
    let mut expected = shape;
    if matches!(scheme, FiniteDifference3DScheme::StaggeredForward) {
        expected[axis.index()] -= 1;
    }
    if upstream.shape() != expected {
        return Err(LetoError::InvalidInput(format!(
            "{scheme:?} adjoint: upstream shape {:?} must be {expected:?} for grad shape {shape:?}",
            upstream.shape(),
        )));
    }
    let s = scales(h);
    let [nx, ny, nz] = shape;
    match axis {
        Axis::X => {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        grad[[i, j, k]] = lane(scheme, n, i, |t| upstream[[t, j, k]], &s);
                    }
                }
            }
        }
        Axis::Y => {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        grad[[i, j, k]] = lane(scheme, n, j, |t| upstream[[i, t, k]], &s);
                    }
                }
            }
        }
        Axis::Z => {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        grad[[i, j, k]] = lane(scheme, n, k, |t| upstream[[i, j, t]], &s);
                    }
                }
            }
        }
    }
    Ok(())
}

/// One adjoint lane, monomorphized over the upstream reader so each axis
/// traversal inlines its strided read.
fn lane<T, F>(scheme: FiniteDifference3DScheme, n: usize, j: usize, read: F, s: &Scales<T>) -> T
where
    T: RealField + FloatElement + Copy,
    F: Fn(usize) -> T,
{
    match scheme {
        FiniteDifference3DScheme::CentralSecondOrder => adjoint2_lane(read, n, j, s),
        FiniteDifference3DScheme::CentralFourthOrder => adjoint4_lane(read, n, j, s),
        FiniteDifference3DScheme::CentralSixthOrder => adjoint6_lane(read, n, j, s),
        FiniteDifference3DScheme::StaggeredForward => adjoint_forward_lane(read, n, j, s.inv_h),
        FiniteDifference3DScheme::StaggeredBackward => adjoint_backward_lane(read, n, j, s.inv_h),
    }
}
