//! Named component access (`.x`/`.y`/`.z`/`.w`) for low-dimension vectors and
//! points, for nalgebra-style ergonomics.
//!
//! Each low-dim [`Vector`]/[`Point`] `Deref`s to a `#[repr(C)]` named-field view
//! with identical memory layout, so `v.x` reads/writes the first component with
//! zero cost. The casts are sound because every type here is `#[repr(C)]` and
//! `[T; N]` has the same layout as a struct of `N` `T` fields in declaration
//! order.

use super::{Point, Vector};
use core::ops::{Deref, DerefMut};

/// Named-field view of a 1-component value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct X<T> {
    /// First component.
    pub x: T,
}

/// Named-field view of a 2-component value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct XY<T> {
    /// First component.
    pub x: T,
    /// Second component.
    pub y: T,
}

/// Named-field view of a 3-component value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct XYZ<T> {
    /// First component.
    pub x: T,
    /// Second component.
    pub y: T,
    /// Third component.
    pub z: T,
}

/// Named-field view of a 4-component value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct XYZW<T> {
    /// First component.
    pub x: T,
    /// Second component.
    pub y: T,
    /// Third component.
    pub z: T,
    /// Fourth component.
    pub w: T,
}

macro_rules! impl_named_access {
    ($outer:ty, $view:ident, $n:literal) => {
        impl<T> Deref for $outer {
            type Target = $view<T>;
            #[inline(always)]
            fn deref(&self) -> &$view<T> {
                // SAFETY: `Self` is `#[repr(C)]` over `[T; $n]` and `$view<T>` is
                // `#[repr(C)]` over `$n` `T` fields — identical layout.
                unsafe { &*(self as *const Self as *const $view<T>) }
            }
        }
        impl<T> DerefMut for $outer {
            #[inline(always)]
            fn deref_mut(&mut self) -> &mut $view<T> {
                // SAFETY: see `deref`.
                unsafe { &mut *(self as *mut Self as *mut $view<T>) }
            }
        }
    };
}

impl_named_access!(Vector<T, 1>, X, 1);
impl_named_access!(Vector<T, 2>, XY, 2);
impl_named_access!(Vector<T, 3>, XYZ, 3);
impl_named_access!(Vector<T, 4>, XYZW, 4);
impl_named_access!(Point<T, 1>, X, 1);
impl_named_access!(Point<T, 2>, XY, 2);
impl_named_access!(Point<T, 3>, XYZ, 3);

#[cfg(test)]
mod tests {
    use crate::geometry::{Point3, Vector3};

    #[test]
    fn named_access_reads_and_writes() {
        let mut v = Vector3::new(1.0_f64, 2.0, 3.0);
        assert_eq!((v.x, v.y, v.z), (1.0, 2.0, 3.0));
        v.y = 9.0;
        assert_eq!(v.data, [1.0, 9.0, 3.0]);
        let p = Point3::new(4.0_f64, 5.0, 6.0);
        assert_eq!((p.x, p.y, p.z), (4.0, 5.0, 6.0));
        // layout parity guard
        assert_eq!(
            core::mem::size_of::<Vector3<f64>>(),
            core::mem::size_of::<super::XYZ<f64>>()
        );
    }
}
