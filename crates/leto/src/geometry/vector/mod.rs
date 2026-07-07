//! Fixed-dimension geometric [`Vector`]: the type, constructors, and indexing.
//!
//! Operator overloads live in `ops`, Euclidean geometry in `euclidean`, and
//! the const-generic serde impls in `serde` — one concern per leaf module.

mod euclidean;
mod ops;
#[cfg(feature = "serde")]
mod serde;

use core::ops::{Index, IndexMut};

/// A fixed-dimension column vector in Euclidean space, generic over the scalar
/// field `T` and the dimension `N` (const generic — one definition per
/// dimension, monomorphized). [`Vector2`]/[`Vector3`] are dimension aliases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Vector<T, const N: usize> {
    /// Components.
    pub data: [T; N],
}

/// A 2-dimensional vector.
pub type Vector2<T> = Vector<T, 2>;
/// A 3-dimensional vector.
pub type Vector3<T> = Vector<T, 3>;

impl<T, const N: usize> Vector<T, N> {
    /// Construct from a component array.
    #[inline(always)]
    pub const fn from_array(data: [T; N]) -> Self {
        Self { data }
    }
}

impl<T: Copy, const N: usize> Vector<T, N> {
    /// A vector with every component equal to `value`.
    #[inline(always)]
    pub const fn splat(value: T) -> Self {
        Self { data: [value; N] }
    }
}

impl<T> Vector<T, 2> {
    /// Construct from `x`, `y` components.
    #[inline(always)]
    pub const fn new(x: T, y: T) -> Self {
        Self { data: [x, y] }
    }
}

impl<T> Vector<T, 3> {
    /// Construct from `x`, `y`, `z` components.
    #[inline(always)]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { data: [x, y, z] }
    }
}

impl<T> Vector<T, 4> {
    /// Construct from `x`, `y`, `z`, `w` components.
    #[inline(always)]
    pub const fn new(x: T, y: T, z: T, w: T) -> Self {
        Self { data: [x, y, z, w] }
    }
}

impl<T: Default + Copy, const N: usize> Default for Vector<T, N> {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(T::default())
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    #[inline(always)]
    fn from(data: [T; N]) -> Self {
        Self { data }
    }
}

impl<T, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}
