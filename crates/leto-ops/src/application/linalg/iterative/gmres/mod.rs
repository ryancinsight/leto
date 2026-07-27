//! GMRES sub-modules.
mod arnoldi;
mod givens;
mod solver;

pub use solver::GMRES;

use leto::{Array1, Array2};

/// Flat mutable view of a workspace vector.
///
/// Every buffer reached through these helpers is allocated by [`GMRES`] itself
/// via `Array1::zeros`/`Array2::zeros`, so C-contiguity is a construction
/// invariant rather than a caller-supplied property.
#[inline]
fn flat_mut<'a, T>(name: &str, array: &'a mut Array1<T>) -> &'a mut [T] {
    array.as_slice_mut().unwrap_or_else(|| {
        panic!("invariant: {name} is an owned contiguous GMRES workspace vector")
    })
}

/// Flat immutable view of a workspace vector.
#[inline]
fn flat<'a, T>(name: &str, array: &'a Array1<T>) -> &'a [T] {
    array
        .as_slice()
        .unwrap_or_else(|| panic!("invariant: {name} is an owned contiguous GMRES workspace vector"))
}

/// Flat immutable view of a workspace matrix, in row-major order.
#[inline]
fn flat2<'a, T>(name: &str, array: &'a Array2<T>) -> &'a [T] {
    array
        .as_slice()
        .unwrap_or_else(|| panic!("invariant: {name} is an owned contiguous GMRES workspace matrix"))
}

/// Flat mutable view of a workspace matrix, in row-major order.
#[inline]
fn flat2_mut<'a, T>(name: &str, array: &'a mut Array2<T>) -> &'a mut [T] {
    array.as_slice_mut().unwrap_or_else(|| {
        panic!("invariant: {name} is an owned contiguous GMRES workspace matrix")
    })
}
