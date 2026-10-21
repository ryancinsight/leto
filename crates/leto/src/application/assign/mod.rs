//! Element assignment into arrays and mutable views.
//!
//! Vertical structure: the `assign_into` leaf owns the validated
//! layout-dispatching assignment kernels; the `transpose` leaf owns the
//! cache-blocked transpose copies the mixed-order fast paths use; the
//! `methods` leaf owns the [`Array::try_assign`](crate::application::array::Array::try_assign)
//! and [`ArrayViewMut::try_assign`](crate::application::view::ArrayViewMut::try_assign)
//! entry points.

mod assign_into;
mod methods;
mod transpose;

pub use transpose::{transpose_copy, transpose_copy_strided};
