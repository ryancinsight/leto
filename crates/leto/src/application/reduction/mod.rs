//! Element-wise reduction operations over N-dimensional strided arrays.
//!
//! # Submodules
//! - [`sum`](crate::application::reduction::sum) — [`sum_all`], [`sum_axis`]
//! - [`mean`](crate::application::reduction::mean) — [`mean_all`], [`mean_axis`]
//! - [`min_max`](crate::application::reduction::min_max) — [`min_all`], [`max_all`], [`min_axis`], [`max_axis`], [`argmin`], [`argmax`], [`argmin_all`], [`argmax_all`]
//! - [`variance`](crate::application::reduction::variance) — [`var_all`], [`std_all`], [`var_axis`], [`std_axis`]
//! - [`quantile`](crate::application::reduction::quantile) — [`quantile_all`], [`median_all`], [`quantile_axis`], [`median_axis`]
//!
//! # Shared infrastructure
//! `iter_elements` is a `pub(crate)` helper that yields `&T` references for
//! every logical element of an [`ArrayView`](crate::application::view::ArrayView),
//! respecting arbitrary strides.
//! All leaf modules import it via `crate::application::reduction::iter_elements`.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

mod elements;
pub mod mean;
pub mod min_max;
pub mod quantile;
pub mod sum;
pub mod variance;

pub use mean::{mean_all, mean_axis};
pub use min_max::{argmax, argmax_all, argmin, argmin_all, max_all, max_axis, min_all, min_axis};
pub use quantile::{median_all, median_axis, quantile_all, quantile_axis, Interpolation};
pub use sum::{sum_all, sum_axis};
pub use variance::{std_all, std_axis, var_all, var_axis};

pub(crate) use elements::iter_elements;

#[cfg(test)]
mod tests;
