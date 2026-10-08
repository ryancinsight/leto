//! Axis-aware keep-dim reduction operations, plus whole-array `min`/`max`.
//!
//! `strategies` defines the [`AxisReduction`]
//! marker-type contract (`Sum`/`Product`/`Mean`/`Min`/`Max`); `whole_array`
//! reduces every element to one scalar; `axis` applies a keep-dim
//! reduction along one axis, dispatching a 2D row-major fast path and a
//! parallel path before falling back to a strided walk; `convenience`
//! names each strategy's `_axis`/`_axis_into` entry points plus the
//! whole-array `min`/`max` wrappers; `topk` selects the `k` largest elements
//! per lane, CPU counterpart of `hephaestus_core::TopKOps`.

mod axis;
mod convenience;
mod strategies;
mod topk;
mod whole_array;

pub use axis::{reduce_axis, reduce_axis_into};
pub use convenience::{
    max, max_axis, max_axis_into, mean_axis, mean_axis_into, min, min_axis, min_axis_into,
    product_axis, product_axis_into, sum_axis, sum_axis_into,
};
pub use strategies::{AxisReduction, MaxAxis, MeanAxis, MinAxis, ProductAxis, SumAxis};
pub use topk::{topk_axis, topk_axis_into, Topk};
pub use whole_array::reduce_all;
