//! Zip-style traversal, fold, and mutation over one or more array views.
//!
//! Operation families live in their own leaf modules: `fold` and
//! `coordinate` cover read-only reduction and sparse-coordinate mutation,
//! `map` covers dense indexed in-place mapping, `sources` and `outputs`
//! carry the statically dispatched multi-arity zip machinery behind
//! [`zip_mut_with`] and [`indexed_zip_mut_with`], and `traversal` holds the
//! shared row-major traversal helper every family drives.

mod coordinate;
mod fold;
mod map;
mod outputs;
mod sources;
mod traversal;

pub use coordinate::{
    coordinate_map_inplace, coordinate_map_plan, coordinate_map_plan_inplace, CoordinateMapPlan,
};
pub use fold::{indexed_fold, indexed_fold_fortran, zip_fold};
pub use map::{indexed_map4_inplace, indexed_map_inplace};
pub use outputs::{indexed_zip_mut_with, zip_mut_with, IndexedZipMutOutputs, ZipMutOutputs};
pub use sources::ZipSources;
