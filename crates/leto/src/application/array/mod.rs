//! The N-dimensional strided [`Array`] type and its operation families.
//!
//! `types` holds the struct definition and its serde impls, `construction`
//! the validating constructor, `assign_source` the `AssignSource` trait
//! bridging arrays and views as assignment inputs, `accessors` the
//! `Storage`-bound read-only operations, `mutation` the `StorageMut`-bound
//! in-place operations, and `indexing`/`equality` the standard trait impls.

mod accessors;
mod assign_source;
mod construction;
mod equality;
mod indexing;
mod mutation;
mod types;

pub use assign_source::AssignSource;
pub use types::Array;

pub(crate) use mutation::linear_to_index;
