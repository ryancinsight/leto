//! N-dimensional strided layout.
//!
//! [`Layout`] is the shape/stride/offset description of a strided view. Its
//! behaviour is split across leaf modules by concern, so the manifest stays a
//! manifest:
//!
//! | Module | Concern |
//! | --- | --- |
//! | `core` | The [`Layout`] type, its construction, and its accessors. |
//! | `shape` | Shape product, size, and shape arithmetic. |
//! | `strides` | Stride arithmetic and contiguous-layout construction. |
//! | `contiguity` | Contiguity predicates. |
//! | `broadcast` | Broadcasting a layout to a wider shape. |
//! | `slice_with` | ndarray-style slicing over a layout. |
//! | `serde` | `Serialize`/`Deserialize` for the layout. |
//! | `kernels` | Shared layout kernels used by the view types. |

mod broadcast;
mod contiguity;
mod core;
pub(crate) mod kernels;
mod serde;
mod shape;
mod slice_with;
mod strides;

pub use core::Layout;
