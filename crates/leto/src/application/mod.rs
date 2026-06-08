/// Owned array type.
pub mod array;
/// Borrowed array view types.
pub mod view;

pub use array::Array;
pub use view::{ArrayView, ArrayViewMut};
