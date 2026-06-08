/// Rank-specific type aliases.
pub mod aliases;
/// Owned array type.
pub mod array;
mod constructors;
mod index;
/// Subview iteration.
pub mod iter;
/// Borrowed array view types.
pub mod view;

pub use aliases::{
    Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, ArrayViewMut2,
    ArrayViewMut3,
};
pub use array::Array;
pub use iter::{AxisIter, AxisIterMut};
pub use view::{ArrayView, ArrayViewMut};
