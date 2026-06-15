/// Rank-specific type aliases.
pub mod aliases;
/// Elementwise arithmetic operators (`Add`/`Sub`/`Mul`/`Div`/`Neg`).
pub mod arithmetic;
/// Owned array type.
pub mod array;
/// Named row and column view helpers.
mod axis;
mod constructors;
mod index;
/// Subview iteration.
pub mod iter;
/// Element-wise reduction operations (sum, mean, min, max, argmin, argmax).
pub mod reduction;
/// Structural array operations (concat, pad, split).
pub mod structure;
/// Borrowed array view types.
pub mod view;

pub use aliases::{
    Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, ArrayViewMut2,
    ArrayViewMut3,
};
pub use arithmetic::ScalarOperand;
pub use array::Array;
pub use iter::{AxisIter, AxisIterMut};
pub use reduction::{
    argmax, argmin, max_all, max_axis, mean_all, mean_axis, min_all, min_axis, std_all, std_axis,
    sum_all, sum_axis, var_all, var_axis,
};
pub use structure::{concat, pad, split, stack, PadWidth};
pub use view::{ArrayView, ArrayViewMut};
