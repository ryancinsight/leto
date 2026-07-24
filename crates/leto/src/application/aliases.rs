use crate::application::array::Array;
use crate::application::view::{ArrayView, ArrayViewMut};
use crate::infrastructure::storage::VecStorage;

/// A 1-dimensional owned array.
pub type Array1<T> = Array<T, VecStorage<T>, 1>;

/// A 2-dimensional owned array.
pub type Array2<T> = Array<T, VecStorage<T>, 2>;

/// A 3-dimensional owned array.
pub type Array3<T> = Array<T, VecStorage<T>, 3>;

/// A 4-dimensional owned array.
pub type Array4<T> = Array<T, VecStorage<T>, 4>;

/// A 1-dimensional borrowed array view.
pub type ArrayView1<'a, T> = ArrayView<'a, T, 1>;

/// A 2-dimensional borrowed array view.
pub type ArrayView2<'a, T> = ArrayView<'a, T, 2>;

/// A 3-dimensional borrowed array view.
pub type ArrayView3<'a, T> = ArrayView<'a, T, 3>;

/// A 4-dimensional borrowed array view.
pub type ArrayView4<'a, T> = ArrayView<'a, T, 4>;

/// A 1-dimensional mutable borrowed array view.
pub type ArrayViewMut1<'a, T> = ArrayViewMut<'a, T, 1>;

/// A 2-dimensional mutable borrowed array view.
pub type ArrayViewMut2<'a, T> = ArrayViewMut<'a, T, 2>;

/// A 3-dimensional mutable borrowed array view.
pub type ArrayViewMut3<'a, T> = ArrayViewMut<'a, T, 3>;

/// A 4-dimensional mutable borrowed array view.
pub type ArrayViewMut4<'a, T> = ArrayViewMut<'a, T, 4>;
