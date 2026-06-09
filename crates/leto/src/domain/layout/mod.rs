mod broadcast;
mod contiguity;
mod shape;
mod slice_with;
mod strides;

/// Represents an N-dimensional strided layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout<const N: usize> {
    /// The shape of the array (size of each dimension).
    pub shape: [usize; N],
    /// The stride of each dimension in elements (not bytes).
    pub strides: [isize; N],
    /// The starting offset in the storage buffer.
    pub offset: usize,
}

impl<const N: usize> Layout<N> {
    /// Create a new layout with explicit shape, strides, and offset.
    pub const fn new(shape: [usize; N], strides: [isize; N], offset: usize) -> Self {
        Self {
            shape,
            strides,
            offset,
        }
    }
}
