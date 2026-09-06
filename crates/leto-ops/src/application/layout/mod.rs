//! Value-preserving layout movement into caller-owned storage.

mod complex;

pub use complex::{transpose_complex_matrices, transpose_square_inplace, SquareTransposeError};
