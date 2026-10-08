//! Rank-2 triangular masking (tril / triu): [`triangular_into`] and [`triangular`].
//!
//! CPU counterpart of `hephaestus_core::TriangularOps` (1:1 parity):
//! `output[row, col]` copies the input when the element is on the kept side
//! of the diagonal offset by `diagonal`, else [`Scalar::ZERO`].
//! [`TriangularMode::Lower`] keeps `col <= row + diagonal` (numpy's `tril`:
//! `diagonal = 0` is the main diagonal, positive shifts it up-right,
//! negative down-left); [`TriangularMode::Upper`] keeps
//! `col >= row + diagonal` (numpy's `triu`, the mirror condition).
//!
//! [`triangular_keeps`] ports the device seam's predicate verbatim — the
//! same overflow-safe `u128` comparison, including the `usize`/`i64`
//! extremes — so host and device agree cell for cell on every input.
//! Zero-copy: masking writes directly into the caller-owned output view,
//! holding only scalar registers besides the two buffers.

use crate::application::index::validate_mutable_output;
use crate::domain::scalar::Scalar;
use leto::{Array, ArrayView, ArrayViewMut, Layout, LetoError, Result, VecStorage};

/// Which side of the diagonal survives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriangularMode {
    /// Keep `col <= row + diagonal` (numpy's `tril`).
    Lower,
    /// Keep `col >= row + diagonal` (numpy's `triu`).
    Upper,
}

/// Whether element `(row, col)` survives `mode`'s mask at the given
/// `diagonal` offset.
///
/// Verbatim port of the `TriangularOps` seam predicate: the difference is
/// measured in `u128` so `usize::MAX` coordinates and `i64::MIN`/`MAX`
/// diagonals compare mathematically, without clamping.
#[must_use]
pub fn triangular_keeps(mode: TriangularMode, row: usize, col: usize, diagonal: i64) -> bool {
    let difference = if col >= row {
        (
            true,
            u128::try_from(col - row).expect("invariant: usize fits u128"),
        )
    } else {
        (
            false,
            u128::try_from(row - col).expect("invariant: usize fits u128"),
        )
    };
    let diagonal_magnitude = u128::from(diagonal.unsigned_abs());
    match mode {
        TriangularMode::Lower => match difference {
            (true, magnitude) => {
                diagonal >= 0
                    && magnitude
                        <= u128::try_from(diagonal).expect("invariant: nonnegative diagonal")
            }
            (false, magnitude) => diagonal >= 0 || magnitude >= diagonal_magnitude,
        },
        TriangularMode::Upper => match difference {
            (true, magnitude) => {
                diagonal < 0
                    || magnitude
                        >= u128::try_from(diagonal).expect("invariant: nonnegative diagonal")
            }
            (false, magnitude) => diagonal < 0 && magnitude <= diagonal_magnitude,
        },
    }
}

/// Mask `input` by `mode` and `diagonal` into `output`, whose shape must
/// equal `input`'s exactly.
///
/// Strided and offset views are served on both operands; the output must be
/// injective.
///
/// # Errors
///
/// Returns [`LetoError::ShapeMismatch`] when the output shape differs from
/// the input shape, and the layout errors for invalid storage lengths or an
/// aliased output.
pub fn triangular_into<T: Scalar>(
    input: &ArrayView<'_, T, 2>,
    mode: TriangularMode,
    diagonal: i64,
    output: &mut ArrayViewMut<'_, T, 2>,
) -> Result<()> {
    if output.shape() != input.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: input.shape().to_vec(),
            rhs: output.shape().to_vec(),
        });
    }
    input.layout().validate_storage_len(input.data().len())?;
    validate_mutable_output(output, "triangular output")?;

    let [rows, cols] = input.shape();
    let input_layout = input.layout();
    let output_layout = output.layout();
    let input_data = input.data();
    let output_data = output.data_mut();
    let input_base = input_layout.offset() as isize;
    let output_base = output_layout.offset() as isize;
    let (in_row, in_col) = (input_layout.strides()[0], input_layout.strides()[1]);
    let (out_row, out_col) = (output_layout.strides()[0], output_layout.strides()[1]);
    for row in 0..rows {
        for col in 0..cols {
            let from = (input_base + row as isize * in_row + col as isize * in_col) as usize;
            let to = (output_base + row as isize * out_row + col as isize * out_col) as usize;
            output_data[to] = if triangular_keeps(mode, row, col, diagonal) {
                input_data[from]
            } else {
                T::ZERO
            };
        }
    }
    Ok(())
}

/// Mask `input` by `mode` and `diagonal` into a newly allocated C-contiguous
/// output of the input's shape.
///
/// See [`triangular_into`] for the contract and errors.
pub fn triangular<T: Scalar>(
    input: &ArrayView<'_, T, 2>,
    mode: TriangularMode,
    diagonal: i64,
) -> Result<Array<T, VecStorage<T>, 2>> {
    let layout = Layout::c_contiguous(input.shape())?;
    let size = layout.checked_size()?;
    let mut output = Array::new(layout, VecStorage::uninit(size))?;
    triangular_into(input, mode, diagonal, &mut output.view_mut())?;
    Ok(output)
}
