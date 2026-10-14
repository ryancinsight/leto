//! Rank-2 triangular masking (tril / triu): [`triangular_into`](crate::application::triangular::triangular_into) and [`triangular`].
//!
//! CPU counterpart of `hephaestus_core::TriangularOps` (1:1 parity):
//! `output[row, col]` copies the input when the element is on the kept side
//! of the diagonal offset by `diagonal`, else [`NumericElement::ZERO`](eunomia::NumericElement::ZERO).
//! [`TriangularMode::Lower`](crate::application::triangular::TriangularMode::Lower) keeps `col <= row + diagonal` (numpy's `tril`:
//! `diagonal = 0` is the main diagonal, positive shifts it up-right,
//! negative down-left); [`TriangularMode::Upper`](crate::application::triangular::TriangularMode::Upper) keeps
//! `col >= row + diagonal` (numpy's `triu`, the mirror condition).
//!
//! [`triangular_keeps`](crate::application::triangular::triangular_keeps) ports the device seam's predicate verbatim — the
//! same overflow-safe `u128` comparison, including the `usize`/`i64`
//! extremes — so host and device agree cell for cell on every input.
//! Zero-copy: masking writes directly into the caller-owned output view,
//! holding only scalar registers besides the two buffers.

use crate::application::index::validate_mutable_output;
use crate::application::output::OutputSlot;
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
/// diagonals compare mathematically, without clamping. `const`: the widening
/// casts are lossless (`usize`/`u64` magnitudes always fit `u128`), and the
/// sign-extending `diagonal as u128` only evaluates under a `diagonal >= 0`
/// guard, where it is exact.
#[inline]
#[must_use]
pub const fn triangular_keeps(mode: TriangularMode, row: usize, col: usize, diagonal: i64) -> bool {
    // Both operands widen to `u128` for the mixed-width compare: the
    // predicate evaluates in const contexts (the compile-time-asserted
    // contract in the test), and `usize -> u128` is a lossless widening
    // for which std provides no `From`/`TryFrom` on 64-bit targets - the
    // conversion-module class the stack's cast rule names: all three are
    // lossless 64 -> 128-bit widenings, and `From`'s associated function
    // is not yet stable as const. The eunomia conversion module owning
    // these widenings is a separate item.
    let difference = if col >= row {
        (true, (col - row) as u128)
    } else {
        (false, (row - col) as u128)
    };
    let diagonal_magnitude = diagonal.unsigned_abs() as u128;
    match mode {
        TriangularMode::Lower => match difference {
            (true, magnitude) => diagonal >= 0 && magnitude <= diagonal_magnitude,
            (false, magnitude) => diagonal >= 0 || magnitude >= diagonal_magnitude,
        },
        TriangularMode::Upper => match difference {
            (true, magnitude) => diagonal < 0 || magnitude >= diagonal_magnitude,
            (false, magnitude) => diagonal < 0 && magnitude <= diagonal_magnitude,
        },
    }
}

/// Row-granular masking for C-contiguous operands: each row copies its kept
/// run and fills the masked remainder, replacing the per-element branch with
/// vectorized slice operations. The split point derives from the same
/// `col <= row + diagonal` / `col >= row + diagonal` boundary
/// [`triangular_keeps`](crate::application::triangular::triangular_keeps) tests, computed in `i128` so `usize` rows and any
/// `i64` diagonal combine without overflow or clamping.
fn triangular_contiguous<T: Scalar, S: OutputSlot<T>>(
    input: &[T],
    output: &mut [S],
    rows: usize,
    cols: usize,
    mode: TriangularMode,
    diagonal: i64,
) {
    for row in 0..rows {
        let base = row * cols;
        let bound = i128::try_from(row).expect("invariant: row fits i128") + i128::from(diagonal);
        let split = match mode {
            // Lower keeps `col <= row + diagonal`: the kept prefix length.
            TriangularMode::Lower => usize::try_from(
                (bound + 1).clamp(0, i128::try_from(cols).expect("invariant: cols fits i128")),
            )
            .expect("invariant: prefix length fits usize"),
            // Upper keeps `col >= row + diagonal`: the masked prefix length.
            TriangularMode::Upper => usize::try_from(
                bound.clamp(0, i128::try_from(cols).expect("invariant: cols fits i128")),
            )
            .expect("invariant: prefix length fits usize"),
        };
        let (out_head, out_tail) = output[base..base + cols].split_at_mut(split);
        let (in_head, in_tail) = input[base..base + cols].split_at(split);
        match mode {
            TriangularMode::Lower => {
                S::copy_from_slice(out_head, in_head);
                S::fill(out_tail, T::ZERO);
            }
            TriangularMode::Upper => {
                S::fill(out_head, T::ZERO);
                S::copy_from_slice(out_tail, in_tail);
            }
        }
    }
}

/// Mask `input` by `mode` and `diagonal` into `output`, whose shape must
/// equal `input`'s exactly.
///
/// Strided and offset views are served on both operands; the output must be
/// injective. C-contiguous operands take a row-granular copy/fill fast path
/// instead of the per-element branch.
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
    triangular_slots::<T, T>(input, mode, diagonal, output)
}

fn triangular_slots<T: Scalar, S: OutputSlot<T>>(
    input: &ArrayView<'_, T, 2>,
    mode: TriangularMode,
    diagonal: i64,
    output: &mut ArrayViewMut<'_, S, 2>,
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
    if let (Some(input_slice), Some(output_slice)) = (input.as_slice(), output.as_mut_slice()) {
        triangular_contiguous(input_slice, output_slice, rows, cols, mode, diagonal);
        return Ok(());
    }

    let input_layout = input.layout();
    let output_layout = output.layout();
    let input_data = input.data();
    let output_data = output.data_mut();
    let input_base = isize::try_from(input_layout.offset()).expect("invariant: offset fits isize");
    let output_base =
        isize::try_from(output_layout.offset()).expect("invariant: offset fits isize");
    let (in_row, in_col) = (input_layout.strides()[0], input_layout.strides()[1]);
    let (out_row, out_col) = (output_layout.strides()[0], output_layout.strides()[1]);
    for row in 0..rows {
        for col in 0..cols {
            let from = usize::try_from(
                input_base
                    + isize::try_from(row).expect("invariant: row fits isize") * in_row
                    + isize::try_from(col).expect("invariant: col fits isize") * in_col,
            )
            .expect("invariant: index fits usize");
            let to = usize::try_from(
                output_base
                    + isize::try_from(row).expect("invariant: row fits isize") * out_row
                    + isize::try_from(col).expect("invariant: col fits isize") * out_col,
            )
            .expect("invariant: index fits usize");
            output_data[to].write(if triangular_keeps(mode, row, col, diagonal) {
                input_data[from]
            } else {
                T::ZERO
            });
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
    triangular_slots::<T, _>(input, mode, diagonal, &mut output.view_mut())?;
    // SAFETY: both mask branches write every dense output element. T is Copy.
    Array::new(layout, unsafe { output.into_storage().assume_init() })
}
