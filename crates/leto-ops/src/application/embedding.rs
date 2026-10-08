//! Embedding-table row gather: [`embedding_gather_into`](crate::application::embedding::embedding_gather_into) and [`embedding_gather`](crate::application::embedding::embedding_gather).
//!
//! CPU counterpart of `hephaestus_core::EmbeddingOps` (1:1 parity):
//! `output[row] = table[indices[row]]`, each a full `embedding_dim` row,
//! gathered in index order. Indices are `u32`, exactly as on the device seam.
//!
//! An out-of-range index is a typed [`LetoError::OutOfBounds`](leto::LetoError::OutOfBounds) rejection -
//! never a clamped or wrapped read - matching the seam's `InvalidConfiguration`
//! clause. Rows are gathered in order and the first out-of-range index aborts
//! immediately, so rows before the failing index are already written when the
//! error returns (the host reference behaves the same way).
//! Zero-copy: rows copy directly into the caller-owned output view, holding
//! only scalar registers besides the three buffers.

use crate::application::index::validate_mutable_output;
use crate::domain::scalar::Scalar;
use leto::{Array, ArrayView, ArrayViewMut, Layout, LetoError, Result, VecStorage};

/// Gather `table` rows selected by `indices` into `output`.
///
/// `table` is `[num_embeddings, embedding_dim]`, `indices` has length `n`,
/// and `output` must be `[n, embedding_dim]` exactly. Strided and offset
/// views are served on all three operands; the output must be injective.
///
/// # Errors
///
/// Returns [`LetoError::ShapeMismatch`] when the output shape differs from
/// `[indices.len(), embedding_dim]`, [`LetoError::OutOfBounds`] naming the
/// first out-of-range index, and the layout errors for invalid storage
/// lengths or an aliased output.
pub fn embedding_gather_into<T: Scalar>(
    table: &ArrayView<'_, T, 2>,
    indices: &ArrayView<'_, u32, 1>,
    output: &mut ArrayViewMut<'_, T, 2>,
) -> Result<()> {
    let [num_embeddings, embedding_dim] = table.shape();
    let [rows] = indices.shape();
    if output.shape() != [rows, embedding_dim] {
        return Err(LetoError::ShapeMismatch {
            lhs: [rows, embedding_dim].to_vec(),
            rhs: output.shape().to_vec(),
        });
    }
    table.layout().validate_storage_len(table.data().len())?;
    indices
        .layout()
        .validate_storage_len(indices.data().len())?;
    validate_mutable_output(output, "embedding gather output")?;

    if let (Some(table_slice), Some(index_slice), Some(out_slice)) =
        (table.as_slice(), indices.as_slice(), output.as_mut_slice())
    {
        for (row, &index) in index_slice.iter().enumerate() {
            let selected = usize::try_from(index).map_err(|_| LetoError::Overflow {
                reason: "embedding index exceeds the addressable row count",
            })?;
            if selected >= num_embeddings {
                return Err(LetoError::OutOfBounds {
                    index: vec![selected],
                    shape: vec![num_embeddings],
                });
            }
            let table_base = selected * embedding_dim;
            let out_base = row * embedding_dim;
            out_slice[out_base..out_base + embedding_dim]
                .copy_from_slice(&table_slice[table_base..table_base + embedding_dim]);
        }
        return Ok(());
    }

    let table_layout = table.layout();
    let index_layout = indices.layout();
    let output_layout = output.layout();
    let table_data = table.data();
    let index_data = indices.data();
    let output_data = output.data_mut();
    let table_base = table_layout.offset() as isize;
    let index_base = index_layout.offset() as isize;
    let output_base = output_layout.offset() as isize;
    let (tab_row, tab_col) = (table_layout.strides()[0], table_layout.strides()[1]);
    let index_stride = index_layout.strides()[0];
    let (out_row, out_col) = (output_layout.strides()[0], output_layout.strides()[1]);
    for row in 0..rows {
        let at = (index_base + row as isize * index_stride) as usize;
        let selected = usize::try_from(index_data[at]).map_err(|_| LetoError::Overflow {
            reason: "embedding index exceeds the addressable row count",
        })?;
        if selected >= num_embeddings {
            return Err(LetoError::OutOfBounds {
                index: vec![selected],
                shape: vec![num_embeddings],
            });
        }
        for col in 0..embedding_dim {
            let from = (table_base + selected as isize * tab_row + col as isize * tab_col) as usize;
            let to = (output_base + row as isize * out_row + col as isize * out_col) as usize;
            output_data[to] = table_data[from];
        }
    }
    Ok(())
}

/// Gather `table` rows selected by `indices` into a newly allocated
/// C-contiguous `[indices.len(), embedding_dim]` output.
///
/// See [`embedding_gather_into`] for the contract and errors.
pub fn embedding_gather<T: Scalar>(
    table: &ArrayView<'_, T, 2>,
    indices: &ArrayView<'_, u32, 1>,
) -> Result<Array<T, VecStorage<T>, 2>> {
    let [_, embedding_dim] = table.shape();
    let [rows] = indices.shape();
    let layout = Layout::c_contiguous([rows, embedding_dim])?;
    let size = layout.checked_size()?;
    let mut output = Array::new(layout, VecStorage::uninit(size))?;
    embedding_gather_into(table, indices, &mut output.view_mut())?;
    Ok(output)
}
