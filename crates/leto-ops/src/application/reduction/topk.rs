//! Rank-2 top-`k` selection along one axis: [`topk_axis_into`] and [`topk_axis`].
//!
//! CPU counterpart of `hephaestus_core::TopKOps` (1:1 parity): the `k` largest
//! elements of each lane along `axis` — both value and source index, sorted
//! descending — with the reduced axis resized to `k` rather than collapsed.
//!
//! The selection mirrors the WGSL kernel's comparison structure exactly, so
//! the two are the same program in two languages: each input element in lane
//! order either fills the next slot (while `a < k`) or displaces the current
//! worst (while strictly greater than it), then bubbles up until the previous
//! slot dominates. Ties break toward the lowest index — an equal value never
//! displaces — matching the argmax/argmin first-strict-improvement rule.
//! Unordered values (NaN) fall out of the same comparisons rather than any
//! special case: a NaN candidate never strictly exceeds the worst slot, and
//! the bubble stops only under `>=`, exactly as on the device.
//!
//! Zero-copy: selection writes directly into the caller-owned output views,
//! holding only scalar registers besides the three buffers.

use crate::application::index::validate_mutable_output;
use crate::domain::scalar::Scalar;
use leto::{Array, ArrayView, ArrayViewMut, Layout, LetoError, Result, VecStorage};

/// Write the `k` largest elements of each lane of `input` along `axis` into
/// `values` (descending) and their source indices into `indices`.
///
/// Both outputs share the input shape with `axis` resized to `k`. Strided and
/// offset views are served on all three operands; outputs must be injective.
/// Indices are `u32`, matching the device seam.
///
/// # Errors
///
/// Returns [`LetoError::StorageError`] when `axis >= 2`, `k == 0`, `k` exceeds
/// the reduced axis's length, or the reduced axis is longer than `u32::MAX`;
/// [`LetoError::ShapeMismatch`] when either output shape differs from the
/// input shape with `axis` resized to `k`; and the layout errors for invalid
/// storage lengths or aliased outputs.
pub fn topk_axis_into<T: Scalar>(
    input: &ArrayView<'_, T, 2>,
    axis: usize,
    k: usize,
    values: &mut ArrayViewMut<'_, T, 2>,
    indices: &mut ArrayViewMut<'_, u32, 2>,
) -> Result<()> {
    if axis >= 2 {
        return Err(LetoError::StorageError {
            reason: format!("axis {axis} out of bounds for rank 2"),
        });
    }
    if k == 0 {
        return Err(LetoError::StorageError {
            reason: "top-k k must be at least 1".to_string(),
        });
    }
    let shape = input.shape();
    let axis_len = shape[axis];
    if k > axis_len {
        return Err(LetoError::StorageError {
            reason: format!("top-k k={k} exceeds the reduced axis length {axis_len}"),
        });
    }
    if axis_len > u32::MAX as usize {
        return Err(LetoError::StorageError {
            reason: format!("top-k reduced axis length {axis_len} does not fit u32"),
        });
    }
    let mut expected = shape;
    expected[axis] = k;
    if values.shape() != expected {
        return Err(LetoError::ShapeMismatch {
            lhs: expected.to_vec(),
            rhs: values.shape().to_vec(),
        });
    }
    if indices.shape() != expected {
        return Err(LetoError::ShapeMismatch {
            lhs: expected.to_vec(),
            rhs: indices.shape().to_vec(),
        });
    }

    input.layout().validate_storage_len(input.data().len())?;
    validate_mutable_output(values, "top-k values")?;
    validate_mutable_output(indices, "top-k indices")?;

    let other = 1 - axis;
    let lanes = shape[other];
    let input_layout = input.layout();
    let values_layout = values.layout();
    let indices_layout = indices.layout();
    let input_data = input.data();
    let values_data = values.data_mut();
    let indices_data = indices.data_mut();
    let input_base = input_layout.offset();
    let values_base = values_layout.offset();
    let indices_base = indices_layout.offset();
    let input_lane_stride = input_layout.strides()[other];
    let input_axis_stride = input_layout.strides()[axis];
    let values_lane_stride = values_layout.strides()[other];
    let values_axis_stride = values_layout.strides()[axis];
    let indices_lane_stride = indices_layout.strides()[other];
    let indices_axis_stride = indices_layout.strides()[axis];

    for lane in 0..lanes {
        let lane_in = input_base as isize + lane as isize * input_lane_stride;
        let mut slots = LaneSlots {
            values: &mut *values_data,
            indices: &mut *indices_data,
            values_base: values_base as isize + lane as isize * values_lane_stride,
            indices_base: indices_base as isize + lane as isize * indices_lane_stride,
            values_stride: values_axis_stride,
            indices_stride: indices_axis_stride,
        };
        for a in 0..axis_len {
            let value = input_data[(lane_in + a as isize * input_axis_stride) as usize];
            if a < k {
                slots.place(a, value, a as u32);
                slots.bubble_up(a);
            } else if value > slots.worst(k) {
                slots.place(k - 1, value, a as u32);
                slots.bubble_up(k - 1);
            }
        }
    }
    Ok(())
}

/// Owned rank-2 top-`k` selection: `values` holds the `k` largest elements
/// per lane in descending order and `indices` their source positions along
/// the reduced axis. Both share the input shape with the reduced axis resized
/// to `k`.
#[derive(Debug, Clone)]
pub struct Topk<T> {
    /// Selected values, descending per lane.
    pub values: Array<T, VecStorage<T>, 2>,
    /// Source indices along the reduced axis.
    pub indices: Array<u32, VecStorage<u32>, 2>,
}

/// Apply a rank-2 top-`k` selection into newly allocated C-contiguous outputs.
///
/// See [`topk_axis_into`] for the contract and errors.
pub fn topk_axis<T: Scalar>(input: &ArrayView<'_, T, 2>, axis: usize, k: usize) -> Result<Topk<T>> {
    if axis >= 2 {
        return Err(LetoError::StorageError {
            reason: format!("axis {axis} out of bounds for rank 2"),
        });
    }
    let mut shape = input.shape();
    shape[axis] = k;
    let values_layout = Layout::c_contiguous(shape)?;
    let indices_layout = Layout::c_contiguous(shape)?;
    let size = values_layout.checked_size()?;
    let mut values = Array::new(values_layout, VecStorage::uninit(size))?;
    let mut indices = Array::new(indices_layout, VecStorage::uninit(size))?;
    topk_axis_into(
        input,
        axis,
        k,
        &mut values.view_mut(),
        &mut indices.view_mut(),
    )?;
    Ok(Topk { values, indices })
}

/// One lane's output slots: the value/index buffers plus this lane's base
/// offsets and axis strides.
struct LaneSlots<'a, T> {
    values: &'a mut [T],
    indices: &'a mut [u32],
    values_base: isize,
    indices_base: isize,
    values_stride: isize,
    indices_stride: isize,
}

impl<T: Scalar> LaneSlots<'_, T> {
    /// Write one selected element into its output slots.
    #[inline]
    fn place(&mut self, slot: usize, value: T, index: u32) {
        self.values[(self.values_base + slot as isize * self.values_stride) as usize] = value;
        self.indices[(self.indices_base + slot as isize * self.indices_stride) as usize] = index;
    }

    /// Read the current worst slot's value.
    #[inline]
    fn worst(&self, k: usize) -> T {
        self.values[(self.values_base + (k - 1) as isize * self.values_stride) as usize]
    }

    /// Bubble the occupant of `slot` up while the previous slot does not
    /// dominate it — the kernel's `>=`-break, verbatim.
    #[inline]
    fn bubble_up(&mut self, mut slot: usize) {
        while slot > 0 {
            let curr = (self.values_base + slot as isize * self.values_stride) as usize;
            let prev = (self.values_base + (slot - 1) as isize * self.values_stride) as usize;
            if self.values[prev] >= self.values[curr] {
                break;
            }
            self.values.swap(prev, curr);
            let curr_idx = (self.indices_base + slot as isize * self.indices_stride) as usize;
            let prev_idx = (self.indices_base + (slot - 1) as isize * self.indices_stride) as usize;
            self.indices.swap(prev_idx, curr_idx);
            slot -= 1;
        }
    }
}
