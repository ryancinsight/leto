use crate::domain::error::{LetoError, Result};
use crate::domain::layout::kernels;

use super::Layout;

impl<const N: usize> Layout<N> {
    /// Returns the logical number of elements represented by this layout.
    pub fn size(&self) -> usize {
        self.checked_size()
            .expect("layout shape product must fit in usize")
    }

    /// Returns the logical number of elements with overflow validation.
    pub fn checked_size(&self) -> Result<usize> {
        kernels::shape_size(&self.shape)
    }

    /// Returns the minimum and maximum physical offsets spanned by this layout.
    pub fn min_max_offsets(&self) -> (usize, usize) {
        if N == 1 {
            let dim = self.shape[0];
            if dim == 0 {
                return (self.offset, self.offset);
            }
            let s = self.strides[0];
            let bound = (dim - 1) as isize * s;
            if bound >= 0 {
                (self.offset, self.offset + bound as usize)
            } else {
                let min_offset = self.offset as isize + bound;
                assert!(min_offset >= 0, "layout accesses negative physical offset");
                (min_offset as usize, self.offset)
            }
        } else {
            self.checked_min_max_offsets()
                .expect("layout physical offsets must be non-negative and fit in usize")
        }
    }

    /// Returns the minimum and maximum physical offsets with signed overflow validation.
    pub fn checked_min_max_offsets(&self) -> Result<(usize, usize)> {
        kernels::min_max_offsets(&self.shape, &self.strides, self.offset)
    }

    /// Validates that every addressable physical element is inside `storage_len`.
    pub fn validate_storage_len(&self, storage_len: usize) -> Result<()> {
        kernels::validate_storage(&self.shape, &self.strides, self.offset, storage_len)
    }

    /// Returns true when multiple logical mutable indices can address one element.
    ///
    /// An empty layout (total element count zero) can never alias: there are no
    /// addressable elements, so overlapping writes are impossible. This arises
    /// naturally when a zero-sized axis collapses the row-major stride of a
    /// leading dimension to 0 (e.g. `shape=[8,0,8] \rightarrow strides=[0,8,1]`).
    /// Treating that as aliasing would reject no-op kernels on degenerate slices
    /// that provably cannot conflict.
    pub fn has_zero_stride_aliasing(&self) -> bool {
        if self.size() == 0 {
            return false;
        }
        self.shape
            .iter()
            .zip(self.strides.iter())
            .any(|(&dim, &stride)| dim > 1 && stride == 0)
    }

    /// Reinterpret a dense row-major layout with a new shape.
    ///
    /// This preserves logical row-major element order and performs no
    /// materialization. Strided, Fortran-order, or broadcasted layouts must be
    /// materialized with `to_contiguous` before reshaping.
    pub fn reshape<const M: usize>(&self, shape: [usize; M]) -> Result<Layout<M>> {
        let target = Layout::<M>::c_contiguous(shape)?;
        if self.checked_size()? != target.checked_size()? {
            return Err(LetoError::ShapeMismatch {
                lhs: self.shape.to_vec(),
                rhs: shape.to_vec(),
            });
        }
        if !self.is_c_dense() {
            return Err(LetoError::StorageError {
                reason: "reshape requires a dense row-major layout".to_string(),
            });
        }

        Ok(Layout {
            shape,
            strides: target.strides,
            offset: self.offset,
        })
    }
}

impl Layout<3> {
    /// Returns whether distinct logical indices address distinct elements.
    ///
    /// The separated-stride case completes in constant time. Ambiguous rank-3
    /// layouts use an exact bounded integer-difference search, so valid
    /// transposed and padded layouts are accepted while every colliding layout
    /// is rejected. Empty and single-element layouts are injective.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::Overflow`] when the span proof exceeds `u128`.
    pub fn is_injective(&self) -> Result<bool> {
        if self.checked_size()? <= 1 {
            return Ok(true);
        }

        let mut axes = [(0_u128, 0_usize); 3];
        let mut axis_count = 0;
        for (&dimension, &stride) in self.shape.iter().zip(self.strides.iter()) {
            if dimension <= 1 {
                continue;
            }
            let magnitude = stride.unsigned_abs() as u128;
            if magnitude == 0 {
                return Ok(false);
            }
            axes[axis_count] = (magnitude, dimension);
            axis_count += 1;
        }
        axes[..axis_count].sort_unstable_by_key(|&(stride, _)| stride);

        let mut covered_span = 0_u128;
        let mut separated = true;
        for &(stride, dimension) in &axes[..axis_count] {
            if stride <= covered_span {
                separated = false;
                break;
            }
            covered_span = covered_span
                .checked_add((dimension - 1) as u128 * stride)
                .ok_or(LetoError::Overflow {
                    reason: "rank-3 layout injectivity span",
                })?;
        }
        if separated {
            return Ok(true);
        }

        // A collision exists exactly when a bounded, non-zero index-difference
        // vector has zero stride dot product. Solve the largest-range axis and
        // enumerate the other two dimensions.
        let bounds = self
            .shape
            .map(|dimension| (dimension.saturating_sub(1)) as i128);
        let strides = self.strides.map(|stride| stride as i128);
        let solve_axis = bounds
            .iter()
            .enumerate()
            .max_by_key(|&(_, bound)| bound)
            .map_or(0, |(axis, _)| axis);
        let pair = match solve_axis {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        };
        let solve_stride = strides[solve_axis];
        for first in -bounds[pair[0]]..=bounds[pair[0]] {
            for second in -bounds[pair[1]]..=bounds[pair[1]] {
                let residual = first * strides[pair[0]] + second * strides[pair[1]];
                if residual % solve_stride != 0 {
                    continue;
                }
                let solved = -residual / solve_stride;
                if solved.abs() <= bounds[solve_axis] && (first != 0 || second != 0 || solved != 0)
                {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::Layout;

    /// C-contiguous stride computation collapses the leading stride to 0 when an
    /// interior axis has size 0 (see `kernels::c_contiguous_strides`). The
    /// resulting layout is empty (size 0), so it can never alias regardless of
    /// the zero stride on its leading dimension.
    #[test]
    fn empty_c_contiguous_layout_is_not_aliasing() {
        let layout = Layout::<3>::c_contiguous([8, 0, 8]).expect("invariant: valid shape");
        assert_eq!(layout.strides, [0, 8, 1]);
        assert_eq!(layout.size(), 0);
        assert!(!layout.has_zero_stride_aliasing());
    }

    /// F-contiguous layouts apply the same defensive collapse for zero dims.
    #[test]
    fn empty_f_contiguous_layout_is_not_aliasing() {
        let layout = Layout::<3>::f_contiguous([8, 0, 8]).expect("invariant: valid shape");
        assert_eq!(layout.size(), 0);
        assert!(!layout.has_zero_stride_aliasing());
    }

    /// A genuine zero-stride axis with `dim > 1` readdresses the same physical
    /// element from multiple logical indices, so it DOES alias. Positive
    /// control: every other axis remains non-degenerate to keep size nonzero.
    #[test]
    fn zero_stride_axis_with_nonunit_size_does_alias() {
        // shape [3, 4, 5], strides [5, 0, 1]: axis-1 has dim 4 and stride 0,
        // so logical indices `[_, 0..4, _]` all map to the same row.
        let layout = Layout::<3>::new([3, 4, 5], [5, 0, 1], 0);
        assert_eq!(layout.size(), 60);
        assert!(layout.has_zero_stride_aliasing());
    }

    /// A broadcast axis (`dim = 1, stride = 0`) does NOT by itself alias, because
    /// a single-element axis has only one logical index. Anti-regression for
    /// the (mistaken) notion that any zero stride triggers the predicate.
    #[test]
    fn broadcast_axis_alone_is_not_aliasing() {
        let layout = Layout::<3>::new([3, 1, 5], [5, 0, 1], 0);
        assert_eq!(layout.size(), 15);
        assert!(!layout.has_zero_stride_aliasing());
    }

    /// A broadcast layout whose total size is zero (broadcast axis plus a
    /// zero-dim axis) cannot alias: there are no writes to overlap.
    #[test]
    fn broadcast_layout_with_zero_dim_is_not_aliasing() {
        // shape [3, 1, 0], strides [0, 0, 1]: size 0 so no aliasing.
        let layout = Layout::<3>::new([3, 1, 0], [0, 0, 1], 0);
        assert_eq!(layout.size(), 0);
        assert!(!layout.has_zero_stride_aliasing());
    }

    #[test]
    fn rank_three_injectivity_accepts_ambiguous_non_overlapping_layout() {
        let layout = Layout::<3>::new([1, 3, 2], [6, 2, 3], 0);
        assert!(layout.is_injective().expect("injectivity proof"));
    }

    #[test]
    fn rank_three_injectivity_rejects_nonzero_stride_collision() {
        let layout = Layout::<3>::new([1, 3, 2], [6, 2, 4], 0);
        assert!(!layout.is_injective().expect("injectivity proof"));
    }
}
