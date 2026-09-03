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

    /// Returns whether distinct logical indices address distinct elements.
    ///
    /// The separated-stride case completes in `O(N log N)` time without heap
    /// allocation. Ambiguous layouts use an exact bounded integer-difference
    /// search, preserving arbitrary injective views without allocating an
    /// offset set. Empty and single-element layouts are injective.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::Overflow`] when exact difference arithmetic exceeds
    /// `i128`.
    pub fn is_injective(&self) -> Result<bool> {
        kernels::is_injective(&self.shape, &self.strides)
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

        Layout::try_new(shape, target.strides, self.offset)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

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
        let layout = Layout::<3>::from_parts_unchecked([3, 4, 5], [5, 0, 1], 0);
        assert_eq!(layout.size(), 60);
        assert!(layout.has_zero_stride_aliasing());
    }

    /// A broadcast axis (`dim = 1, stride = 0`) does NOT by itself alias, because
    /// a single-element axis has only one logical index. Anti-regression for
    /// the (mistaken) notion that any zero stride triggers the predicate.
    #[test]
    fn broadcast_axis_alone_is_not_aliasing() {
        let layout = Layout::<3>::from_parts_unchecked([3, 1, 5], [5, 0, 1], 0);
        assert_eq!(layout.size(), 15);
        assert!(!layout.has_zero_stride_aliasing());
    }

    /// A broadcast layout whose total size is zero (broadcast axis plus a
    /// zero-dim axis) cannot alias: there are no writes to overlap.
    #[test]
    fn broadcast_layout_with_zero_dim_is_not_aliasing() {
        // shape [3, 1, 0], strides [0, 0, 1]: size 0 so no aliasing.
        let layout = Layout::<3>::from_parts_unchecked([3, 1, 0], [0, 0, 1], 0);
        assert_eq!(layout.size(), 0);
        assert!(!layout.has_zero_stride_aliasing());
    }

    #[test]
    fn rank_three_injectivity_accepts_ambiguous_non_overlapping_layout() {
        let layout = Layout::<3>::from_parts_unchecked([1, 3, 2], [6, 2, 3], 0);
        assert!(layout.is_injective().expect("injectivity proof"));
    }

    #[test]
    fn rank_three_injectivity_rejects_nonzero_stride_collision() {
        let layout = Layout::<3>::from_parts_unchecked([1, 3, 2], [6, 2, 4], 0);
        assert!(!layout.is_injective().expect("injectivity proof"));
    }

    #[test]
    fn rank_eight_separation_proves_transposed_layout_injective() {
        let layout = Layout::<8>::from_parts_unchecked(
            [2, 3, 1, 1, 1, 1, 1, 1],
            [1, 2, 0, 0, 0, 0, 0, 0],
            0,
        );
        assert!(layout.is_injective().expect("injectivity proof"));
    }

    #[test]
    fn generic_separation_rejects_nonzero_stride_collision() {
        let layout = Layout::<4>::from_parts_unchecked([2, 2, 1, 1], [1, 1, 0, 0], 0);
        assert!(!layout.is_injective().expect("injectivity proof"));
    }

    #[test]
    fn generic_injectivity_accepts_interleaved_non_overlapping_layout() {
        let layout = Layout::<2>::from_parts_unchecked([2, 3], [3, 2], 0);
        assert!(layout.is_injective().expect("injectivity proof"));
    }

    #[test]
    fn generic_injectivity_matches_exhaustive_offset_oracle() {
        for first_dimension in 1..=3 {
            for second_dimension in 1..=3 {
                for third_dimension in 1..=3 {
                    let shape = [first_dimension, second_dimension, third_dimension];
                    for first_stride in -3..=3 {
                        for second_stride in -3..=3 {
                            for third_stride in -3..=3 {
                                let strides = [first_stride, second_stride, third_stride];
                                let mut offsets = BTreeSet::new();
                                for first in 0..first_dimension {
                                    for second in 0..second_dimension {
                                        for third in 0..third_dimension {
                                            offsets.insert(
                                                first as isize * first_stride
                                                    + second as isize * second_stride
                                                    + third as isize * third_stride,
                                            );
                                        }
                                    }
                                }
                                let expected = offsets.len()
                                    == first_dimension * second_dimension * third_dimension;
                                let actual = Layout::<3>::from_parts_unchecked(shape, strides, 0)
                                    .is_injective()
                                    .expect("small exact injectivity domain");
                                assert_eq!(
                                    actual, expected,
                                    "shape={shape:?}, strides={strides:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
