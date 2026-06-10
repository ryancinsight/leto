use crate::domain::error::{LetoError, Result};
use crate::domain::remove_axis::RankMarker;

/// A trait mapping dimension rank `N` to `N + 1` at compile-time on stable Rust.
///
/// The dual of [`RemoveAxis`](crate::domain::remove_axis::RemoveAxis): it lets a
/// rank-increasing operation (`stack`) name its `[usize; N + 1]` output shape
/// without nightly `generic_const_exprs`. Shared with `RemoveAxis` through the
/// same [`RankMarker`] zero-sized type.
pub trait InsertAxis<const N: usize> {
    /// The resulting rank after inserting one axis (always `N + 1`).
    const LARGER_RANK: usize;

    /// The shape type of the expanded layout (always `[usize; N + 1]`).
    type LargerShape: Copy + Send + Sync + 'static + AsRef<[usize]> + AsMut<[usize]>;

    /// Insert a dimension of length `value` at `axis` (valid range `0..=N`).
    fn insert_shape(
        &self,
        shape: [usize; N],
        axis: usize,
        value: usize,
    ) -> Result<Self::LargerShape>;
}

/// Fill `out` (length `N + 1`) with `shape` (length `N`) having `value`
/// inserted at `axis`. Single authoritative insertion body shared by every
/// rank implementation.
#[inline]
fn build_inserted(out: &mut [usize], shape: &[usize], axis: usize, value: usize) {
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = if i < axis {
            shape[i]
        } else if i == axis {
            value
        } else {
            shape[i - 1]
        };
    }
}

macro_rules! impl_insert_axis {
    ($from:literal => $to:literal) => {
        impl InsertAxis<$from> for RankMarker<$from> {
            const LARGER_RANK: usize = $to;
            type LargerShape = [usize; $to];

            #[inline]
            fn insert_shape(
                &self,
                shape: [usize; $from],
                axis: usize,
                value: usize,
            ) -> Result<Self::LargerShape> {
                if axis > $from {
                    return Err(LetoError::StorageError {
                        reason: format!("Insert axis {axis} out of bounds for rank {}", $from),
                    });
                }
                let mut out = [0usize; $to];
                build_inserted(&mut out, &shape, axis, value);
                Ok(out)
            }
        }
    };
}

impl_insert_axis!(0 => 1);
impl_insert_axis!(1 => 2);
impl_insert_axis!(2 => 3);
impl_insert_axis!(3 => 4);
impl_insert_axis!(4 => 5);
impl_insert_axis!(5 => 6);
impl_insert_axis!(6 => 7);
impl_insert_axis!(7 => 8);
