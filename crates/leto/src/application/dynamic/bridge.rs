//! Zero-copy rank bridge between the const-rank [`Array`] and the runtime-rank
//! [`ArrayD`] (ADR 0007).
//!
//! Both directions move the storage `S` by value and translate only the
//! `O(ndim)` shape/stride scalars between `[_; N]` and `Box<[_]>` — no element is
//! read, copied, or reallocated (see the ADR's allocation-free theorem). This is
//! the single sanctioned path from runtime rank into the const-rank compute
//! kernels: `arr_d.into_dimensionality::<N>()?` then any existing operation.

use std::marker::PhantomData;

use crate::application::array::Array;
use crate::application::dynamic::ArrayD;
use crate::domain::dynamic::LayoutDyn;
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::infrastructure::storage::Storage;

impl<T, S, const N: usize> Array<T, S, N>
where
    S: Storage<T>,
{
    /// Erase the compile-time rank, producing a runtime-rank [`ArrayD`].
    ///
    /// Zero-copy: the storage is moved unchanged; only the `N` shape/stride
    /// scalars are boxed. Always succeeds.
    pub fn into_dyn(self) -> ArrayD<T, S> {
        let layout = self.layout; // `Layout<N>` is `Copy`.
        let dyn_layout = LayoutDyn::new(
            layout.shape().to_vec().into_boxed_slice(),
            layout.strides().to_vec().into_boxed_slice(),
            layout.offset(),
        )
        .expect("invariant: shape and strides both have length N");
        ArrayD {
            layout: dyn_layout,
            storage: self.storage,
            _marker: PhantomData,
        }
    }
}

impl<T, S> ArrayD<T, S>
where
    S: Storage<T>,
{
    /// Recover a const-rank [`Array<T, S, N>`], the gateway to all compute.
    ///
    /// Zero-copy: the storage is moved unchanged; only the shape/stride scalars
    /// are copied back into `[_; N]`.
    ///
    /// # Errors
    /// [`LetoError::StorageError`] if the runtime rank does not equal `N`.
    pub fn into_dimensionality<const N: usize>(self) -> Result<Array<T, S, N>> {
        if self.layout.ndim() != N {
            return Err(LetoError::StorageError {
                reason: format!(
                    "array rank {} does not match requested const rank {N}",
                    self.layout.ndim()
                ),
            });
        }
        let shape: [usize; N] =
            self.layout
                .shape
                .as_ref()
                .try_into()
                .map_err(|_| LetoError::StorageError {
                    reason: "rank-checked shape failed fixed-length conversion".to_string(),
                })?;
        let strides: [isize; N] =
            self.layout
                .strides
                .as_ref()
                .try_into()
                .map_err(|_| LetoError::StorageError {
                    reason: "rank-checked strides failed fixed-length conversion".to_string(),
                })?;
        let layout = Layout::from_parts_unchecked(shape, strides, self.layout.offset);
        Array::new(layout, self.storage)
    }
}
