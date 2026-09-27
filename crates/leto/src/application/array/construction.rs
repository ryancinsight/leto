use super::types::Array;
use crate::domain::error::Result;
use crate::domain::layout::Layout;
use crate::infrastructure::storage::Storage;
use std::marker::PhantomData;

impl<T, S, const N: usize> Array<T, S, N>
where
    S: Storage<T>,
{
    /// Create a new Array from a layout and a storage backing.
    ///
    /// # Errors
    /// Returns an error if the layout accesses memory before offset 0, or exceeds the storage bounds.
    pub fn new(layout: Layout<N>, storage: S) -> Result<Self> {
        layout.validate_storage_len(storage.len())?;

        Ok(Self {
            layout,
            storage,
            _marker: PhantomData,
        })
    }
}
