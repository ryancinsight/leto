/// Storage traits and backing implementations.
pub mod storage;
/// Sparse array storage formats (CSR, CSC, COO, block-sparse).
pub mod sparse;

pub use storage::{CowStorage, SliceStorage, SliceStorageMut, Storage, StorageMut, VecStorage};

#[cfg(feature = "mnemosyne-alloc")]
pub use storage::MnemosyneStorage;

#[cfg(feature = "ndarray-compat")]
pub mod ndarray_compat;
