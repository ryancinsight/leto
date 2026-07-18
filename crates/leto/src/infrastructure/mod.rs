/// Sparse array storage formats (CSR, CSC, COO, block-sparse).
pub mod sparse;
/// Storage traits and backing implementations.
pub mod storage;

pub use storage::{CowStorage, SliceStorage, SliceStorageMut, Storage, StorageMut, VecStorage};

#[cfg(feature = "mnemosyne-alloc")]
pub use storage::MnemosyneStorage;

#[cfg(feature = "ndarray-compat")]
/// Zero-copy and ownership-aware ndarray boundary conversions.
pub mod ndarray_compat;
