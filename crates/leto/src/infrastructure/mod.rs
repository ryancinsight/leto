pub mod storage;

pub use storage::{Storage, StorageMut, SliceStorage, SliceStorageMut, VecStorage};

#[cfg(feature = "mnemosyne-alloc")]
pub use storage::MnemosyneStorage;
