/// Storage traits and backing implementations.
pub mod storage;

pub use storage::{CowStorage, SliceStorage, SliceStorageMut, Storage, StorageMut, VecStorage};

#[cfg(feature = "mnemosyne-alloc")]
pub use storage::MnemosyneStorage;
