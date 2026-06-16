//! Storage traits and backing implementations.

/// Copy-on-write storage that preserves borrowed read paths until mutation.
pub mod cow;
#[cfg(feature = "mnemosyne-alloc")]
/// Mnemosyne-backed owned storage.
pub mod mnemosyne;
/// Borrowed immutable and mutable slice-backed storage.
pub mod slice;
/// Stack-allocated fixed-capacity array storage.
pub mod stack;
/// Storage trait contracts shared by all backing implementations.
pub mod traits;
/// Owned vector-backed storage.
pub mod vec;

pub use cow::CowStorage;
#[cfg(feature = "mnemosyne-alloc")]
pub use mnemosyne::MnemosyneStorage;
pub use slice::{SliceStorage, SliceStorageMut};
pub use stack::StackStorage;
pub use traits::{Storage, StorageMut};
pub use vec::VecStorage;
