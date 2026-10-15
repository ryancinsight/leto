//! Sealed representation policies. Address decoding follows checked construction.

mod sealed {
    pub trait Index {
        fn checked(self) -> Option<usize>;
        fn address(self) -> usize;
    }

    pub trait Ordering {
        const CANONICAL: bool;
    }
}

/// CSR index representations supported by the checked borrowed builder.
///
/// Sealed because the row kernel relies on `checked` being lossless and on
/// subsequent address decoding agreeing with it. No slice is reinterpreted.
pub trait CsrIndex: sealed::Index + Copy + core::fmt::Debug + Send + Sync + 'static {}

impl sealed::Index for usize {
    #[inline]
    fn checked(self) -> Option<usize> {
        Some(self)
    }

    #[inline]
    fn address(self) -> usize {
        self
    }
}
impl CsrIndex for usize {}

impl sealed::Index for i64 {
    #[inline]
    fn checked(self) -> Option<usize> {
        usize::try_from(self).ok()
    }

    #[inline]
    fn address(self) -> usize {
        // Construction proves 0 <= self <= usize::MAX on the target. This
        // address conversion is lossless even when pointer width is 32 bits.
        self as usize
    }
}
impl CsrIndex for i64 {}

/// Sealed CSR ordering invariant, chosen at construction, with no runtime state.
pub trait CsrOrdering: sealed::Ordering + Copy + core::fmt::Debug {}

/// Unique, strictly increasing columns in every row (the default).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Canonical;
impl sealed::Ordering for Canonical {
    const CANONICAL: bool = true;
}
impl CsrOrdering for Canonical {}

/// Preserve every stored entry in physical order, including duplicates and zeros.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct StoredOrder;
impl sealed::Ordering for StoredOrder {
    const CANONICAL: bool = false;
}
impl CsrOrdering for StoredOrder {}

#[inline]
pub(super) fn checked<I: CsrIndex>(index: I) -> Option<usize> {
    index.checked()
}

/// Only call for an index in a successfully validated descriptor.
#[inline]
pub(in crate::application::sparse) fn address<I: CsrIndex>(index: I) -> usize {
    index.address()
}

pub(super) const fn canonical<O: CsrOrdering>() -> bool {
    O::CANONICAL
}
