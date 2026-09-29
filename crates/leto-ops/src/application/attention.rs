//! Scaled dot-product attention over borrowed rank-3 Leto views.

mod backward;
mod error;
mod forward;
mod validation;

use core::num::NonZeroUsize;
use leto::{ArrayView, ArrayViewMut};

pub use backward::scaled_dot_product_attention_backward_accumulate;
pub use error::{AttentionError, AttentionOperand, AttentionResult};
pub use forward::scaled_dot_product_attention_into;

/// Masking policy for scaled dot-product attention.
#[derive(Clone, Copy)]
pub enum AttentionMask<'a, T> {
    /// Every query may attend to every key.
    Unmasked,
    /// Query `i` may attend only to keys `j <= i`.
    Causal,
    /// A nonzero value keeps a score; zero masks it.
    Keep(ArrayView<'a, T, 3>),
    /// Applies both causal and broadcast keep-mask constraints.
    CausalKeep(ArrayView<'a, T, 3>),
    /// A rank-two `[group, key]` keep mask repeated over a fixed number of
    /// consecutive execution batches per group.
    GroupedKeep(GroupedKeepMask<'a, T>),
    /// Applies causal masking and a grouped keep mask.
    CausalGroupedKeep(GroupedKeepMask<'a, T>),
}

impl<'a, T> AttentionMask<'a, T> {
    pub(super) const fn view(self) -> Option<ArrayView<'a, T, 3>> {
        match self {
            Self::Unmasked | Self::Causal | Self::GroupedKeep(_) | Self::CausalGroupedKeep(_) => {
                None
            }
            Self::Keep(view) | Self::CausalKeep(view) => Some(view),
        }
    }

    pub(super) const fn grouped(self) -> Option<GroupedKeepMask<'a, T>> {
        match self {
            Self::GroupedKeep(mask) | Self::CausalGroupedKeep(mask) => Some(mask),
            Self::Unmasked | Self::Causal | Self::Keep(_) | Self::CausalKeep(_) => None,
        }
    }

    pub(super) const fn is_causal(&self) -> bool {
        matches!(
            self,
            Self::Causal | Self::CausalKeep(_) | Self::CausalGroupedKeep(_)
        )
    }
}

/// Borrowed rank-two keep mask shared by consecutive attention batches.
#[derive(Clone, Copy)]
pub struct GroupedKeepMask<'a, T> {
    pub(super) view: ArrayView<'a, T, 2>,
    pub(super) batches_per_group: NonZeroUsize,
}

impl<'a, T> GroupedKeepMask<'a, T> {
    /// Creates a grouped mask from `[group, key]` values and the nonzero number
    /// of consecutive execution batches represented by each group.
    #[must_use]
    pub const fn new(view: ArrayView<'a, T, 2>, batches_per_group: NonZeroUsize) -> Self {
        Self {
            view,
            batches_per_group,
        }
    }
}

/// Optional additive gradient destinations for attention backward.
pub struct AttentionGradients<'a, T> {
    pub(super) query: Option<ArrayViewMut<'a, T, 3>>,
    pub(super) key: Option<ArrayViewMut<'a, T, 3>>,
    pub(super) value: Option<ArrayViewMut<'a, T, 3>>,
}

impl<'a, T> AttentionGradients<'a, T> {
    /// Creates a bundle of selected additive gradient destinations.
    #[must_use]
    pub const fn new(
        query: Option<ArrayViewMut<'a, T, 3>>,
        key: Option<ArrayViewMut<'a, T, 3>>,
        value: Option<ArrayViewMut<'a, T, 3>>,
    ) -> Self {
        Self { query, key, value }
    }

    pub(super) const fn has_any(&self) -> bool {
        self.query.is_some() || self.key.is_some() || self.value.is_some()
    }
}
