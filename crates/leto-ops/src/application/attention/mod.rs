//! Scaled dot-product attention over borrowed rank-3 Leto views.

mod backward;
mod error;
mod forward;
mod validation;

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
}

impl<'a, T> AttentionMask<'a, T> {
    pub(super) const fn view(self) -> Option<ArrayView<'a, T, 3>> {
        match self {
            Self::Unmasked | Self::Causal => None,
            Self::Keep(view) | Self::CausalKeep(view) => Some(view),
        }
    }

    pub(super) const fn is_causal(&self) -> bool {
        matches!(self, Self::Causal | Self::CausalKeep(_))
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
