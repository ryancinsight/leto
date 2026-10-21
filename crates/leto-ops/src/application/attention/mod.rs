//! Scaled dot-product attention over borrowed rank-3 Leto views.

mod backward;
mod error;
mod forward;
mod types;
mod validation;

pub use backward::scaled_dot_product_attention_backward_accumulate;
pub use error::{AttentionError, AttentionOperand, AttentionResult};
pub use forward::scaled_dot_product_attention_into;
pub use types::{AttentionGradients, AttentionMask, GroupedKeepMask};
