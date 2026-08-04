//! Loss operations over borrowed Leto views.

mod backward;
mod error;
mod forward;
mod validation;

pub use backward::cross_entropy_backward_accumulate;
pub use error::{CrossEntropyError, CrossEntropyOperand, CrossEntropyResult};
pub use forward::cross_entropy_forward_into;
