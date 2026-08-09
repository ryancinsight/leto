//! Provider-owned transposed-convolution contract.

mod backward;
mod forward;
mod plan;

pub use backward::{convolution_transposed_backward_accumulate, TransposedConvolutionGradients};
pub use forward::convolution_transposed_forward_into;
