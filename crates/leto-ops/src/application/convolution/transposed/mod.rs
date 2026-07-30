//! Provider-owned transposed-convolution contract.

mod forward;
mod parameters;
mod plan;

pub use forward::convolution_transposed_forward_into;
pub use parameters::TransposedConvolutionParameters;
