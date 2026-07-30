//! Provider-owned N-dimensional convolution kernels.

mod forward;
mod plan;

pub use forward::convolution_forward_into;
pub use plan::ConvolutionParameters;
