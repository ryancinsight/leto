//! Validated contiguous CSR products; see the co-located sparse design and ADR 0036.

mod arithmetic;
mod descriptor;
mod task;
mod width;

pub use arithmetic::CsrArithmetic;
pub use descriptor::{CsrOutput, CsrProduct};
pub use task::CsrRowTask;
pub use width::{DynamicWidth, FixedWidth, ProductWidth};
