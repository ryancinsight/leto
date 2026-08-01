//! Scalar-preserving, zero-copy stateful parameter updates.

mod parameters;
mod rules;
mod validation;

pub use parameters::{
    AdaGradParameters, AdamParameters, AdamWParameters, RmsPropParameters, SgdParameters,
};
pub use rules::{stateful_update, AdaGrad, Adam, AdamW, RmsProp, Sgd, StatefulUpdateRule};
