#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

pub mod backward;
pub mod fixtures;
pub mod forward;
pub mod transposed;
