//! Executable witnesses for ADR 0036, using the current provider API.
//!
//! These characterize the compatibility gap; they do not implement or certify
//! the proposed signed, stored-order, externally scheduled product.

mod dispatch;
mod ordering;
mod partitions;
mod precision;
mod product;
mod properties;
mod rounding;
mod signed;

use tracing_subscriber::fmt::format::FmtSpan;

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .without_time()
            .with_test_writer()
            .with_span_events(FmtSpan::FULL)
            .finish(),
    )
}
