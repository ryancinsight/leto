//! Python bindings for the Leto linear-algebra library.
//!
//! Exposes Leto and `leto-ops` operations to Python via PyO3.

#![allow(clippy::type_complexity)]
#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

mod elementwise;
mod linalg;
pub(crate) mod numpy_bridge;
#[cfg(test)]
mod support;

use pyo3::prelude::*;

// Preserve the GIL-required module contract from PyO3 0.23. The new PyO3
// default advertises free-threaded support, which needs a separate audit.
#[pymodule(gil_used = true)]
fn leto_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    elementwise::register(m)?;
    linalg::register(m)?;
    Ok(())
}
