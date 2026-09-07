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

#[pymodule]
fn leto_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    elementwise::register(m)?;
    linalg::register(m)?;
    Ok(())
}
