#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use eunomia::Complex64;
use leto::{Array, Array1, Array2, SliceArg, Storage};
use leto_ops::{
    csc_spmv_into, csc_spmv_view_into, spgemm, spmm, spmm_into, spmm_view_into, spmv, spmv_into,
    spmv_view_into, CooMatrix, CscMatrix, CscView, CsrMatrix, CsrView,
};

mod csr;
mod products;
mod views;
