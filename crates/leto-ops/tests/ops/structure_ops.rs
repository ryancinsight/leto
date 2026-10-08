#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array, Layout, LetoError, SliceArg, Storage, VecStorage};
use leto_ops::{
    batched_matmul, coordinate_map_inplace, coordinate_map_plan, coordinate_map_plan_inplace,
    cumsum, indexed_fold, indexed_fold_fortran, indexed_map4_inplace, indexed_map_inplace,
    indexed_zip_mut_with, max as reduce_max, min as reduce_min, normal_with_seed, scan_axis,
    uniform_with_seed, zip_fold, zip_mut_with, CumProdOp, ScanDirection,
};

fn arr<const N: usize>(shape: [usize; N], data: Vec<f64>) -> Array<f64, VecStorage<f64>, N> {
    Array::new(Layout::c_contiguous(shape).unwrap(), VecStorage::new(data)).unwrap()
}

mod batched;
mod fold;
mod indexed;
mod zip;
