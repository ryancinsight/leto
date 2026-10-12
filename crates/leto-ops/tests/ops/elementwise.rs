#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array, Layout, Storage, VecStorage};
use leto_ops::{
    add, binary_map, binary_map_with_cache_geometry, div, indexed_zip_mut_with, map, map_inplace,
    map_into, map_into_with_cache_geometry, mapv, mul, scalar_map, sub, unary_map, zip_mut_with,
    AddOp, CacheGeometry, EqOp, ErfOp, ErfcOp, GeOp, GtOp, LeOp, LgammaOp, LtOp, MulOp, NeOp,
};

fn assert_scalar_supertrait<T>()
where
    T: leto_ops::Scalar + eunomia::NumericElement,
{
}

fn assert_real_supertrait<T>()
where
    T: leto_ops::RealScalar,
{
}

fn scalar_count<T: leto_ops::Scalar>(count: usize) -> Result<T, eunomia::CountRangeError> {
    T::try_from_count(count)
}

mod binary;
mod reduced_precision;
mod scalar_traits;
mod unary;
mod zip;
