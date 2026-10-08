//! Shared CTC fixture helpers.

use leto::{Array, Layout, VecStorage};
use leto_ops::RealScalar;

pub(crate) fn array<T, const N: usize>(
    shape: [usize; N],
    values: Vec<T>,
) -> Array<T, VecStorage<T>, N> {
    Array::new(
        Layout::c_contiguous(shape)
            .expect("invariant: analytical fixture satisfies the operation boundary"),
        VecStorage::new(values),
    )
    .expect("invariant: analytical fixture satisfies the operation boundary")
}

pub(crate) fn close<T: RealScalar>(actual: T, expected: f64, operations: usize) {
    // Unit roundoff u=epsilon/2. The caller counts the arithmetic and
    // elementary-function rounding sites along its longest dependency path.
    // gamma(k)=ku/(1-ku), scaled by 1+|reference|, also covers absolute error
    // near zero. Oracles enumerate paths independently in double precision.
    let half = T::ONE / (T::ONE + T::ONE);
    let mut epsilon = T::ONE;
    while T::ONE + epsilon * half > T::ONE {
        epsilon *= half;
    }
    let ku = epsilon.to_f64() * 0.5 * operations as f64;
    let bound = ku / (1.0 - ku) * (1.0 + expected.abs());
    assert!(ku < 1.0);
    assert!(
        (actual.to_f64() - expected).abs() <= bound,
        "actual {:?}, expected {expected}, bound {bound}",
        actual
    );
}

pub(crate) fn unit_roundoff<T: RealScalar>() -> f64 {
    let half = T::ONE / (T::ONE + T::ONE);
    let mut epsilon = T::ONE;
    while T::ONE + epsilon * half > T::ONE {
        epsilon *= half;
    }
    epsilon.to_f64() * 0.5
}
