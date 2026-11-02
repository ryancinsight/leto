use super::kernel;
use crate::application::linalg::thresholds::machine_epsilon;
use crate::domain::real::RealScalar;
use eunomia::{Bf16, F16};

fn native_precision_contract<T: RealScalar>() {
    let n = 33;
    let mut parts = vec![T::ZERO; 2 * n * n];
    for i in 0..n {
        parts[2 * (i * n + i)] = T::from_count(if i < 2 { 3 } else { 5 });
    }
    parts[2 * (n - 1) * (n + 1)] = T::from_count(7);
    parts[3] = T::ONE;
    parts[2 * n + 1] = T::ONE.neg();
    let bound = machine_epsilon::<T>().mul(T::from_count(32));
    let result = kernel::solve(parts, n, bound, 128).expect("native Hermitian block");
    for (i, &value) in result.values.iter().enumerate() {
        let expected = T::from_count(match i {
            0 => 2,
            1 => 4,
            32 => 7,
            _ => 5,
        });
        assert!(value.sub(expected).abs() <= bound);
    }
    // The first eigenvector belongs to the (3, i; -i, 3) block.
    // Check A v = 2 v in T, not after widening to a reference precision.
    let x = result.vectors[0];
    let y = result.vectors[n];
    let imaginary = eunomia::Complex::new(T::ZERO, T::ONE);
    let residual = x + imaginary * y;
    assert!(residual.re.abs() <= bound && residual.im.abs() <= bound);
    let norm = (x.conj() * x + y.conj() * y).re;
    assert!(norm.sub(T::ONE).abs() <= bound);
    assert!(result.discarded_norm <= bound);
}

#[test]
fn hermitian_kernel_preserves_native_scalar_arithmetic() {
    native_precision_contract::<F16>();
    native_precision_contract::<Bf16>();
    native_precision_contract::<f32>();
    native_precision_contract::<f64>();
}
