use std::fmt::Debug;

use leto::{Array1, Array2, Storage};
use leto_ops::{spmm_view_into, spmv_view_into, CsrProduct, CsrView, RealScalar, ScalarStrategy};

use super::trace;

pub(super) trait BinaryWitness: RealScalar {
    type Bits: Debug + Eq;
    const SPACING: Self;
    const MIN_NORMAL: Self;
    fn bits(self) -> Self::Bits;
}

impl BinaryWitness for f32 {
    type Bits = u32;
    const SPACING: Self = Self::EPSILON;
    const MIN_NORMAL: Self = Self::MIN_POSITIVE;

    fn bits(self) -> Self::Bits {
        self.to_bits()
    }
}

impl BinaryWitness for f64 {
    type Bits = u64;
    const SPACING: Self = Self::EPSILON;
    const MIN_NORMAL: Self = Self::MIN_POSITIVE;

    fn bits(self) -> Self::Bits {
        self.to_bits()
    }
}

impl BinaryWitness for eunomia::F16 {
    type Bits = u16;
    const SPACING: Self = Self::EPSILON;
    const MIN_NORMAL: Self = Self::MIN_POSITIVE;
    fn bits(self) -> Self::Bits {
        self.to_bits()
    }
}

impl BinaryWitness for eunomia::Bf16 {
    type Bits = u16;
    const SPACING: Self = Self::EPSILON;
    const MIN_NORMAL: Self = Self::MIN_POSITIVE;
    fn bits(self) -> Self::Bits {
        self.to_bits()
    }
}

fn compare_product_policies<T: BinaryWitness>() {
    let _case =
        tracing::info_span!("rounding_policy", scalar = std::any::type_name::<T>()).entered();
    let (values, columns, offsets, x, b) = {
        let _phase = tracing::info_span!("arrange").entered();
        // epsilon is the spacing at 1, not unit roundoff. Both inputs are
        // exactly representable; their exact product is 1 - epsilon^2.
        let values = [-T::ONE, T::ONE + T::SPACING];
        let x =
            Array1::from_shape_vec([2], vec![T::ONE, T::ONE - T::SPACING]).expect("vector shape");
        // A vector body plus a ragged tail; the legacy hook chooses its ISA.
        let mut dense = vec![T::ONE; 34];
        dense[17..].fill(T::ONE - T::SPACING);
        let b = Array2::from_shape_vec([2, 17], dense).expect("matrix shape");
        (values, [0, 1], [0, 2], x, b)
    };
    let (vector, matrix, hook_reference, fused_residual, separate) = {
        let _phase = tracing::info_span!("act").entered();
        let view = CsrView::from_slices(&values, &columns, &offsets, 1, 2).expect("canonical CSR");
        let mut vector = [T::ONE];
        let mut matrix = [T::ONE; 17];
        spmv_view_into(&view, &x.view(), &mut vector).expect("vector product");
        spmm_view_into(&view, &b.view(), &mut matrix).expect("matrix product");
        let mut hook_reference = [T::ZERO; 17];
        T::axpy_slice(-T::ONE, &[T::ONE; 17], &mut hook_reference);
        T::axpy_slice(
            T::ONE + T::SPACING,
            &[T::ONE - T::SPACING; 17],
            &mut hook_reference,
        );
        let mut separate = [T::ONE; 17];
        CsrProduct::<_, _, _, ScalarStrategy>::prepare(view, b.storage().as_slice(), 17)
            .expect("separate policy shape")
            .execute(&mut separate)
            .expect("separate output");
        (
            vector,
            matrix,
            hook_reference,
            -(T::SPACING * T::SPACING),
            separate,
        )
    };
    let _phase = tracing::info_span!("assert").entered();
    // Separate rounding makes the second product 1 before it cancels -1.
    assert_eq!(vector[0].bits(), T::ZERO.bits());
    assert_ne!(fused_residual, T::ZERO);
    assert!(separate.iter().all(|value| value.bits() == T::ZERO.bits()));
    for (actual, hook) in matrix.into_iter().zip(hook_reference) {
        assert_eq!(actual.bits(), hook.bits());
        // This is the existing ISA-dependent policy, not the acceptance
        // criterion for the proposed separately rounded product (only +0).
        assert!(actual.bits() == T::ZERO.bits() || actual.bits() == fused_residual.bits());
    }
}

#[test]
fn vector_rounding_and_matrix_hook_are_distinct_contracts() {
    let _trace = trace();
    compare_product_policies::<f32>();
    compare_product_policies::<f64>();
}
