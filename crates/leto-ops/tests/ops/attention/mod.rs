use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array, ArrayView, ArrayViewMut, Layout, Storage, VecStorage};
use leto_ops::{
    scaled_dot_product_attention_backward_accumulate, scaled_dot_product_attention_into,
    AttentionError, AttentionGradients, AttentionMask, AttentionOperand, RealScalar,
};

fn array<T: Clone>(shape: [usize; 3], values: Vec<T>) -> Array<T, VecStorage<T>, 3> {
    Array::new(
        Layout::c_contiguous(shape).expect("test shape is representable"),
        VecStorage::new(values),
    )
    .expect("test storage matches its shape")
}

fn assert_close<T: NumericElement>(actual: &[T], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual.to_f64() - expected).abs();
        assert!(
            error <= tolerance,
            "index {index}: actual {}, expected {expected}, error {error}, tolerance {tolerance}",
            actual.to_f64()
        );
    }
}

mod backward;
mod forward;
mod validation;
