#[path = "../../../../../tests/ops/layout/payloads.rs"]
mod payloads;
#[path = "../../../../../tests/ops/layout/square_contract.rs"]
mod square_contract;

use eunomia::{Bf16, F16};

#[test]
fn scalar_square_transpose_preserves_coordinate_payloads_and_guards() {
    square_contract::assert_squares::<f32>(super::transpose_scalar);
    square_contract::assert_squares::<f64>(super::transpose_scalar);
    square_contract::assert_squares::<F16>(super::transpose_scalar);
    square_contract::assert_squares::<Bf16>(super::transpose_scalar);
}
