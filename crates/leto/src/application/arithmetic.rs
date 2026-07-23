//! Elementwise arithmetic operators on [`Array`] (ADR 0004).
//!
//! These operators are the **allocating convenience tier**: `&a + &b`,
//! `&a * scalar`, and `-&a` each produce a fresh C-contiguous array. They share
//! one logical-order traversal (`iter_elements`) so no second elementwise loop
//! exists in core. For hot paths that reuse an output buffer or need SIMD /
//! broadcasting, use the `leto-ops` `binary_map` / `scalar_map` family (the
//! performance tier) — the same two-tier split core already uses for reductions.
//!
//! `*` is **elementwise** (Hadamard product), matching `ndarray`. Matrix
//! multiplication is the explicit `MatrixProduct::matmul` method in `leto-ops`
//! (ADR 0003), so the consolidated array type has no `*`-means-matmul ambiguity.
//!
//! Operators cannot return `Result`; an unequal-shape `&Array op &Array` panics
//! with a message naming the violated invariant (the sanctioned operator
//! exception, matching `ndarray`/`nalgebra`). Scalar operators and `Neg` never
//! fail.

use crate::application::array::Array;
use crate::application::reduction::iter_elements;
use crate::infrastructure::storage::{Storage, VecStorage};
use eunomia::{Bf16, F16};

mod sealed {
    pub trait Sealed {}
}

/// Scalar types permitted on the right-hand side of an array–scalar operator
/// (`&array op scalar`).
///
/// Sealed: implemented only for the numeric primitives. The bound disambiguates
/// `&Array op scalar` from `&Array op &Array` under coherence (the `ndarray`
/// `ScalarOperand` pattern) and prevents an array from being treated as a
/// scalar operand.
pub trait ScalarOperand: sealed::Sealed + Copy {}

macro_rules! impl_scalar_operand {
    ($($t:ty),* $(,)?) => {
        $(
            impl sealed::Sealed for $t {}
            impl ScalarOperand for $t {}
        )*
    };
}

impl_scalar_operand!(
    f32, f64, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, F16, Bf16
);

/// Single core traversal for `&Array op &Array`. Panics on shape mismatch
/// (operator contract; see module docs).
#[track_caller]
fn binary_elementwise<T, Sa, Sb, F, const N: usize>(
    lhs: &Array<T, Sa, N>,
    rhs: &Array<T, Sb, N>,
    op: F,
) -> Array<T, VecStorage<T>, N>
where
    T: Copy,
    Sa: Storage<T>,
    Sb: Storage<T>,
    F: Fn(T, T) -> T,
{
    let shape = lhs.shape();
    assert_eq!(
        shape,
        rhs.shape(),
        "elementwise operator requires equal shapes: lhs {:?} != rhs {:?}",
        shape,
        rhs.shape()
    );
    let lhs_view = lhs.view();
    let rhs_view = rhs.view();
    let values: Vec<T> =
        if let (Some(l_slice), Some(r_slice)) = (lhs_view.as_slice(), rhs_view.as_slice()) {
            l_slice
                .iter()
                .zip(r_slice)
                .map(|(&a, &b)| op(a, b))
                .collect()
        } else {
            iter_elements(&lhs_view)
                .zip(iter_elements(&rhs_view))
                .map(|(a, b)| op(*a, *b))
                .collect()
        };
    Array::from_shape_vec(shape, values)
        .expect("invariant: C-contiguous shape matches the produced element count")
}

/// Single core traversal for `&Array op scalar` and `-&Array`.
fn unary_elementwise<T, Sa, F, const N: usize>(
    arr: &Array<T, Sa, N>,
    op: F,
) -> Array<T, VecStorage<T>, N>
where
    T: Copy,
    Sa: Storage<T>,
    F: Fn(T) -> T,
{
    let shape = arr.shape();
    let view = arr.view();
    let values: Vec<T> = if let Some(slice) = view.as_slice() {
        slice.iter().map(|&a| op(a)).collect()
    } else {
        iter_elements(&view).map(|a| op(*a)).collect()
    };
    Array::from_shape_vec(shape, values)
        .expect("invariant: C-contiguous shape matches the produced element count")
}

macro_rules! impl_array_operator {
    ($trait:ident, $method:ident) => {
        // Elementwise `&Array op &Array` (equal shape).
        impl<T, Sa, Sb, const N: usize> core::ops::$trait<&Array<T, Sb, N>> for &Array<T, Sa, N>
        where
            T: Copy + core::ops::$trait<Output = T>,
            Sa: Storage<T>,
            Sb: Storage<T>,
        {
            type Output = Array<T, VecStorage<T>, N>;
            #[inline]
            fn $method(self, rhs: &Array<T, Sb, N>) -> Self::Output {
                binary_elementwise(self, rhs, |a, b| core::ops::$trait::$method(a, b))
            }
        }

        // `&Array op scalar` broadcast of a single scalar over every element.
        impl<T, Sa, const N: usize> core::ops::$trait<T> for &Array<T, Sa, N>
        where
            T: ScalarOperand + core::ops::$trait<Output = T>,
            Sa: Storage<T>,
        {
            type Output = Array<T, VecStorage<T>, N>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                unary_elementwise(self, move |a| core::ops::$trait::$method(a, rhs))
            }
        }
    };
}

impl_array_operator!(Add, add);
impl_array_operator!(Sub, sub);
impl_array_operator!(Mul, mul);
impl_array_operator!(Div, div);

/// Elementwise negation `-&Array`.
impl<T, Sa, const N: usize> core::ops::Neg for &Array<T, Sa, N>
where
    T: Copy + core::ops::Neg<Output = T>,
    Sa: Storage<T>,
{
    type Output = Array<T, VecStorage<T>, N>;
    #[inline]
    fn neg(self) -> Self::Output {
        unary_elementwise(self, |a| -a)
    }
}
