//! Elementwise arithmetic operators on [`Array`] (ADR 0004).
//!
//! Two receiver forms share one traversal each:
//!
//! - **Borrowed lhs** (`&a + &b`, `&a * scalar`, `-&a`) is the allocating
//!   convenience tier: each produces a fresh C-contiguous array.
//! - **Owned lhs** (`a + &b`, `a + b`, `a * scalar`, `-a`) consumes the left
//!   operand and writes the result **into its existing allocation**, so a
//!   chained expression `&a + &b + &c` allocates once (at the leading `&a + &b`)
//!   instead of once per term. This is the ADR 0004 "owned receiver forms are
//!   additive follow-ups" consequence, not a change to the borrowed tier.
//!
//! For hot paths that reuse a caller-owned output or need SIMD / broadcasting,
//! use the `leto-ops` `binary_map` / `scalar_map` family (the performance tier)
//! — the same two-tier split core already uses for reductions. Core cannot call
//! those kernels itself: `leto-ops` depends on `leto`, so the SIMD tier is
//! downstream of this module by construction.
//!
//! `*` is **elementwise** (Hadamard product), matching `leto`. Matrix
//! multiplication is the explicit `MatrixProduct::matmul` method in `leto-ops`
//! (ADR 0003), so the consolidated array type has no `*`-means-matmul ambiguity.
//!
//! Operators cannot return `Result`; an unequal-shape `Array op Array` panics
//! with a message naming the violated invariant (the sanctioned operator
//! exception, matching `leto`/`leto`). Scalar operators and `Neg` never
//! fail. An owned-lhs operator additionally requires that the left operand's
//! layout address each element exactly once, since it writes through that
//! layout; see `binary_elementwise_in_place` (private).

use crate::application::array::Array;
use crate::application::reduction::iter_elements;
use crate::infrastructure::storage::{Storage, StorageMut, VecStorage};
use eunomia::{Bf16, F16};

mod sealed {
    pub trait Sealed {}
}

/// Scalar types permitted on the right-hand side of an array–scalar operator
/// (`&array op scalar`).
///
/// Sealed: implemented only for the numeric primitives. The bound disambiguates
/// `&Array op scalar` from `&Array op &Array` under coherence (the `leto`
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

/// Single core traversal for `Array op &Array`, writing the result back into the
/// owned left operand's existing allocation. Panics on shape mismatch (operator
/// contract; see module docs).
///
/// The dense route reuses the logical row-major slice; the strided route walks
/// `try_iter_mut`, which validates that the layout addresses each element
/// exactly once before yielding a single mutable reference. A left operand whose
/// layout aliases (a zero stride — reachable only by constructing an `Array`
/// from a hand-built [`crate::domain::layout::Layout`]) cannot be written
/// through and panics: the borrowed-lhs operators remain the reading form for
/// such layouts.
#[track_caller]
fn binary_elementwise_in_place<T, Sa, Sb, F, const N: usize>(
    lhs: &mut Array<T, Sa, N>,
    rhs: &Array<T, Sb, N>,
    op: F,
) where
    T: Copy,
    Sa: StorageMut<T>,
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
    let rhs_view = rhs.view();
    if let Some(out) = lhs.as_slice_mut() {
        if let Some(r_slice) = rhs_view.as_slice() {
            for (slot, &b) in out.iter_mut().zip(r_slice) {
                *slot = op(*slot, b);
            }
        } else {
            for (slot, b) in out.iter_mut().zip(iter_elements(&rhs_view)) {
                *slot = op(*slot, *b);
            }
        }
        return;
    }

    let slots = lhs
        .try_iter_mut()
        .expect("invariant: an owned operator lhs addresses each element exactly once");
    for (slot, b) in slots.zip(iter_elements(&rhs_view)) {
        *slot = op(*slot, *b);
    }
}

/// Single core traversal for `Array op scalar` and `-Array`, writing the result
/// back into the owned operand's existing allocation.
///
/// Element order is irrelevant for a unary map, so the dense route accepts any
/// contiguous layout (C or F order); the strided route carries the same
/// exactly-once requirement as [`binary_elementwise_in_place`].
#[track_caller]
fn unary_elementwise_in_place<T, Sa, F, const N: usize>(arr: &mut Array<T, Sa, N>, op: F)
where
    T: Copy,
    Sa: StorageMut<T>,
    F: Fn(T) -> T,
{
    if let Some(out) = arr.as_slice_memory_order_mut() {
        for slot in out.iter_mut() {
            *slot = op(*slot);
        }
        return;
    }

    let slots = arr
        .try_iter_mut()
        .expect("invariant: an owned operator operand addresses each element exactly once");
    for slot in slots {
        *slot = op(*slot);
    }
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

        // Owned `Array op &Array`: reuses the left operand's allocation, so a
        // chained expression allocates once rather than once per term.
        impl<T, Sa, Sb, const N: usize> core::ops::$trait<&Array<T, Sb, N>> for Array<T, Sa, N>
        where
            T: Copy + core::ops::$trait<Output = T>,
            Sa: StorageMut<T>,
            Sb: Storage<T>,
        {
            type Output = Array<T, Sa, N>;
            #[inline]
            #[track_caller]
            fn $method(mut self, rhs: &Array<T, Sb, N>) -> Self::Output {
                binary_elementwise_in_place(&mut self, rhs, |a, b| {
                    core::ops::$trait::$method(a, b)
                });
                self
            }
        }

        // Owned `Array op Array`: keeps the left allocation, drops the right.
        impl<T, Sa, Sb, const N: usize> core::ops::$trait<Array<T, Sb, N>> for Array<T, Sa, N>
        where
            T: Copy + core::ops::$trait<Output = T>,
            Sa: StorageMut<T>,
            Sb: Storage<T>,
        {
            type Output = Array<T, Sa, N>;
            #[inline]
            #[track_caller]
            fn $method(self, rhs: Array<T, Sb, N>) -> Self::Output {
                core::ops::$trait::$method(self, &rhs)
            }
        }

        // Owned `Array op scalar` in the left operand's allocation.
        impl<T, Sa, const N: usize> core::ops::$trait<T> for Array<T, Sa, N>
        where
            T: ScalarOperand + core::ops::$trait<Output = T>,
            Sa: StorageMut<T>,
        {
            type Output = Array<T, Sa, N>;
            #[inline]
            #[track_caller]
            fn $method(mut self, rhs: T) -> Self::Output {
                unary_elementwise_in_place(&mut self, move |a| core::ops::$trait::$method(a, rhs));
                self
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

/// Elementwise negation `-Array`, in the owned operand's allocation.
impl<T, Sa, const N: usize> core::ops::Neg for Array<T, Sa, N>
where
    T: Copy + core::ops::Neg<Output = T>,
    Sa: StorageMut<T>,
{
    type Output = Array<T, Sa, N>;
    #[inline]
    #[track_caller]
    fn neg(mut self) -> Self::Output {
        unary_elementwise_in_place(&mut self, |a| -a);
        self
    }
}
