//! Validated contiguous CSR products; see the co-located sparse design and ADR 0036.

mod arithmetic;
mod task;

use super::{Canonical, CsrIndex, CsrOrdering, CsrView};
use crate::domain::strategy::ScalarStrategy;
use core::marker::PhantomData;
use eunomia::NumericElement;
use leto::{LetoError, Result};

pub use arithmetic::CsrArithmetic;
pub use task::CsrRowTask;

/// Prepared `m*k` CSR times contiguous row-major `k*n` dense multiplication.
///
/// Borrows all original input slices. Preparation checks all shape arithmetic;
/// binding checks output length before any mutation. Each row is computed by
/// one authoritative provider kernel, independently of the execution policy.
///
/// Borrowed RHS storage cannot expire before execution:
/// ```compile_fail
/// use leto_ops::{CsrProduct, CsrView, ScalarStrategy};
/// let view = CsrView::from_slices(&[3i64], &[0], &[0, 1], 1, 1).unwrap();
/// let product = {
///     let rhs = vec![7i64];
///     CsrProduct::<_, _, _, ScalarStrategy>::prepare(view, &rhs, 1).unwrap()
/// };
/// product.execute(&mut [0]).unwrap();
/// ```
#[derive(Debug)]
pub struct CsrProduct<
    'a,
    T: NumericElement,
    I: CsrIndex = usize,
    O: CsrOrdering = Canonical,
    A: CsrArithmetic<T> = ScalarStrategy,
> {
    view: CsrView<'a, T, I, O>,
    rhs: &'a [T],
    width: usize,
    output_len: usize,
    arithmetic: PhantomData<A>,
}

// A is a type-level arithmetic selection, never an owned strategy value. A
// descriptor copy duplicates only shared borrows/dimensions, so it must not
// require (or add) Copy on the existing public strategy types.
impl<T: NumericElement, I: CsrIndex, O: CsrOrdering, A: CsrArithmetic<T>> Copy
    for CsrProduct<'_, T, I, O, A>
{
}

impl<T: NumericElement, I: CsrIndex, O: CsrOrdering, A: CsrArithmetic<T>> Clone
    for CsrProduct<'_, T, I, O, A>
{
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: NumericElement, I: CsrIndex, O: CsrOrdering, A: CsrArithmetic<T>>
    CsrProduct<'a, T, I, O, A>
{
    /// Check exact dense shape and representable input/output element counts.
    /// No allocation occurs for a valid request.
    ///
    /// # Errors
    /// [`LetoError::StorageError`] for count overflow; [`LetoError::ShapeMismatch`]
    /// for an RHS length other than `view.ncols() * rhs_columns`.
    #[inline]
    pub fn prepare(view: CsrView<'a, T, I, O>, rhs: &'a [T], rhs_columns: usize) -> Result<Self> {
        let count = |rows: usize| {
            rows.checked_mul(rhs_columns)
                .ok_or_else(|| LetoError::StorageError {
                    reason: "CSR product element count overflows usize".into(),
                })
        };
        let output_len = count(view.nrows())?;
        let rhs_len = count(view.ncols())?;
        if rhs.len() != rhs_len {
            return Err(LetoError::ShapeMismatch {
                lhs: vec![rhs.len()],
                rhs: vec![rhs_len],
            });
        }
        Ok(Self {
            view,
            rhs,
            width: rhs_columns,
            output_len,
            arithmetic: PhantomData,
        })
    }

    /// Bind initialized output storage exclusively without writing it.
    ///
    /// # Errors
    /// [`LetoError::ShapeMismatch`] unless output has exactly `m*n` elements.
    #[inline]
    pub fn bind_output<'o>(&self, output: &'o mut [T]) -> Result<CsrOutput<'a, 'o, T, I, O, A>> {
        if output.len() != self.output_len {
            return Err(LetoError::ShapeMismatch {
                lhs: vec![output.len()],
                rhs: vec![self.output_len],
            });
        }
        Ok(CsrOutput {
            product: *self,
            output,
        })
    }

    /// Overwrite the complete output serially through the common row task.
    ///
    /// # Errors
    /// [`LetoError::ShapeMismatch`] on a wrong output length, before any write.
    #[inline]
    pub fn execute(&self, output: &mut [T]) -> Result<()> {
        let bound = self.bind_output(output)?;
        // SAFETY: the loop invokes each in-range row once synchronously. Task
        // copies never escape; the binding keeps inputs/output borrowed.
        unsafe {
            bound.dispatch_with(|rows, task| {
                for row in 0..rows {
                    task.run(row);
                }
            });
        }
        Ok(())
    }
}

/// Exclusively borrowed destination for a prepared CSR product.
///
/// The destination contains initialized `T` values before binding. Computation
/// overwrites every row with the additive identity before accumulating entries.
///
/// A live output binding excludes other mutable access:
/// ```compile_fail
/// use leto_ops::{CsrProduct, CsrView, ScalarStrategy};
/// let view = CsrView::from_slices(&[3i64], &[0], &[0, 1], 1, 1).unwrap();
/// let product = CsrProduct::<_, _, _, ScalarStrategy>::prepare(view, &[7], 1).unwrap();
/// let mut output = [0];
/// let bound = product.bind_output(&mut output).unwrap();
/// output[0] = 9;
/// unsafe { bound.dispatch_with(|_, task| task.run(0)); }
/// ```
#[derive(Debug)]
pub struct CsrOutput<'a, 'o, T: NumericElement, I: CsrIndex, O: CsrOrdering, A: CsrArithmetic<T>> {
    product: CsrProduct<'a, T, I, O, A>,
    output: &'o mut [T],
}

impl<T: NumericElement, I: CsrIndex, O: CsrOrdering, A: CsrArithmetic<T>>
    CsrOutput<'_, '_, T, I, O, A>
{
    /// Dispatch the prepared row task within this synchronous borrow scope.
    /// Returns the dispatcher's result unchanged, including errors.
    ///
    /// # Safety
    /// On success invoke every row in `0..m` exactly once, and no other row.
    /// On error/panic any subset may run, at most once per row. All admitted
    /// callbacks must finish before this call returns **or unwinds**, and no
    /// task copy may be used afterward. Keep shared inputs immutable/alive and
    /// do not access output through another alias until dispatch completes.
    /// This obligation includes joining workers whose callbacks unwind.
    ///
    /// The erased task lifetime accommodates `'static` backend callbacks; it
    /// does not extend any allocation's lifetime. See [`CsrRowTask::run`].
    #[inline]
    pub unsafe fn dispatch_with<R>(
        self,
        dispatch: impl FnOnce(usize, CsrRowTask<T, I, A>) -> R,
    ) -> R {
        let task = CsrRowTask::new(&self.product, self.output.as_mut_ptr());
        dispatch(self.product.view.nrows(), task)
    }
}
