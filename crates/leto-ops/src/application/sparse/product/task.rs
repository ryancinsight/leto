//! Lifetime-erased row task, constructible only inside a validated output binding.

use super::super::{csr::address, CsrIndex, CsrOrdering};
use super::{arithmetic::accumulate, CsrArithmetic, CsrProduct};
use core::marker::PhantomData;
use eunomia::NumericElement;

/// Copyable provider-owned row kernel for a synchronous dispatch scope.
///
/// It owns no buffer and exposes no safe pointer access. Copies remain usable
/// only within the unsafe [`super::CsrOutput::dispatch_with`] scope that creates
/// them. Scalar/index/arithmetic policies monomorphize; no virtual dispatch or
/// scheduler selection occurs in a worker.
///
/// A task never grants safe row or pointer access, even inside dispatch:
/// ```compile_fail
/// fn outside_scope(task: leto_ops::CsrRowTask<i64, usize, leto_ops::ScalarStrategy>) {
///     task.run(0);
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CsrRowTask<T: NumericElement, I: CsrIndex, A: CsrArithmetic<T>> {
    values: *const T,
    columns: *const I,
    offsets: *const I,
    rhs: *const T,
    output: *mut T,
    width: usize,
    arithmetic: PhantomData<A>,
}

// SAFETY: private construction retains shared initialized inputs and exclusive
// output in the dispatch scope. T/I are Send+Sync. The unsafe invocation contract
// restricts each worker to one disjoint row and joins before any borrow ends.
unsafe impl<T: NumericElement, I: CsrIndex, A: CsrArithmetic<T>> Send for CsrRowTask<T, I, A> {}
// SAFETY: sharing the descriptor grants only unsafe row invocation. The same
// scope, unique-row and input-immutability obligations apply to all its copies.
unsafe impl<T: NumericElement, I: CsrIndex, A: CsrArithmetic<T>> Sync for CsrRowTask<T, I, A> {}

impl<T: NumericElement, I: CsrIndex, A: CsrArithmetic<T>> CsrRowTask<T, I, A> {
    #[inline]
    pub(super) fn new<O: CsrOrdering>(
        product: &CsrProduct<'_, T, I, O, A>,
        output: *mut T,
    ) -> Self {
        let (values, columns, offsets) = product.view.as_parts();
        Self {
            values: values.as_ptr(),
            columns: columns.as_ptr(),
            offsets: offsets.as_ptr(),
            rhs: product.rhs.as_ptr(),
            output,
            width: product.width,
            arithmetic: PhantomData,
        }
    }

    /// Overwrite and accumulate one complete output row in stored entry order.
    ///
    /// # Safety
    /// Invoke only during the creating dispatch scope, with `row < m`. Invoke
    /// each row at most once and never alias its output with another access.
    /// All scope obligations of [`super::CsrOutput::dispatch_with`] apply.
    ///
    /// # Proof
    /// Construction checks `m*n`, `k*n`, and exact slice lengths. Validated
    /// offsets give `0 <= start <= end <= nnz`; each column is below `k` and
    /// decodes losslessly. Thus all input reads are in bounds. Distinct rows
    /// address disjoint `[r*n,(r+1)*n)` intervals. Initialization establishes
    /// `s_0=ZERO`; each subsequent row update establishes the stored-order
    /// recurrence `s_(p+1)=s_p + values[p]*rhs[column[p],j]` in the chosen
    /// arithmetic policy. Width zero forms no references and performs no reads.
    #[inline]
    pub unsafe fn run(self, row: usize) {
        if self.width == 0 {
            return;
        }
        // SAFETY: the invocation and construction invariants above prove every
        // pointer range. No reference to the complete output is reconstructed.
        unsafe {
            let output =
                core::slice::from_raw_parts_mut(self.output.add(row * self.width), self.width);
            output.fill(T::ZERO);
            let start = address(self.offsets.add(row).read());
            let end = address(self.offsets.add(row + 1).read());
            let values = core::slice::from_raw_parts(self.values.add(start), end - start);
            let columns = core::slice::from_raw_parts(self.columns.add(start), end - start);
            for (&value, &column) in values.iter().zip(columns) {
                let rhs = core::slice::from_raw_parts(
                    self.rhs.add(address(column) * self.width),
                    self.width,
                );
                accumulate::<T, A>(value, rhs, output);
            }
        }
    }
}
