use super::packing::zero_output;
use super::types::{validate_matmul, MatmulPtrs};
use crate::domain::scalar::Scalar;
use leto::{ArrayView, ArrayViewMut};

#[inline(always)]
unsafe fn dot_matmul_row<T: Scalar>(
    ptrs: MatmulPtrs<T>,
    i: usize,
    k: usize,
    n: usize,
    accumulate: bool,
) {
    let lhs_row = core::slice::from_raw_parts(ptrs.lhs.add(ptrs.lhs_offset + i * k), k);
    for j in 0..n {
        let rhs_col = core::slice::from_raw_parts(ptrs.rhs.add(ptrs.rhs_offset + j * k), k);
        let val = T::dot_slice(lhs_row, rhs_col);
        let out_addr = ptrs.out.add(ptrs.out_offset + i * n + j);
        if accumulate {
            *out_addr = (*out_addr).add(val);
        } else {
            *out_addr = val;
        }
    }
}

pub(super) fn serial_dot_matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
    accumulate: bool,
) {
    let [m, k] = lhs.shape();
    let [_, n] = rhs.shape();
    let lhs_offset = lhs.offset();
    let rhs_offset = rhs.offset();
    let out_offset = out.offset();
    let lhs_data = lhs.data();
    let rhs_data = rhs.data();
    let out_data = out.data_mut();
    let ptrs = MatmulPtrs {
        lhs: lhs_data.as_ptr(),
        rhs: rhs_data.as_ptr(),
        out: out_data.as_mut_ptr(),
        lhs_offset,
        rhs_offset,
        out_offset,
    };

    for i in 0..m {
        // SAFETY: `route_matmul` selects this kernel only for dense row/column
        // layouts; the validated shapes make every row and output address valid.
        unsafe {
            dot_matmul_row(ptrs, i, k, n, accumulate);
        }
    }
}

/// Number of output rows processed per parallel task.
///
/// Each task handles `PARALLEL_ROW_BLOCK` consecutive rows of the output
/// matrix.  Blocking reduces the task-dispatch overhead by `PARALLEL_ROW_BLOCK×`
/// vs one-task-per-row, and ensures each task writes to a contiguous output
/// region of at least `PARALLEL_ROW_BLOCK * n * size_of::<T>()` bytes — large
/// enough to avoid false sharing for any practical `n`.
#[cfg(feature = "parallel")]
const PARALLEL_ROW_BLOCK: usize = 4;

#[cfg(feature = "parallel")]
pub(super) fn parallel_dot_matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
    accumulate: bool,
) {
    let [m, k] = lhs.shape();
    let [_, n] = rhs.shape();
    let lhs_offset = lhs.offset();
    let rhs_offset = rhs.offset();
    let out_offset = out.offset();
    let lhs_ptr = lhs.data().as_ptr() as usize;
    let rhs_ptr = rhs.data().as_ptr() as usize;
    let out_ptr = out.data_mut().as_mut_ptr() as usize;

    // Dispatch in row blocks to amortise per-task scheduling overhead.
    let n_blocks = m.div_ceil(PARALLEL_ROW_BLOCK);
    moirai::for_each_index_with::<moirai::AdaptiveWithThreshold<16>, _>(n_blocks, move |block| {
        let ptrs = MatmulPtrs {
            lhs: lhs_ptr as *const T,
            rhs: rhs_ptr as *const T,
            out: out_ptr as *mut T,
            lhs_offset,
            rhs_offset,
            out_offset,
        };
        let i_start = block * PARALLEL_ROW_BLOCK;
        let i_end = (i_start + PARALLEL_ROW_BLOCK).min(m);
        for i in i_start..i_end {
            unsafe {
                dot_matmul_row(ptrs, i, k, n, accumulate);
            }
        }
    });
}

#[inline(always)]
unsafe fn outer_matmul_row<T: Scalar>(ptrs: MatmulPtrs<T>, i: usize, m: usize, k: usize, n: usize) {
    let out_row = core::slice::from_raw_parts_mut(ptrs.out.add(ptrs.out_offset + i * n), n);
    for kk in 0..k {
        let alpha = *ptrs.lhs.add(ptrs.lhs_offset + kk * m + i);
        if alpha == T::ZERO {
            continue;
        }
        let rhs_row = core::slice::from_raw_parts(ptrs.rhs.add(ptrs.rhs_offset + kk * n), n);
        T::axpy_slice(alpha, rhs_row, out_row);
    }
}

pub(super) fn serial_outer_matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
    accumulate: bool,
) {
    let [m, k] = lhs.shape();
    let [_, n] = rhs.shape();

    if !accumulate {
        zero_output(
            validate_matmul(lhs, rhs, out).expect("route_matmul validated dimensions"),
            out,
        );
    }

    let lhs_offset = lhs.offset();
    let rhs_offset = rhs.offset();
    let out_offset = out.offset();
    let lhs_data = lhs.data();
    let rhs_data = rhs.data();
    let out_data = out.data_mut();
    let ptrs = MatmulPtrs {
        lhs: lhs_data.as_ptr(),
        rhs: rhs_data.as_ptr(),
        out: out_data.as_mut_ptr(),
        lhs_offset,
        rhs_offset,
        out_offset,
    };

    for i in 0..m {
        // SAFETY: `route_matmul` selects this kernel only for dense row/column
        // layouts; the validated shapes make every row and output address valid.
        unsafe {
            outer_matmul_row(ptrs, i, m, k, n);
        }
    }
}

#[cfg(feature = "parallel")]
pub(super) fn parallel_outer_matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
    accumulate: bool,
) {
    let [m, k] = lhs.shape();
    let [_, n] = rhs.shape();

    if !accumulate {
        zero_output(
            validate_matmul(lhs, rhs, out).expect("route_matmul validated dimensions"),
            out,
        );
    }

    let lhs_offset = lhs.offset();
    let rhs_offset = rhs.offset();
    let out_offset = out.offset();
    let lhs_ptr = lhs.data().as_ptr() as usize;
    let rhs_ptr = rhs.data().as_ptr() as usize;
    let out_ptr = out.data_mut().as_mut_ptr() as usize;

    let n_blocks = m.div_ceil(PARALLEL_ROW_BLOCK);
    moirai::for_each_index_with::<moirai::AdaptiveWithThreshold<16>, _>(n_blocks, move |block| {
        let ptrs = MatmulPtrs {
            lhs: lhs_ptr as *const T,
            rhs: rhs_ptr as *const T,
            out: out_ptr as *mut T,
            lhs_offset,
            rhs_offset,
            out_offset,
        };
        let i_start = block * PARALLEL_ROW_BLOCK;
        let i_end = (i_start + PARALLEL_ROW_BLOCK).min(m);
        for i in i_start..i_end {
            unsafe {
                outer_matmul_row(ptrs, i, m, k, n);
            }
        }
    });
}
