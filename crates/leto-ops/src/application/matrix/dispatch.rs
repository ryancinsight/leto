#[cfg(feature = "parallel")]
use super::dot_outer_kernels::{parallel_dot_matmul, parallel_outer_matmul};
use super::dot_outer_kernels::{serial_dot_matmul, serial_outer_matmul};
#[cfg(feature = "parallel")]
use super::generic_kernel::parallel_matmul;
use super::generic_kernel::serial_matmul;
use super::packing::zero_output;
use super::types::validate_matmul;
#[cfg(feature = "parallel")]
use super::types::MatmulLayout;
use crate::domain::scalar::Scalar;
use crate::infrastructure::cache::MatmulTilePolicy;
#[cfg(feature = "parallel")]
use crate::infrastructure::parallel::{PARALLEL_MIN_MATMUL_MACS, PARALLEL_MIN_MATMUL_ROWS};
use leto::{ArrayView, ArrayViewMut, Result};

/// Perform matrix multiplication `out = lhs * rhs` for 2D views.
///
/// The output is caller-owned. This function automatically detects when `lhs` is
/// sparse (below `SPARSE_DENSITY_THRESHOLD`) and `out` is contiguous, routing to the
/// sparse `spmm` kernel. Otherwise, it executes the optimized dense `matmul` logic,
/// which uses an `i-k-j` loop order for locality, row-blocks dense rows to reuse
/// RHS values, and dispatches parallel tasks when `parallel` is enabled.
#[cfg(feature = "parallel")]
#[inline]
fn is_parallel_beneficial(layout: MatmulLayout) -> bool {
    layout.rows * layout.cols * layout.shared >= PARALLEL_MIN_MATMUL_MACS
        && layout.rows >= PARALLEL_MIN_MATMUL_ROWS
}

pub(super) fn route_matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
    accumulate: bool,
    tile_policy: MatmulTilePolicy,
) -> Result<()> {
    let layout = validate_matmul(lhs, rhs, out)?;
    #[cfg(not(feature = "parallel"))]
    let _ = layout;

    // Fast-path selection uses offset-independent dense predicates: the dot/cc/
    // outer kernels address every operand through its layout's own `offset`, so
    // a batched or sliced sub-view that is dense-but-offset (e.g. batch `b` of a
    // C-contiguous 3-D output, offset `b·m·n`) is served in place with no
    // operand copy and no scratch allocation. Pinning `offset == 0` here
    // (`is_c_contiguous`) was forcing those views down the allocating fallback.
    if lhs.is_c_dense() && rhs.is_f_dense() && out.is_c_dense() {
        #[cfg(feature = "parallel")]
        {
            if is_parallel_beneficial(layout) {
                parallel_dot_matmul(lhs, rhs, out, accumulate);
                return Ok(());
            }
        }
        serial_dot_matmul(lhs, rhs, out, accumulate);
        return Ok(());
    }

    if lhs.is_f_dense() && rhs.is_c_dense() && out.is_c_dense() {
        #[cfg(feature = "parallel")]
        {
            if is_parallel_beneficial(layout) {
                parallel_outer_matmul(lhs, rhs, out, accumulate);
                return Ok(());
            }
        }
        serial_outer_matmul(lhs, rhs, out, accumulate);
        return Ok(());
    }

    // Fallback: copy only genuinely non-dense operands to contiguous. A
    // dense-but-offset operand is kept in place (the generic kernel addresses it
    // through its layout offset), so only strided/broadcast operands pay a copy.
    let lhs_contig;
    let lhs_view = if lhs.is_c_dense() {
        lhs.reborrow()
    } else {
        lhs_contig = lhs.to_contiguous();
        lhs_contig.view()
    };

    let rhs_contig;
    let rhs_view = if rhs.is_c_dense() {
        rhs.reborrow()
    } else {
        rhs_contig = rhs.to_contiguous();
        rhs_contig.view()
    };

    let layout = validate_matmul(&lhs_view, &rhs_view, out)?;
    if !accumulate {
        zero_output(layout, out);
    }

    #[cfg(feature = "parallel")]
    {
        if is_parallel_beneficial(layout) {
            parallel_matmul(&lhs_view, &rhs_view, out, layout, tile_policy);
            return Ok(());
        }
    }
    serial_matmul(&lhs_view, &rhs_view, out, layout, tile_policy);
    Ok(())
}
