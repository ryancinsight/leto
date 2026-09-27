use super::dispatch::route_matmul;
use super::packing::copy_back_to_out;
use super::types::MATMUL_ROW_BLOCK;
use crate::domain::scalar::Scalar;
use crate::infrastructure::cache::MatmulTilePolicy;
use leto::{Array, ArrayView, ArrayViewMut, Layout, LetoError, Result};

/// Perform matrix multiplication `out = lhs * rhs` for 2D views.
pub fn matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
) -> Result<()> {
    matmul_with_tile_policy(
        lhs,
        rhs,
        out,
        MatmulTilePolicy::fixed(MATMUL_ROW_BLOCK)
            .expect("the measured default row block is supported"),
    )
}

/// Perform matrix multiplication with an explicit bounded row-tile policy.
///
/// This is primarily useful for controlled provider benchmarks and callers
/// that already own a topology policy. Normal callers should use [`matmul`],
/// which retains the measured fixed 32-row production policy.
pub fn matmul_with_tile_policy<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
    tile_policy: MatmulTilePolicy,
) -> Result<()> {
    // Dense-but-offset outputs (a batched/sliced sub-view) route in place: the
    // kernels write through the layout offset, so only a genuinely strided
    // output needs the scratch + copy-back.
    if out.is_c_dense() {
        let mut out_view = out.reborrow();
        route_matmul(lhs, rhs, &mut out_view, false, tile_policy)
    } else if out.is_f_dense() {
        let lhs_t = lhs.transpose([1, 0])?;
        let rhs_t = rhs.transpose([1, 0])?;
        let mut out_t = out.reborrow().transpose_mut([1, 0])?;
        route_matmul(&rhs_t, &lhs_t, &mut out_t, false, tile_policy)
    } else {
        let mut out_contig = Array::from_elem(out.shape(), T::ZERO);
        let mut out_view = out_contig.view_mut();
        // The scratch is already zeroed; accumulating into it skips the
        // kernel's own zero pass, so the output is written once, not twice.
        route_matmul(lhs, rhs, &mut out_view, true, tile_policy)?;
        copy_back_to_out(&out_view, out)?;
        Ok(())
    }
}

/// Perform accumulating matrix multiplication `out += lhs * rhs` for 2D views.
pub fn matmul_accumulate<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
) -> Result<()> {
    // Dense-but-offset outputs route in place (kernels honor the layout offset);
    // only a genuinely strided output needs the scratch + copy-back.
    if out.is_c_dense() {
        let mut out_view = out.reborrow();
        route_matmul(
            lhs,
            rhs,
            &mut out_view,
            true,
            MatmulTilePolicy::fixed(MATMUL_ROW_BLOCK)
                .expect("the measured default row block is supported"),
        )
    } else if out.is_f_dense() {
        let lhs_t = lhs.transpose([1, 0])?;
        let rhs_t = rhs.transpose([1, 0])?;
        let mut out_t = out.reborrow().transpose_mut([1, 0])?;
        route_matmul(
            &rhs_t,
            &lhs_t,
            &mut out_t,
            true,
            MatmulTilePolicy::fixed(MATMUL_ROW_BLOCK)
                .expect("the measured default row block is supported"),
        )
    } else {
        let mut out_contig = out.to_contiguous();
        let mut out_view = out_contig.view_mut();
        route_matmul(
            lhs,
            rhs,
            &mut out_view,
            true,
            MatmulTilePolicy::fixed(MATMUL_ROW_BLOCK)
                .expect("the measured default row block is supported"),
        )?;
        copy_back_to_out(&out_view, out)?;
        Ok(())
    }
}

/// Perform batched matrix multiplication `out[i] = lhs[i] * rhs[i]` for rank-3
/// views shaped `[B, M, K] x [B, K, N] -> [B, M, N]`.
///
/// The batch dimension of either input may be `1`, in which case that operand
/// is broadcast across all `B` batches at zero stride (no materialization).
/// Each batch slice is dispatched to the rank-2 [`matmul`] kernel, so there is
/// one authoritative contraction implementation; this function only resolves
/// per-batch 2D layouts.
pub fn batched_matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 3>,
    rhs: &ArrayView<'_, T, 3>,
    out: &mut ArrayViewMut<'_, T, 3>,
) -> Result<()> {
    let [lhs_batch, m, lhs_k] = lhs.shape();
    let [rhs_batch, rhs_k, n] = rhs.shape();
    let [out_batch, out_m, out_n] = out.shape();

    let batch = out_batch;
    let lhs_batches_ok = lhs_batch == batch || lhs_batch == 1;
    let rhs_batches_ok = rhs_batch == batch || rhs_batch == 1;
    if !lhs_batches_ok || !rhs_batches_ok || lhs_k != rhs_k || m != out_m || n != out_n {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }

    lhs.layout().validate_storage_len(lhs.data().len())?;
    rhs.layout().validate_storage_len(rhs.data().len())?;
    out.layout().validate_storage_len(out.data().len())?;

    let lhs_batch_stride = if lhs_batch == 1 { 0 } else { lhs.strides()[0] };
    let rhs_batch_stride = if rhs_batch == 1 { 0 } else { rhs.strides()[0] };
    let out_batch_stride = out.strides()[0];

    let lhs_mat = |b: usize| {
        Layout::try_new(
            [m, lhs_k],
            [lhs.strides()[1], lhs.strides()[2]],
            (lhs.offset() as isize + b as isize * lhs_batch_stride) as usize,
        )
        .expect("invariant: batch submatrix layout derives from a validated parent")
    };
    let rhs_mat = |b: usize| {
        Layout::try_new(
            [rhs_k, n],
            [rhs.strides()[1], rhs.strides()[2]],
            (rhs.offset() as isize + b as isize * rhs_batch_stride) as usize,
        )
        .expect("invariant: batch submatrix layout derives from a validated parent")
    };
    let out_offset = out.offset() as isize;
    let out_strides = [out.strides()[1], out.strides()[2]];

    #[cfg(feature = "parallel")]
    {
        // The parallel path hands each task a `&mut` over only its own batch's
        // physical footprint, so concurrent tasks never hold overlapping `&mut`
        // slices. (Forming N `&mut` over the same full buffer is UB under
        // Stacked/Tree Borrows even when the writes are physically disjoint.)
        // This requires the per-batch footprints to be physically disjoint;
        // they are unless the batch stride is smaller than one matrix's physical
        // span (an interleaved-batch output view), in which case we fall through
        // to the sequential loop below, which reborrows one batch at a time and
        // is unconditionally sound.
        let out_span = 1
            + out_m.saturating_sub(1) * out_strides[0].unsigned_abs()
            + out_n.saturating_sub(1) * out_strides[1].unsigned_abs();
        let batches_disjoint = out_batch_stride.unsigned_abs() >= out_span;
        // Non-empty guard: an empty output matrix has no work and would make the
        // per-batch `min_max_offsets` on a zero-shape layout degenerate, so let
        // the sequential loop handle it.
        if batch > 1 && batches_disjoint && out_m > 0 && out_n > 0 {
            // Hot-path early-out uses a relaxed atomic flag rather than locking a
            // mutex on every batch index; the mutex is acquired only on the
            // (pre-validated, effectively unreachable) error path to record the
            // first failure. The `for_each_index_with` join supplies the
            // happens-before barrier for the recorded error.
            let had_error = std::sync::Arc::new(core::sync::atomic::AtomicBool::new(false));
            let error_slot = std::sync::Arc::new(std::sync::Mutex::new(None::<LetoError>));
            let had_error_w = had_error.clone();
            let error_slot_w = error_slot.clone();
            let lhs_ptr = lhs.data().as_ptr() as usize;
            let rhs_ptr = rhs.data().as_ptr() as usize;
            let out_ptr = out.data_mut().as_mut_ptr() as usize;
            let lhs_len = lhs.data().len();
            let rhs_len = rhs.data().len();

            let lhs_offset = lhs.offset() as isize;
            let rhs_offset = rhs.offset() as isize;
            let lhs_strides = [lhs.strides()[1], lhs.strides()[2]];
            let rhs_strides = [rhs.strides()[1], rhs.strides()[2]];

            moirai::for_each_index_with::<moirai::Adaptive, _>(batch, move |b| {
                if had_error_w.load(core::sync::atomic::Ordering::Relaxed) {
                    return;
                }

                let lhs_ptr = lhs_ptr as *const T;
                let rhs_ptr = rhs_ptr as *const T;
                let out_ptr = out_ptr as *mut T;

                let lhs_layout = Layout::try_new(
                    [m, lhs_k],
                    lhs_strides,
                    (lhs_offset + b as isize * lhs_batch_stride) as usize,
                )
                .expect("invariant: batch submatrix layout derives from a validated parent");
                let rhs_layout = Layout::try_new(
                    [rhs_k, n],
                    rhs_strides,
                    (rhs_offset + b as isize * rhs_batch_stride) as usize,
                )
                .expect("invariant: batch submatrix layout derives from a validated parent");

                let lhs_view = unsafe {
                    ArrayView::new(lhs_layout, core::slice::from_raw_parts(lhs_ptr, lhs_len))
                };
                let rhs_view = unsafe {
                    ArrayView::new(rhs_layout, core::slice::from_raw_parts(rhs_ptr, rhs_len))
                };
                let abs_offset = (out_offset + b as isize * out_batch_stride) as usize;
                let out_layout = Layout::try_new([out_m, out_n], out_strides, abs_offset)
                    .expect("invariant: batch submatrix layout derives from a validated parent");
                // Borrow only this batch's physical span `[lo, hi]` and rebase
                // the offset into it. `batches_disjoint` (checked above)
                // guarantees these per-batch slices never overlap across tasks,
                // so no two concurrent `&mut` alias.
                let (lo, hi) = out_layout.min_max_offsets();
                let mut out_view = unsafe {
                    ArrayViewMut::new(
                        Layout::try_new([out_m, out_n], out_strides, abs_offset - lo).expect(
                            "invariant: batch submatrix layout derives from a validated parent",
                        ),
                        core::slice::from_raw_parts_mut(out_ptr.add(lo), hi - lo + 1),
                    )
                };

                if let Err(e) = matmul(&lhs_view, &rhs_view, &mut out_view) {
                    let mut slot = error_slot_w.lock().expect("mutex not poisoned");
                    if slot.is_none() {
                        *slot = Some(e);
                    }
                    had_error_w.store(true, core::sync::atomic::Ordering::Relaxed);
                }
            });

            if let Some(e) = error_slot.lock().expect("mutex not poisoned").take() {
                return Err(e);
            }
            return Ok(());
        }
    }

    for b in 0..batch {
        let lhs_view = ArrayView::new(lhs_mat(b), lhs.data());
        let rhs_view = ArrayView::new(rhs_mat(b), rhs.data());
        let out_layout = Layout::try_new(
            [out_m, out_n],
            out_strides,
            (out_offset + b as isize * out_batch_stride) as usize,
        )
        .expect("invariant: batch submatrix layout derives from a validated parent");
        let mut out_view = ArrayViewMut::new(out_layout, out.data_mut());
        matmul(&lhs_view, &rhs_view, &mut out_view)?;
    }

    Ok(())
}
