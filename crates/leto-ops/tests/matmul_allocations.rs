//! Allocation census for dense `matmul` layout dispatch.
//!
//! An FxF product with a C-dense output needs only one operand in C order:
//! copying the lhs yields the (C, F, C) dot kernel and copying the rhs yields
//! the (F, C, C) outer kernel. The dispatch must therefore materialize
//! exactly one operand — the smaller one — per call.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use core::alloc::{GlobalAlloc, Layout};
use core::cell::Cell;
use leto::{Array, Layout as ArrayLayout, Storage, VecStorage};
use leto_ops::matmul;

thread_local! {
    static LARGE_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static LARGE_BYTES: Cell<usize> = const { Cell::new(0) };
}

/// Operand copies at the census size are 512 KiB; scheduler/task bookkeeping
/// allocations are orders of magnitude smaller. Counting only allocations at
/// or above 64 KiB isolates operand materializations on the calling thread.
const LARGE_THRESHOLD: usize = 64 * 1024;

struct CountingAllocator;

// SAFETY: every operation forwards the unchanged pointer and layout to the
// system allocator. Const-initialized thread-local counters add no allocation
// and isolate the measured test thread from the rest of the test process.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= LARGE_THRESHOLD {
            let _ = LARGE_ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
            let _ = LARGE_BYTES.try_with(|bytes| bytes.set(bytes.get() + layout.size()));
        }
        // SAFETY: GlobalAlloc's caller supplies the valid, unchanged layout.
        unsafe { std::alloc::System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the pointer came from System and retains its allocation layout.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= LARGE_THRESHOLD {
            let _ = LARGE_ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
            let _ = LARGE_BYTES.try_with(|bytes| bytes.set(bytes.get() + layout.size()));
        }
        // SAFETY: GlobalAlloc's caller supplies the valid, unchanged layout.
        unsafe { std::alloc::System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measured<R>(body: impl FnOnce() -> R) -> (R, usize, usize) {
    let allocations_before = LARGE_ALLOCATIONS.with(Cell::get);
    let bytes_before = LARGE_BYTES.with(Cell::get);
    let output = body();
    let allocations = LARGE_ALLOCATIONS.with(Cell::get) - allocations_before;
    let bytes = LARGE_BYTES.with(Cell::get) - bytes_before;
    (output, allocations, bytes)
}

/// Logical element `(i, j)` shared by both census operands; every value is
/// nonzero so the dense route (not the sparse shortcut) is exercised.
fn element(i: usize, j: usize) -> f64 {
    f64::from(u32::try_from((i + 2 * j) % 5).unwrap()) + 1.0
}

fn f_dense(side: usize) -> Array<f64, VecStorage<f64>, 2> {
    // F-contiguous storage holds column `j` at `j * side .. (j + 1) * side`.
    let storage: Vec<f64> = (0..side)
        .flat_map(|j| (0..side).map(move |i| element(i, j)))
        .collect();
    Array::new(
        ArrayLayout::f_contiguous([side, side]).unwrap(),
        VecStorage::new(storage),
    )
    .unwrap()
}

#[test]
fn fxf_matmul_copies_exactly_one_operand() {
    const SIDE: usize = 256;
    let lhs = f_dense(SIDE);
    let rhs = f_dense(SIDE);
    let mut out = Array::zeros([SIDE, SIDE]);
    assert!(lhs.view().is_f_dense() && rhs.view().is_f_dense());

    // Warm call: absorbs one-time parallel-pool setup.
    matmul(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();

    let (result, allocations, bytes) =
        measured(|| matmul(&lhs.view(), &rhs.view(), &mut out.view_mut()));
    result.unwrap();
    assert_eq!(allocations, 1, "FxF matmul must copy one operand only");
    assert_eq!(bytes, SIDE * SIDE * core::mem::size_of::<f64>());

    // Exact corner value: out[0, 0] = sum_k lhs[0, k] * rhs[k, 0].
    let expected: f64 = (0..SIDE).map(|k| element(0, k) * element(k, 0)).sum();
    assert_eq!(out.storage().as_slice()[0], expected);
}
