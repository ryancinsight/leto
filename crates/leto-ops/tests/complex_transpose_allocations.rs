//! Allocation census for caller-owned complex layout movement.

use core::alloc::{GlobalAlloc, Layout};
use core::cell::Cell;
use eunomia::{Bf16, F16};
use leto_ops::{transpose_complex_matrices, transpose_square_inplace, SquareTransposeError};

#[path = "ops/layout/payloads.rs"]
mod payloads;
use payloads::{assert_bits, expected, values, PayloadScalar};

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static REALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

// SAFETY: every operation forwards the unchanged pointer and layout to the
// system allocator. Const-initialized thread-local counters add no allocation
// and isolate the measured test thread from the rest of the test process.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        // SAFETY: GlobalAlloc's caller supplies the valid, unchanged layout.
        unsafe { std::alloc::System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the pointer came from System and retains its allocation layout.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        // SAFETY: GlobalAlloc's caller supplies the valid, unchanged layout.
        unsafe { std::alloc::System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = REALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        // SAFETY: System owns ptr; its layout and valid new size pass through unchanged.
        unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measured<R>(body: impl FnOnce() -> R) -> (R, usize, usize) {
    let allocations_before = ALLOCATIONS.with(Cell::get);
    let reallocations_before = REALLOCATIONS.with(Cell::get);
    let output = body();
    let allocations = ALLOCATIONS.with(Cell::get) - allocations_before;
    let reallocations = REALLOCATIONS.with(Cell::get) - reallocations_before;
    (output, allocations, reallocations)
}

fn assert_batch_allocations<T: PayloadScalar>() {
    const MATRICES: usize = 256;
    const ROWS: usize = 15;
    const COLUMNS: usize = 13;
    const LEN: usize = MATRICES * ROWS * COLUMNS;

    let source = values::<T>(LEN);
    let oracle = expected(&source, MATRICES, ROWS, COLUMNS);
    let mut destination = source.clone();
    transpose_complex_matrices(&source, &mut destination, MATRICES, ROWS, COLUMNS)
        .expect("warm batch transpose succeeds");
    let (result, allocations, reallocations) =
        measured(|| transpose_complex_matrices(&source, &mut destination, MATRICES, ROWS, COLUMNS));
    result.expect("measured batch transpose succeeds");
    assert_eq!((allocations, reallocations), (0, 0));
    assert_bits(&destination, &oracle);
}

#[test]
fn warmed_selected_batches_allocate_nothing() {
    assert_batch_allocations::<f32>();
    assert_batch_allocations::<f64>();
    assert_batch_allocations::<F16>();
    assert_batch_allocations::<Bf16>();
}

fn assert_square_allocations<T: PayloadScalar>() {
    for side in [0, 1, 3, 8, 17, 256, 512] {
        let original = values::<T>(side * side);
        let oracle = expected(&original, 1, side, side);
        let mut matrix = original.clone();
        let (result, allocations, reallocations) =
            measured(|| transpose_square_inplace(&mut matrix, side));
        result.expect("first square submission succeeds");
        assert_eq!((allocations, reallocations), (0, 0));
        assert_bits(&matrix, &oracle);
        let (result, allocations, reallocations) =
            measured(|| transpose_square_inplace(&mut matrix, side));
        result.expect("repeated square submission succeeds");
        assert_eq!((allocations, reallocations), (0, 0));
        assert_bits(&matrix, &original);
    }

    // The first overflowing side is representable without constructing its
    // impossible matrix. Short and long valid extents exercise both directions.
    let first_overflow = 1usize << (usize::BITS / 2);
    for (side, len) in [
        (16usize, 255),
        (16, 257),
        (first_overflow, 5),
        (usize::MAX, 5),
    ] {
        let oracle = match side.checked_mul(side) {
            Some(expected) => SquareTransposeError::Length {
                side,
                expected,
                actual: len,
            },
            None => SquareTransposeError::Overflow { side },
        };
        let mut storage = values::<T>(len + 4);
        let original = storage.clone();
        for _ in 0..2 {
            let (result, allocations, reallocations) =
                measured(|| transpose_square_inplace(&mut storage[1..=len], side));
            assert_eq!(result, Err(oracle));
            assert_eq!((allocations, reallocations), (0, 0));
            assert_bits(&storage, &original);
        }
    }
}

#[test]
fn square_submissions_allocate_nothing() {
    assert_square_allocations::<f32>();
    assert_square_allocations::<f64>();
    assert_square_allocations::<F16>();
    assert_square_allocations::<Bf16>();
}
