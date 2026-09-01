//! Warm allocation census for complex matrix-batch layout movement.

use core::alloc::{GlobalAlloc, Layout};
use core::cell::Cell;
use leto::Complex;
use leto_ops::transpose_complex_matrices;

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
        unsafe { std::alloc::System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        unsafe { std::alloc::System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = REALLOCATIONS.try_with(|count| count.set(count.get() + 1));
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

fn expected<T: Copy + Default>(
    source: &[Complex<T>],
    matrix_count: usize,
    rows: usize,
    columns: usize,
) -> Vec<Complex<T>> {
    let matrix_len = rows * columns;
    let mut output = vec![Complex::default(); source.len()];
    for matrix in 0..matrix_count {
        let base = matrix * matrix_len;
        for row in 0..rows {
            for column in 0..columns {
                output[base + column * rows + row] = source[base + row * columns + column];
            }
        }
    }
    output
}

#[test]
fn warmed_selected_batches_allocate_nothing() {
    const MATRICES: usize = 256;
    const ROWS: usize = 15;
    const COLUMNS: usize = 13;
    const LEN: usize = MATRICES * ROWS * COLUMNS;

    let source_f32 = (0..LEN)
        .map(|index| Complex::new(index as f32 + 0.25, -(index as f32) - 0.5))
        .collect::<Vec<_>>();
    let expected_f32 = expected(&source_f32, MATRICES, ROWS, COLUMNS);
    let mut destination_f32 = vec![Complex::default(); LEN];
    transpose_complex_matrices(&source_f32, &mut destination_f32, MATRICES, ROWS, COLUMNS)
        .expect("warm f32 transpose succeeds");
    let (result, allocations, reallocations) = measured(|| {
        transpose_complex_matrices(&source_f32, &mut destination_f32, MATRICES, ROWS, COLUMNS)
    });
    result.expect("measured f32 transpose succeeds");
    assert_eq!((allocations, reallocations), (0, 0));
    assert_eq!(destination_f32, expected_f32);

    let source_f64 = (0..LEN)
        .map(|index| Complex::new(index as f64 + 0.25, -(index as f64) - 0.5))
        .collect::<Vec<_>>();
    let expected_f64 = expected(&source_f64, MATRICES, ROWS, COLUMNS);
    let mut destination_f64 = vec![Complex::default(); LEN];
    transpose_complex_matrices(&source_f64, &mut destination_f64, MATRICES, ROWS, COLUMNS)
        .expect("warm f64 transpose succeeds");
    let (result, allocations, reallocations) = measured(|| {
        transpose_complex_matrices(&source_f64, &mut destination_f64, MATRICES, ROWS, COLUMNS)
    });
    result.expect("measured f64 transpose succeeds");
    assert_eq!((allocations, reallocations), (0, 0));
    assert_eq!(destination_f64, expected_f64);
}
