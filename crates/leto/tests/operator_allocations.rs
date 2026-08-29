//! Allocation-count oracle for the owned-lhs elementwise operators (ADR 0004).
//!
//! This harness is separate from `core_tests` because it installs a counting
//! global allocator, which must not perturb the shared test binary. The counts
//! are the machine-independent acceptance signal for operator buffer reuse: an
//! n-term chain allocates once, not n-1 times.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use core::alloc::{GlobalAlloc, Layout};
use core::cell::Cell;
use leto::{Array2, Storage};

// Per-thread so that concurrently running tests cannot contaminate one
// another's counts. Const-initialized: the slot needs no lazy allocation, which
// would re-enter the allocator being instrumented.
thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

#[inline]
fn record_allocation() {
    let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
}

struct CountingAllocator;

// SAFETY: every method forwards to the system allocator with the caller's
// unmodified pointer and layout, so the allocation contract is the system
// allocator's. The only added effect is a `Cell` increment on a const-init
// thread-local, which allocates nothing and cannot re-enter this allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { std::alloc::System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { std::alloc::System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Allocations performed on this thread while `body` runs. Each test builds its
/// inputs before the measured region, so only the operator chain is counted.
fn allocations_during<R>(body: impl FnOnce() -> R) -> (R, usize) {
    let start = ALLOCATIONS.with(Cell::get);
    let value = body();
    let end = ALLOCATIONS.with(Cell::get);
    (value, end - start)
}

fn operand(bias: f64) -> Array2<f64> {
    let values: Vec<f64> = (0..4096).map(|i| i as f64 * 0.5 + bias).collect();
    Array2::from_shape_vec([64, 64], values).unwrap()
}

#[test]
fn three_term_chain_allocates_once() {
    let a = operand(1.0);
    let b = operand(2.0);
    let c = operand(3.0);

    let (sum, allocations) = allocations_during(|| &a + &b + &c);

    assert_eq!(
        allocations, 1,
        "a 3-term chain must allocate exactly one output array"
    );
    let expected: Vec<f64> = a
        .storage()
        .as_slice()
        .iter()
        .zip(b.storage().as_slice())
        .zip(c.storage().as_slice())
        .map(|((x, y), z)| x + y + z)
        .collect();
    assert_eq!(sum.storage().as_slice(), &expected);
}

#[test]
fn five_term_chain_allocates_once() {
    let a = operand(1.0);
    let b = operand(2.0);
    let c = operand(3.0);
    let d = operand(4.0);
    let e = operand(5.0);

    let (sum, allocations) = allocations_during(|| &a + &b + &c + &d + &e);

    assert_eq!(
        allocations, 1,
        "chain allocation count must not grow with the term count"
    );
    let expected: Vec<f64> = (0..a.len())
        .map(|i| {
            a.storage().as_slice()[i]
                + b.storage().as_slice()[i]
                + c.storage().as_slice()[i]
                + d.storage().as_slice()[i]
                + e.storage().as_slice()[i]
        })
        .collect();
    assert_eq!(sum.storage().as_slice(), &expected);
}

#[test]
fn borrowed_lhs_chain_still_allocates_per_term() {
    // The borrowed tier is unchanged: re-borrowing each intermediate keeps the
    // original n-1 allocations. This pins the contrast the owned form removes.
    let a = operand(1.0);
    let b = operand(2.0);
    let c = operand(3.0);

    let (_, allocations) = allocations_during(|| &(&a + &b) + &c);

    assert_eq!(
        allocations, 2,
        "borrowed-lhs chain allocates one array per binary operator"
    );
}

#[test]
fn owned_scalar_and_neg_allocate_nothing() {
    let a = operand(1.0);

    let (scaled, allocations) = allocations_during(|| -(a * 2.0 + 1.0));

    assert_eq!(
        allocations, 0,
        "owned scalar operators and negation reuse the operand allocation"
    );
    assert!(scaled.storage().as_slice().iter().all(|v| *v < 0.0));
}
