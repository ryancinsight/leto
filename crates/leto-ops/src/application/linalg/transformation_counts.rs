//! Instrumented transformation counts backing the probabilistic error oracle.
//!
//! The a-priori bounds of the test suite (`tests/ops/backward_error.rs`) are
//! taken at iteration caps, which makes them vacuous for the iterative
//! routines in `F16`/`Bf16`. The probabilistic oracle
//! (`tests/ops/backward_error.rs`, `gamma_tilde`) evaluates the same
//! per-entry rounding counts at the *actual* transformation counts instead,
//! which this module records: total Francis steps per [`schur`](crate::application::linalg::schur::schur())
//! / [`eigenvalues`](crate::application::linalg::eigenvalues::eigenvalues()) run and the `dbdsqr` sweeps and
//! negligible-diagonal chases per SVD run (split: chases cost fewer
//! roundings per entry than sweeps).
//!
//! Thread-local `Cell`s: the kernels are single-threaded (SIMD is
//! intra-thread; no `parallel` branch exists on these paths), so the count a
//! test takes after its own call is exactly that call's, under both `cargo
//! test` (threads) and nextest (processes). Cost is one thread-local write
//! per routine call plus one increment per Francis step — nanoseconds
//! against millisecond runs — so recording is unconditional, with no
//! feature gate and no behavior change.
//!
//! Discipline: take immediately after the call whose count is needed; `take`
//! resets to zero, so a take without a preceding call yields `0` (a bound
//! for a run that did nothing — an assert against it fails loudly, never
//! silently).

use core::cell::Cell;

thread_local! {
    static FRANCIS_STEPS: Cell<usize> = const { Cell::new(0) };
    static DBDSQR_SWEEPS: Cell<usize> = const { Cell::new(0) };
    static DBDSQR_CHASES: Cell<usize> = const { Cell::new(0) };
    static QL_SWEEPS: Cell<usize> = const { Cell::new(0) };
}

/// Record the total Francis steps of a completed [`schur`](crate::application::linalg::schur::schur()) /
/// [`eigenvalues`](crate::application::linalg::eigenvalues::eigenvalues()) run (`francis::run`).
pub(crate) fn record_francis_steps(steps: usize) {
    FRANCIS_STEPS.with(|count| count.set(steps));
}

/// Record the `dbdsqr` sweeps and negligible-diagonal chases
/// (`qr_iterate`'s return) of a completed SVD run.
pub(crate) fn record_dbdsqr_counts(sweeps: usize, chases: usize) {
    DBDSQR_SWEEPS.with(|count| count.set(sweeps));
    DBDSQR_CHASES.with(|count| count.set(chases));
}

/// Take the recorded Francis steps since the last take (zero when no
/// [`schur`](crate::application::linalg::schur::schur()) / [`eigenvalues`](crate::application::linalg::eigenvalues::eigenvalues()) ran since).
#[must_use]
pub fn take_francis_steps() -> usize {
    FRANCIS_STEPS.with(|count| count.replace(0))
}

/// Take the recorded `(sweeps, chases)` since the last take (both zero when
/// no SVD ran since).
#[must_use]
pub fn take_dbdsqr_counts() -> (usize, usize) {
    (
        DBDSQR_SWEEPS.with(|count| count.replace(0)),
        DBDSQR_CHASES.with(|count| count.replace(0)),
    )
}

/// Record the implicit-QL sweeps of a completed symmetric eigendecomposition.
pub(crate) fn record_ql_sweeps(sweeps: usize) {
    QL_SWEEPS.with(|count| count.set(sweeps));
}

/// Take the recorded QL sweeps since the last take (zero when no symmetric
/// eigendecomposition ran since).
#[must_use]
pub fn take_ql_sweeps() -> usize {
    QL_SWEEPS.with(|count| count.replace(0))
}
