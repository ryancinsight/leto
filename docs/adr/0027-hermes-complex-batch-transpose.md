# ADR 0027: Hermes complex matrix-batch transpose

- Status: Accepted
- Date: 2026-09-01
- Class: [minor, perf]

## Context

Apollo's retained 3-D fast Fourier transform (FFT) executes layout phases over
hundreds of adjacent small complex matrices. Its pinned phase probe showed
that register-resident square transposes reduce this workload, while applying
the same route broadly regressed large rectangular 2-D matrices by 5-52%.

Apollo ADR 0040 assigns layout movement to Leto. Leto core also deliberately
owns only layout and storage types; SIMD kernels belong in `leto-ops`, which
already depends on Hermes. An Apollo-local kernel or a new Hermes dependency in
Leto core would violate those boundaries.

## Decision

Add `leto_ops::transpose_complex_matrices`, a scalar-generic operation over
borrowed `Complex<T>` slices and caller-owned output. It validates checked
matrix and batch lengths plus both exact slice lengths before mutation. Empty
batches and zero-sized matrices are no-ops.

For at least 256 matrices with both sides at most 16, the operation requests
exact Hermes scalar widths 16, 8, then 4. The widest available width whose
complex register fits a complete square tile is selected once at the operation
boundary. Each kernel loads a square tile into registers, transposes it with
`ComplexReg::transpose_square`, stores it once, and copies every ragged row or
column tail. No scalar fallback is classified as SIMD capability.

All other shapes and unsupported exact widths retain Leto's existing generic
tiled assignment. Both paths preserve matrix order and allocate no storage.

Rejected alternatives:

- Keep the register kernel in Apollo. Rejected because layout movement is a
  Leto responsibility and a second implementation would duplicate the
  provider contract.
- Add Hermes to Leto core. Rejected because it would collapse the established
  storage/compute dependency boundary.
- Route every shape through register tiles. Rejected because the Apollo phase
  probe measured 5-52% regressions for large rectangular 2-D matrices.
- Request only the host's widest width. Rejected because an AVX-512 host may
  need an exact AVX2-sized 4x4 tile; exact-width descent preserves that route.

## Consequences

This is one additive public function in `leto-ops`; existing assignment APIs
and behavior do not change. Full, ragged, asymmetric, empty, invalid-length,
and overflow cases carry value-semantic coverage for `f32` and `f64`.
Validation is failure-atomic, and a warmed allocator census records zero
allocations and zero reallocations.

Two independently launched same-binary Criterion runs on the local Windows
AVX2 workstation place provider median reductions at 86.7-88.8% (`f32`) and
88.9-89.8% (`f64`) for 1,024 batches of 4x4 matrices, and 28.3-53.3% (`f32`)
and 26.1-30.5% (`f64`) for 256 batches of 16x16 matrices. Every provider/control
95% confidence-interval pair in the second run is disjoint. The control is
Leto's unchanged generic assignment in the same benchmark binary. These
measurements establish only the selected local layout regime; Apollo must
independently verify full FFT values, allocation behavior, and throughput.
