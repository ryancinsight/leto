# ADR 0018: Adopt the typed Cartesian 3-D finite-difference provider in leto-ops

- Status: Accepted (partial SSOT — central half migrated; staggered half pending tracker item below).
- Date: 2026-07-26
- Class: [arch] [ssot]

## Context

`kwavers-math`'s `numerics::operators::differential` module carried
copy-by-copy 3-D central-difference kernels (`CentralDifference2`,
`CentralDifference4`, `CentralDifference6`) and a Moirai-backed
`traversal::row_major_index` helper, plus a Yee staggered-grid
operator (`StaggeredGridOperator`). Each integral style duplicated
Laplacian-and-interpolation patterns already owned by leto-ops at
the bottom of the stack.

`kwavers-solver` wrapped the three central operators in a
thin-dispatch enum (`CentralDifferenceOperator`) to feed
`GenericFdtdSolver::update_velocity` / `update_pressure`. The three
operator structs each pinned `<T = f64>`, broke the generic-scalar
contract leto-ops already enforces (cf. ADR 0016 with the typed
2-D `Laplacian2D<T>`), and re-derived the same stencils with
identical shape contracts.

`apollo` already established the precedent: ADR 0017 retired the
last `ndarray` shim by moving the array-press surface into leto and
deleting the consumer copy. The kwavers 3-D FD operators mapped to
exactly the same shape — except `apollo` had at most one consumer-side
duplicate, whereas kwavers carried three structured variants +
staggered + traversal.

## Decision

Adopt `let_ops::FiniteDifference3D<T>` and
`let_ops::FiniteDifference3DScheme` as the provider-SSOT for 3-D first
derivatives. Migrate the central-difference half of
`kwavers-math::numerics::operators::differential` into leto-ops in
this ADR; defer the staggered-grid half to the follow-up tracker
item listed below.

## Contract

- `let_ops::FiniteDifference3D<T: RealField + FloatElement + Copy>` is
  generic in the same way `let_ops::FiniteDifference` and
  `let_ops::laplacian_2d_into` are (ADR 0016 mirror).
- `FiniteDifference3DScheme` enumerates the kernels the FDTD /
  acoustic / CFD / RT integrators actually call:
  `CentralSecondOrder`, `CentralFourthOrder`, `CentralSixthOrder`,
  `StaggeredForward`, `StaggeredBackward`. Stencil widths map to
  3 / 5 / 7 / 2 / 2.
- The `StaggeredBackward` variant mirrors the kwavers-side convention
  (dst shape `== field.shape`, forward fall-back at `i=0`) verbatim
  to preserve the kwavers-side tests bit-equivalent. The Yee-face
  forward form (`StaggeredForward`) shrinks the differentiated axis
  by one cell.
- `new(scheme, dx, dy, dz)` validates strictly positive spacing via
  `LetoError::InvalidInput`. Construction helpers
  `central_{second,fourth,sixth}_order` and
  `staggered_{forward,backward}` are thin convenience wrappers.
- Kernel methods are inherent (no `DifferentialOperator` trait):
  `apply_x_into`, `apply_y_into`, `apply_z_into` write into a
  caller-supplied `&mut Array3<T>`. Boundary fall-back ordering
  preserved bit-for-bit per kwavers's `central_difference_{2,4,6}.

## Conformance oracle

- Generic `central2_x_of_linear_function_is_exact`,
  `central4_x_of_quadratic_is_exact`,
  `central6_x_of_cubic_is_exact`, `staggered_forward_x_face_centered`,
  `staggered_backward_x_zero_field`, and the
  `dispersion_ordering_central_2_4_6` monotonicity probe in
  `crates/leto-ops/src/application/diff/three_dimensional.rs::tests`.
- The `InfiniteGridPoint` rejection path (`CentralSixthOrder` with
  `nx < 7` → `LetoError::InvalidInput`).
- The spacing-validation path (`dx, dy, dz ≤ 0` → error).
- The `central_difference_2` boundary fall-back parity still matches
  kwavers's `central_difference_2/mod.rs` — same coefficients, same
  one-sided forward/backward at the edges.

## Migration (per-consumer deletion ledger)

### kwavers-side: kreiji-3dfd-kwavers-central-sweep

- **Add** `crates/leto-ops/src/application/diff/three_dimensional.rs`
  with `FiniteDifference3D<T>` + `FiniteDifference3DScheme` enum +
  inline parity tests + boundary fall-back parity preserved from
  `kwavers_math::numerics::operators::differential::central_difference_{2,4,6}`.
- **Re-export** `FiniteDifference3D` and `FiniteDifference3DScheme`
  from `let_ops::{FiniteDifference3D, FiniteDifference3DScheme}`
  (next to the existing `FiniteDifference` / `FiniteDifferenceScheme`
  re-exports).
- **Wrap** `crates/kwavers-solver/src/forward/fdtd/solver/central_diff.rs`'s
  `CentralDifferenceOperator` to instantiate
  `let_ops::FiniteDifference3D<f64>` from a `usize` order at
  construction time. Field shape unchanged (`pub(crate)
  central_operator: CentralDifferenceOperator`) so the
  velocity/pressure updaters don't need to change.
- **Delete**:
  - `crates/kwavers-math/src/numerics/operators/differential/central_difference_2`
  - `crates/kwavers-math/src/numerics/operators/differential/central_difference_4`
  - `crates/kwavers-math/src/numerics/operators/differential/central_difference_6`
  - the four kwavers-side differential tests files that depended on
    the deleted types (`tests/{consistency,boundary,accuracy}.rs`; the
    fourth `conservation.rs` is retained pending the staggered sweep —
    see “Follow-up (staggered half SSOT)” below).
- **Replace** `crates/kwavers-math/src/numerics/operators/differential/mod.rs`
  with a thin shim:
  - `pub use leto_ops::{FiniteDifference3D, FiniteDifference3DScheme};`
  - Keep the `DifferentialOperator` trait and `StaggeredGridOperator`
    re-export pending the staggered half.
- **Update** `crates/kwavers-math/src/numerics/operators/mod.rs` to
  expose the leto-backed types at the math-layer root.
- **Update** `crates/kwavers/benches/critical_path_benchmarks.rs` to
  drive the central-FD benchmarks from
  `FiniteDifference3D::central_{second,fourth,sixth}_order` + the
  `apply_x_into` allocating-once pattern.

### leto-side: provider-SSOT for kwavers/CFDrs/helios first-derivatives

- `crates/leto-ops/src/application/diff/three_dimensional.rs` provides
  the public API. The 1-D `FiniteDifference` (existing) handles
  single-axis problems; the 2-D `laplacian_2d_into` (ADR 0016) handles
  the Laplacian; the new 3-D `FiniteDifference3D` handles the three
  central-difference families + the Yee staggered forward/backward the
  FDTD solvers consume.

## Consequences

- `let_ops` gains a `RealField + FloatElement + Copy`-generic 3-D FD
  provider that mirrors the typed 2-D Laplacian contract (ADR 0016).
- `kwavers_math` no longer owns a 3-D FD kernel implementation; only
  the staggered half remains pending. The math crate's public surface
  re-exports `let_ops` types directly.
- `kwavers_solver`'s `CentralDifferenceOperator` enum is no longer a
  dispatch over three kwavers structs; it is a single-holder for a
  leto-side `FiniteDifference3D<f64>`. Call sites are unchanged.
- `apollo`-style migration reuse: `cfd-math` and `helios-imaging` can
  import `let_ops::FiniteDifference3D` directly without copying any
  kwavers-side kernel.
- A benchmark-shape trade-off: the kwavers-side `CentralDifference2`
  used a `for_each_chunk_mut_enumerated_with::<Adaptive, _, _>`
  standard-layout fast path through `traversal.rs`. The leto-side
  impl uses `zip_many_mut_with` on the leto slice-pair, achieving the same
  density without a separate traversal helper.

## Verification

- `cargo test -p leto-ops --lib` covers the eight parity tests
  inline in the new module (central exact-for-polynomial up to
  cubic; staggered zero-field invariant; dispersion monotonicity
  ordering central2 > central4 > central6 mean absolute error;
  spacing validation; insufficient-grid rejection; stencil_width
  smoke).
- `cargo check -p kwavers` verifies the kwavers-side dereferences,
  including the bench update and the math-crate shim deletions,
  without the kwavers-side central kernels.
- `cargo check -p kwavers-solver` verifies the FDTD construction's
  `CentralDifferenceOperator::new` flow: each spatial-order 2/4/6
  maps to a leto-side `FiniteDifference3D<f64>` scheme; every
  velocity/pressure updater bulk-migration site continues to compile.
- `cargo clippy -p leto-ops --lib --no-deps -- -D warnings` keeps the
  warning-denied Clippy contract intact.

## Follow-up (staggered half SSOT)

- Tracker item: kreiji-3dfd-kwavers-staggered-sweep
- Replace `kwavers_math::numerics::operators::differential::StaggeredGridOperator`
  fields (`forward_*_into`, `backward_*_into`) with two leto-side
  `FiniteDifference3D<f64>` instances (`StaggeredForward`,
  `StaggeredBackward`) inside the FDTD solver struct. The single
  `DifferentialOperator` trait's `apply_x/y/z` allocating variants
  retire; documents need an ADR-0018-amendment that consolidates the
  generic convention.
- Delete `kwavers-math/src/numerics/operators/differential/staggered_grid/`
  directory after the dispatcher & 13 tests in
  `kwavers-math/src/numerics/operators/differential/staggered_grid/tests.rs`
  are re-targeted at the leto types.
- Files that remain on the consumer side pending the staggered half sweep:
  - `crates/kwavers-math/src/numerics/operators/differential/staggered_grid/mod.rs`
  - `crates/kwavers-math/src/numerics/operators/differential/staggered_grid/operator.rs`
  - `crates/kwavers-math/src/numerics/operators/differential/staggered_grid/forward.rs`
  - `crates/kwavers-math/src/numerics/operators/differential/staggered_grid/backward.rs`
  - `crates/kwavers-math/src/numerics/operators/differential/staggered_grid/tests.rs`
  - `crates/kwavers-math/src/numerics/operators/differential/tests/conservation.rs`
    (the only kwavers-side differential/tests/* file that depends on
    `StaggeredGridOperator`, retained for the staggered sweep).
  - `crates/kwavers-math/src/numerics/operators/differential/traversal.rs`
    — compat shim: provides `pub(super) row_major_index` (pure-math row-major
    linear index helper) and `pub(super) #[inline] fn write_standard_layout<F>(dst, value_at)`
    (sequential scalar-loop writer used as the C-contiguous fast path for the
    staggered half). The original `try_fill_standard_layout → bool` always
    returned `false`, defeating the C-contiguous fast path entirely; the
    ADR-0018 cleanup renamed it to `write_standard_layout`, gave it a real
    sequential write body, marked it `#[inline]` so LLVM can hoist bounds and
    fuse the closure with surrounding scalar math, and dropped the dead
    `if { return Ok(()) }` short-circuits in the staggered half. The leto-side
    `zip_many_mut_with` slice-pair path remains as the non-C-contiguous
    fallback.
    The function retires when the staggered half SSOT sweep migrates
    `StaggeredForward` / `StaggeredBackward` to `let_ops::FiniteDifference3D<f64>`.
  - `crates/kwavers-math/src/numerics/operators/differential/mod.rs` retains
    the `DifferentialOperator` trait for `StaggeredGridOperator`'s `impl`.
    `DifferentialOperator::order(&self) -> usize` is required (no default body)
    after the ADR-0018 cleanup — the previous `1` default misclassified central
    schemes; `StaggeredGridOperator` already overrides with `2`, the Yee
    face-center coupling order.
  - `crates/kwavers-math/src/numerics/operators/differential/mod.rs` retains
    the `DifferentialOperator` trait for `StaggeredGridOperator`'s `impl`.

## Bit-equivalence caveat

The kwavers-side kernels used `f64::mul_add(...)` chains to take advantage of
the FMA hardware path. The leto-side impl uses plain `+ - *` chains because
`eunomia::FloatElement` does not expose `mul_add` as a method on `T`. The
interaction at -Copt fuses `f*a + b` into FMA on most targets, but the
producer/consumer are not guaranteed to select the same fused representation.
FDTD regression tolerances should be widened by ≤1 ULP from the previous
baseline; if true bit-equivalence is required, add
`T: num_traits::Float` to the `where T: ...` clause and restore the
`mul_add()` chains.

## Non-goals

- `apollo`/`rtk1` `Helmholtz`-family solvers are out of scope. They
  remain routed via the existing apollo FFT surface.
- WENO / Discontinuous Galerkin / Spectral element methods (all
  marked `[Not yet implemented]` in the kwavers-side
  `DifferentialOperator` doc-comment) stay non-goals. Leto grows them
  only after the 3-D FD + Hermite + frequency-domain providers close.
- Re-naming `DifferentialOperator` is non-goal; the trait is kept as
  a backward-compat shim for the staggered half pending the next SSOT
  move.
