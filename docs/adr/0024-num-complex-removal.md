# ADR 0024: Atlas-native `Complex<T>`, removing the `num-complex` dependency

- Status: Accepted
- Date: 2026-06-29
- Class: [major] (the public `eigenvalues`/`SchurDecomposition::eigenvalues`
  return type changes from `num_complex::Complex<T>` to `leto::Complex<T>`;
  consumed cross-repo by hephaestus)

## Context

`num-complex` was the last third-party numeric dependency in leto's production
graph. It was used solely as a `(re, im)` holder for the non-symmetric
eigensolver spectrum (ADR 0006): `eigenvalues() -> Vec<Complex<T>>`,
`SchurDecomposition::eigenvalues()`, and the `MatrixLinalg::eigenvalues` trait
method. The eigenvalue arithmetic is performed entirely in the real scalar `T`
(the `2×2` block formulas in `eigenvalues/mod.rs`); the complex type never
participates in complex multiplication/division on the production path — it only
carries the result. hephaestus (path dependency) consumes this spectrum into a
`WgpuBuffer<Complex<f32>>`, so the type must be `bytemuck::Pod` and
layout-compatible (`#[repr(C)] { re, im }`).

The atlas goal is to remove third-party deps in favor of atlas-native vocabulary
(coeus-core already defines its own `Complex<T>`). A native leto type closes the
last such dependency for leto + hephaestus.

## Decision

Define the native `Complex<T>` in **`hermes-numeric`**, which declares itself the
Single Source of Truth for numeric representations and already owns the
`F16`/`F32`/`F64`/`Bf16`/`I32` wrapper types. `hermes_numeric::Complex<T>` is a
`#[repr(C)] #[derive(Clone, Copy, Debug, Default, PartialEq)]` pair of `re`/`im`
with `const fn new`, `bytemuck::Pod`/`Zeroable` (gated on `T: Pod`/`Zeroable`),
the four field-wise/algebraic `Add`/`Sub`/`Mul`/`Div`/`Neg` operators, and
`Display` — a faithful drop-in for the `num_complex::Complex` surface.

leto **re-exports** it as `leto::Complex` (`pub use hermes_numeric::Complex`, via a
direct `hermes-numeric` dependency); leto-ops and the hephaestus consumers import
`leto::Complex` unchanged. Placing the type in the numeric foundation keeps one
owned definition for the whole hermes-consuming stack rather than per-crate
copies (SSOT + upstream-ownership).

Migrate every production site (`num_complex::Complex` → `leto::Complex`) in
leto-ops and the hephaestus consumers (wgpu/cuda/metal compute + python
bindings) in one coordinated cross-repo change, since the dependency is by path
(an uncoordinated leto push would break hephaestus immediately).

`num-complex` is retained as a **dev-dependency** in leto-ops and
hephaestus-wgpu/-cuda, because the differential test/bench oracle is nalgebra,
whose `complex_eigenvalues()` returns `num_complex::Complex`. Tests now compare
the `leto::Complex` result against the `num_complex` oracle component-wise on
`re`/`im` (both expose those fields), converting at the oracle boundary where a
collection type is fixed.

## Alternatives rejected

- **Define `Complex` in leto** (initial approach, corrected): rejected — leto is
  not the numeric SSOT; `hermes-numeric` is. A numeric vocabulary type needed by
  leto, hephaestus, and coeus belongs in the deepest common foundation, not a
  mid-layer crate. The leto-local definition was relocated to `hermes-numeric`.
- **Keep `num-complex` as a production dep**: rejected — it is the explicit
  removal target and the type is trivial to own natively.

## Consequences

- The native `Complex<T>` is owned once by `hermes-numeric` (SSOT); leto,
  hephaestus, and coeus consume it via `leto::Complex`. Both leto and hephaestus
  production graphs are `num-complex`-free.
- Verified: `hermes_numeric::Complex` unit tests (arithmetic/fields); 213 leto-ops
  tests (eigenvalues + schur differential vs nalgebra) green through the
  re-export; hephaestus wgpu eigenvalue contract tests green on real GPU; cuda
  eigenvalue contract test green on real GPU (the type round-trips a device
  buffer); bytemuck `Pod` unifies across hermes → leto → hephaestus. All libs,
  tests, and benches across the repos compile.
- Follow-up (filed): coeus-core defines its own `Complex<T>` (with coeus-specific
  `Scalar`/`FloatOps`/`CpuUnaryDispatch` impls) — consolidate it onto
  `hermes_numeric::Complex` by moving those trait impls and re-exporting, closing
  the last duplicate native complex type.
