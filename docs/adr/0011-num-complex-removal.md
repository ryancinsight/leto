# ADR 0011: Atlas-native `Complex<T>`, removing the `num-complex` dependency

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

Introduce `leto::Complex<T>` (`crates/leto/src/domain/complex.rs`): a
`#[repr(C)] #[derive(Clone, Copy, Debug, Default, PartialEq)]` pair of `re`/`im`
with `const fn new`, `bytemuck::Pod`/`Zeroable` (gated on `T: Pod`/`Zeroable`),
the four field-wise/algebraic `Add`/`Sub`/`Mul`/`Div`/`Neg` operators, and
`Display`. It is a faithful drop-in for the `num_complex::Complex` surface leto
and hephaestus actually use.

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

- **Re-export coeus-core's `Complex`**: rejected — coeus is *downstream* of leto;
  the dependency direction forbids it. leto owns the vocabulary; coeus-core can
  later re-export `leto::Complex` to deduplicate (follow-up).
- **Keep `num-complex` as a production dep**: rejected — it is the explicit
  removal target and the type is trivial to own natively.

## Consequences

- leto's production graph is `num-complex`-free; hephaestus production likewise.
- Verified: `leto::Complex` unit tests (arithmetic/layout); 213 leto-ops tests
  (eigenvalues + schur differential vs nalgebra) green; hephaestus wgpu
  eigenvalue contract tests green on real GPU; cuda eigenvalue contract test
  green on real GPU (leto::Complex round-trips a device buffer). All libs,
  tests, and benches across both repos compile.
- Follow-up (filed): coeus-core may re-export `leto::Complex` to collapse the two
  native complex types into one authoritative vocabulary entry.
