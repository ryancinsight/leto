# ADR 0004: Elementwise operator overloading on `Array`

- Status: Accepted
- Date: 2026-06-15
- Class: [minor] (additive surface). Supersedes the deferral in ADR 0001.

## Context

ADR 0001 deferred std operator overloading on `Array` because the obvious
placement was blocked: the elementwise SIMD/parallel kernels live in `leto-ops`,
but the orphan rule means only `leto` core may implement `core::ops::Add` etc.
for `leto::Array`, and core is deliberately `hermes`/`moirai`-free. ADR 0001's
concern was that operators in core would either duplicate the elementwise loop or
split one operation family across two crates with divergent performance.

Two facts make the decision now clean:

1. Core **already** carries a scalar/`num-traits` reduction tier (`sum_all`,
   `mean_all`, `argmin`, …) that coexists with leto-ops' SIMD reduction tier
   (`sum`, `sum_axis`). So "scalar convenience tier in core + accelerated
   caller-owned-output tier in leto-ops" is an *established* pattern here, not a
   new SSOT violation. Operators join that tier.
2. Core already has a single logical-order element iterator (`iter_elements`)
   and the public `Array::from_shape_vec` constructor, so operators reuse one
   traversal — no second elementwise loop is written.

## Decision

Implement `Add`/`Sub`/`Mul`/`Div`/`Neg` in `leto` core for `&Array` receivers,
as the **allocating convenience tier**:

- `&Array op &Array` (equal shape) and `&Array op scalar`; unary `-&Array`.
- Output is a fresh C-contiguous `Array<T, VecStorage<T>, N>`.
- Two private helpers (`binary_elementwise`, `unary_elementwise`) over
  `iter_elements` are the single core traversal; every operator delegates to one
  of them. The leto-ops `binary_map`/`scalar_map` (SIMD, broadcasting, into
  caller-owned output) remain the performance tier — the same two-tier split as
  reductions.

Key semantic decision — **`*` is elementwise (Hadamard), matching ndarray**, not
matrix multiplication. `Array` is an N-dimensional array (ndarray's model), so
all four operators are elementwise for consistency. Matrix product stays the
explicit `MatrixProduct::matmul` method (ADR 0003); this avoids the
ndarray-vs-nalgebra `*` ambiguity in the consolidated type.

Scalar-rhs operators are bounded by a sealed `ScalarOperand` marker trait
(implemented for the numeric primitives) so `&Array op scalar` does not overlap
`&Array op &Array` under coherence — the ndarray `ScalarOperand` pattern.

Shape handling: operators cannot return `Result`, so an unequal-shape
`&Array op &Array` **panics** with a message naming the violated invariant. This
is the sanctioned operator exception to the library no-panic policy (matches
ndarray and nalgebra, which both panic on incompatible operator shapes); the
fallible, broadcasting path is the leto-ops `binary_map`/`add` family. Scalar
operators and `Neg` never fail.

## Consequences

- `&a + &b`, `&a * s`, `-&a` now work on owned/borrowed arrays; ergonomic parity
  with the common ndarray operator surface. `parity_matrix.md` §A "std::ops
  operator impls" flips from Missing to present (elementwise).
- Operators allocate (convenience tier); hot paths keep using the leto-ops
  caller-owned-output kernels. This performance distinction is documented on the
  operators, addressing ADR 0001's expectation-hazard concern by making the two
  tiers explicit rather than hidden.
- Receiver scope this increment is references (`&Array`); owned (`Array op …`)
  and view (`ArrayView op …`) operator forms are additive follow-ups if a
  consumer needs them.
- No leto-ops kernel is relocated; core gains only the new convenience traversal.
  The ADR 0001 [arch] kernel-relocation option is therefore not taken — it is
  unnecessary under the two-tier model.
