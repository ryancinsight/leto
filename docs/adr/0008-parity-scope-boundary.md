# ADR 0008: Parity scope boundary — fixed-size storage, compile-time shape, geometry

- Status: Accepted
- Date: 2026-06-15
- Class: [minor] (adds `StackStorage`; records two exclusions with rationale)

## Context

The ndarray/nalgebra parity matrix (`docs/completeness/parity_matrix.md`) reaches
full coverage of the array surface (§A, 28/28) and the dense-decomposition surface
(§B: LU/QR/Cholesky/SVD/eigen/Hessenberg/Schur/bidiagonal/full-piv-LU/col-piv-QR/
UDU/Bunch–Kaufman/trace/rank/Kronecker/matrix-functions). Two rows remained flagged
`Excluded?` pending a decision:

1. **Small fixed `MatrixN` / `VectorN`** (nalgebra `Matrix3`, `Vector3`): stack-
   allocated, with the dimensions encoded in the *type*.
2. **Geometry**: `Rotation`, `Isometry`, `Quaternion`, `Perspective`,
   `Translation`, etc.

Each conflates several distinct capabilities; this ADR separates them and rules on
each against Leto's architecture (ADR 0002 const-rank/runtime-dims; bounded-context
isolation).

## Decision

### 1a. Stack allocation — DELIVERED (`StackStorage<T, const CAP>`)

The genuinely in-scope part of "small fixed matrix" is *allocation-free* backing.
Leto's [`Storage`] trait is a deliberate extension seam: every array operation is
written once, generic over `S: Storage<T>`. A new inline backing therefore inherits
the **entire** operation surface (reductions, arithmetic, iteration, slicing,
transpose, the LA kernels via views) with **zero** duplicated code (SSOT/DIP).

`StackStorage<T, const CAP>` wraps `[T; CAP]`: no heap allocation, `no_std`-friendly,
and `Copy` when `T: Copy`. `Array::from_stack(shape, [T; CAP])` constructs a
stack-backed array, validating `CAP == ∏ shape`. This is the canonical
"add a backend via the trait, not by cloning algorithms" pattern (cf. the
`ComputeBackend`/`GuiBackend` seams).

### 1b. Compile-time fixed *shape* — EXCLUDED (architecture)

Encoding the dimensions in the type (nalgebra's `Matrix3` ⇒ a `3×3` known to the
compiler, with shape-mismatch as a *compile* error) requires **type-level
dimensions**: `Array<T, S, const R, const C, …>`. Leto deliberately encodes const
**rank** with runtime **dimensions** (`Array<T, S, const N>`, dims in the
`Layout`) per ADR 0002 — the basis of its const-rank kernels and the
const-generic dispatch shim at the Coeus boundary. Adding a parallel type-level-
shape array would fork the core type, an [arch] divergence with no consumer driver.

Rationale: leto's value is the strided-array substrate, not a stack-matrix algebra
library. Callers needing compile-time shape checking use `nalgebra` directly (it
remains a dependency of the LA test oracles) or a downstream fixed-shape wrapper;
leto provides the allocation-free *storage* (1a) and runtime-validated shapes.

### 2. Geometry — EXCLUDED (bounded context)

`Rotation`/`Isometry`/`Quaternion`/`Perspective`/`Translation` are **spatial-
transform** types, not an array substrate. They carry their own invariants
(unit-norm quaternions, orthonormal rotation matrices, the projective structure of
a perspective), a different algebra (composition, `slerp`, exp/log on the manifold),
and a different bounded context. Implementing them inside the array crate violates
SRP and bounded-context isolation (a GUI/physics consumer depending on the array
core's geometry would be the inverted dependency the stack forbids).

Rationale: in the Atlas layering, geometry belongs to a **downstream** domain crate
(or `nalgebra` for consumers who want it), built *on* Leto's arrays — not inside
the infrastructure-tier array library. No current consumer drives geometry into
Leto; adding it speculatively is over-engineering (YAGNI governs feature scope, not
the abstraction seams, which are already present).

## Consequences

- **Delivered**: `StackStorage<T, CAP>` + `Array::from_stack`/`from_stack_elem`,
  re-exported at the crate root; verified that the full op surface runs on
  stack-backed arrays with no per-backend code.
- **Parity matrix**: the "small fixed `MatrixN`/`VectorN`" row resolves to
  *Verified (stack storage)* for the allocation-free capability, with compile-time
  shape recorded as Excluded(architecture); the "Geometry" row resolves to
  Excluded(bounded-context). Both `Excluded?` flags are now decided.
- **Reversibility**: if a concrete consumer later needs type-level shapes or
  geometry, each is a tracked promotion (an [arch] item for type-level dims behind
  `generic_const_exprs`; a new downstream crate for geometry) — not a silent gap.
- This closes the parity program's open exclude-vs-implement decisions; remaining
  work is performance (promoting `Verified` rows to `Complete` with criterion
  baselines) and the consumer-driven PyO3 `ArrayD` interop (ADR 0007).
