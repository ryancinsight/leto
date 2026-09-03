# ADR 0007: Dynamic rank (`IxDyn`) as a boundary carrier with a zero-copy rank bridge

- Status: Accepted
- Date: 2026-06-15
- Class: [minor] (purely additive: a new boundary type + bridge; the const-rank
  core is unchanged)
- Supersedes: refines ADR 0002's option-2 rejection (see *Relationship to ADR
  0002*).

## Revision 2026-09-03

`LayoutDyn` now exposes checked physical-span bounds, exact injectivity
validation, and metadata-only broadcasting. These operations remain
rank-agnostic layout validation and view construction; they do not add a
dynamic-rank compute substrate. Their arithmetic is shared with `Layout<N>`
through `domain::layout::kernels`, so Hephaestus can validate runtime-rank
fusion views without copying element storage or maintaining a parallel layout
algorithm.

## Context

ndarray exposes `IxDyn` — a first-class array whose **rank is a runtime value**
(shape/strides carried in a heap container) — so the *same* `ArrayBase` type
handles rank known only at run time (loading an arbitrary `.npy`, a generic
tensor pipeline, a numpy array crossing the PyO3 boundary). Leto's
`Array<T, S, const N: usize>` carries **rank in the type system**: `[usize; N]`
shape/stride storage, monomorphized stride loops, compile-time shape checks, and
deletion of the runtime shape-mismatch error variants. This is the codebase's
central invariant (ADR 0002).

The parity gap (`parity_matrix.md` §A, "Dynamic rank `IxDyn`") is therefore not a
missing kernel — it is the absence of any way to *carry* and *bridge* rank that
is unknown at compile time. The concrete drivers:

1. **PyO3 boundary** (the project's stated interop mission): a `numpy.ndarray`
   has runtime rank; the binding cannot pick a `const N` a priori.
2. **Generic I/O**: deserializing arrays whose rank is data-dependent.
3. **Ergonomic interop** with ndarray-shaped APIs that hand back `IxDyn`.

## Options

1. **`Dimension`-trait genericization** (ndarray's design): make `Layout<D>` /
   `Array<S, D>` generic over a `Dimension` trait whose associated storage is
   `[usize; N]` (static) or a heap vector (dynamic).
2. **Parallel dynamic compute substrate**: a `DynArray` that re-implements the
   reduction/linalg/iterator kernels over a runtime-rank layout.
3. **Boundary carrier + zero-copy rank bridge** (chosen): a small `ArrayD<T, S>`
   that *holds* runtime-rank data and supports only rank-agnostic operations
   (construct, inspect, index, reshape, materialize); all **compute** is reached
   by recovering a static `Array<T, S, N>` through a zero-copy bridge
   (`into_dimensionality::<N>()` / `into_dyn()`), reusing the existing
   monomorphized kernels unchanged.

## Decision

Adopt **option 3**. Leto remains const-rank for all computation. `ArrayD` is a
**boundary type**, not a compute substrate: it owns a `LayoutDyn`
(`Box<[usize]>` shape/strides + offset) and exposes only the operations that are
genuinely independent of compile-time rank, including checked span and alias
validation plus metadata-only broadcast construction. Numeric work is
performed after a one-line recovery to a typed rank.

The offset/size/validation arithmetic is **not duplicated**: it is extracted into
slice-based domain kernels (`domain::layout::kernels`) that both `Layout<N>` and
`LayoutDyn` delegate to (SSOT). `Layout<N>` keeps its `[usize; N]` storage and
const-generic API; only the *arithmetic body* is shared.

### The bridge (zero-copy)

```text
Array<T, S, N>  --into_dyn()-->            ArrayD<T, S>     (always succeeds)
ArrayD<T, S>    --into_dimensionality::<N>()-->  Result<Array<T, S, N>>  (rank-checked)
```

**Theorem (bridge is allocation-free and value-preserving).** Both directions
move the storage `S` by value and only translate the layout container
(`[usize; N]` ⇆ `Box<[usize]>`); no element is read, copied, or reallocated, and
the logical element at every index is unchanged.

*Proof.* `Array<T,S,N>` and `ArrayD<T,S>` share the identical layout semantics
`offset(i) = base + Σ iₖ·strideₖ` evaluated by the same `physical_offset` kernel;
they differ only in how `(shape, strides)` are stored (`[_; N]` vs `Box<[_]>`).
`into_dyn` copies the `N` shape/stride scalars into freshly boxed slices of length
`N` and moves `S`; `into_dimensionality::<N>` checks `ndim == N`, copies the `N`
scalars back into `[_; N]`, and moves `S`. The scalar values and `base` offset are
preserved bit-for-bit, so for every valid index the computed physical offset — and
hence the addressed element — is identical. Only `O(ndim)` shape/stride scalars
are touched; the `O(len)` element buffer inside `S` is moved, never traversed. ∎

### Dynamic-rank dispatch pattern

When rank is genuinely runtime-valued, callers recover a typed rank with a bounded
`match` (each arm monomorphizes a const-`N` path), exactly as ADR 0002 prescribes
for the Coeus boundary:

```rust
let result = match a.ndim() {
    1 => f(a.into_dimensionality::<1>()?),
    2 => f(a.into_dimensionality::<2>()?),
    3 => f(a.into_dimensionality::<3>()?),
    n => return Err(LetoError::StorageError {
        reason: format!("rank {n} exceeds supported dispatch range"),
    }),
};
```

A dispatch *macro* is deliberately **not** shipped here: it is `macro_rules!`
(last-resort per the macro policy), the supported rank set is consumer-specific,
and the explicit `match` is auditable and IDE-friendly. If a second consumer needs
the same closed dispatch it is consolidated then (rule-of-two), preferably as a
`build.rs`-generated helper rather than a declarative macro.

## Consequences

- **New surface** (additive, `[minor]`): `ArrayD<T, S>`, `LayoutDyn`, the bridge
  methods, and a deep `dynamic/` leaf hierarchy
  (`domain/layout/kernels.rs`, `domain/dynamic/layout.rs`,
  `application/dynamic/{array,bridge}.rs`).
- **Core untouched**: every existing `Array<T,S,N>` signature, kernel, and test is
  unchanged; the kernels refactor is behavior-preserving (guarded by the existing
  suite + miri). Zero regression risk to the hot path.
- **No compute duplication**: `ArrayD` has no reductions/linalg/elementwise
  kernels; those remain single-authored on the const-rank core and are reached via
  the bridge. This is the SSOT/DRY-correct realization of "runtime-dim escape types
  only at I/O boundaries, converted to typed shape immediately after validation"
  (standards).
- **PyO3 follow-up** (consumer-driven, tracked, not built here): the leto-python
  bindings can accept arbitrary-rank `numpy` arrays as `ArrayD` and bridge to the
  typed kernels, removing the current compile-time-rank constraint at that
  boundary.
- **Parity**: `parity_matrix.md` "Dynamic rank `IxDyn`" moves to **Verified** for
  the boundary-carrier + bridge scope; full *dynamic-rank compute* (operating
  without recovering a static rank) remains intentionally out of scope and is
  recorded as such — it is provided by rank recovery, by design.

## Relationship to ADR 0002

ADR 0002 rejected a `DynArray`/`IxDyn` escape type **for the Coeus compute
boundary**, because as a *compute substrate* it would fork the layout model and
reintroduce runtime shape checks. ADR 0007 refines, not contradicts, that ruling:
`ArrayD` is narrowed to a **non-compute boundary carrier**. The const-rank
compute invariant — the basis of 0002's decision — is fully preserved; the dynamic
type never carries a kernel. ADR 0002's bounded-`match` dispatch remains the
sanctioned way to cross from runtime rank into compute; ADR 0007 simply gives that
crossing a typed, zero-copy vehicle.
