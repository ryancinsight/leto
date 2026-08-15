# ADR 0025: Encapsulate `Layout` construction and place storage bounds at the access site

- Status: Accepted
- Date: 2026-08-15
- Class: [major] (`Layout`'s three `pub` fields and `Layout::new` are removed;
  the struct becomes `#[non_exhaustive]`. Consumed cross-repo by coeus,
  hephaestus, kwavers, athena and apollo)

## Context

`Layout<const N: usize>` carried three `pub` fields (`shape`, `strides`,
`offset`) and a non-validating `pub const fn new`. The board item recorded this
as a Tier 0 soundness defect on the premise that leto's 84 `unsafe` blocks rest
on a `Layout` invariant that safe downstream code could violate.

Investigation only partly confirmed that premise, and the difference decides the
design.

**What is true.** Safe code could reach an out-of-bounds read *and* write. A
four-line proof of concept, executed before any change, read roughly 4 KiB past
a 16-byte stack buffer without panicking.

**What is false, and load-bearing.** The defect is not that `Layout` is
unvalidated. `Layout` holds no pointer and no length, so "addresses only memory
inside the buffer" is not a property it can express at all. The proof of concept
above used a layout built by the *validating* `Layout::c_contiguous`
constructor: `c_contiguous([1000])` is a perfectly self-consistent layout that
simply does not fit a four-element buffer. Sealing `Layout` would not have
closed the hole.

The actual break is a three-part perimeter failure:

1. `ArrayView::new` / `ArrayViewMut::new` are safe, `pub`, and non-validating,
   while their `try_new` siblings call `Layout::validate_storage_len`. The
   layout-versus-buffer invariant is therefore a documentation convention, not a
   type-system fact, and every downstream `// SAFETY: the layout is validated`
   comment that rests on an `ArrayView` parameter is unjustified.
2. The four `Index`/`IndexMut` impls on `ArrayViewMut` computed a physical
   offset and dereferenced it without the `offset < self.len` check their
   `get`/`get_mut` siblings perform. This was the shortest path to undefined
   behavior.
3. Three `leto-ops` entry points reached `get_unchecked` or raw pointer writes
   with no storage validation at all, unlike the ~20 sibling entry points that
   do validate: `trace`, `kron`'s strided branch, and `matmul`'s
   `copy_back_to_out`. The last is an operand mixup rather than an omission —
   `validate_matmul` runs against the *scratch* output view, while
   `copy_back_to_out` writes through the caller's `dst`.

By contrast, leto's mutable-iterator constructors are correct: `ElementIterMut`,
`IndexedIterMut`, `TaskPartitionsMut`, `AxisIterMut` and `LanesMut` each
establish `validate_storage_len` (plus `is_injective` or
`has_zero_stride_aliasing`) before any raw dereference. The library was already
disciplined; the perimeter had three gaps.

## Decision

Split the two invariants explicitly and enforce each where it can actually be
decided.

**The self-contained invariant belongs to `Layout`.** Fields become private,
the struct becomes `#[non_exhaustive]`, and `Layout::try_new` (mirrored by
`TryFrom<([usize; N], [isize; N], usize)>`) becomes the only construction path
outside the crate. It validates exactly what a layout can know about itself:

1. the shape product fits in `usize`;
2. every addressed physical offset — `offset` plus each partial sum of
   `(shape[i] - 1) * strides[i]` — neither overflows `isize` nor falls below
   zero.

This is the precondition that makes the infallible accessors (`size`,
`min_max_offsets`, `offset_of`) total rather than panicking, which is a real
guarantee, just not the anti-UB one.

Zero-cost `#[inline] const` accessors (`shape()`, `strides()`, `offset()`)
replace the public fields; all three return by value, the arrays being `Copy`.

A crate-private `from_parts_unchecked` serves the ~15 internal derivation sites
where a layout is computed from an already-validated parent — slices,
transposes, broadcasts, per-step iterator subviews. Several sit inside iterator
`next` bodies, so re-validating there would put an O(N) check on a hot path to
re-establish something already proven. It is safe rather than `unsafe` because a
`Layout` owns no pointer: a violation degrades the infallible accessors to
panics and cannot itself cause UB.

Cross-crate derivation sites (leto-ops, and the consumers' submatrix and rank
bridges) use `try_new` with `expect("invariant: …")`, carrying the proof to the
panic site per the panic policy.

`Deserialize` routes through `try_new`, closing the untrusted-input path that
previously reconstructed the struct field-by-field.

**The layout-versus-buffer invariant belongs to the access site.** The four
`ArrayViewMut` `Index`/`IndexMut` impls now assert `offset < self.len` before
dereferencing, matching `get`/`get_mut`. `trace` and `kron` call
`validate_storage_len` on their operands, and `copy_back_to_out` validates
`dst` rather than relying on the scratch view's check.

## Alternatives rejected

**Seal `Layout` only, per the board's original acceptance oracle.** Rejected on
evidence: the proof of concept survives it unchanged, because the offending
layout is valid. Delivering the oracle alone would have closed the item while
leaving the defect, which is the failure mode the oracle existed to prevent.

**Make `ArrayView::new` / `ArrayViewMut::new` `unsafe`, or validating.** This is
the correct root-cause fix and is *not* delivered here — see Consequences. It
touches 69 call sites across seven repositories and is a second, larger
`[major]` break. Bundling it would have produced one unreviewable change; the
`Index` bound checks close the demonstrated hole in the meantime at the cost of
a per-index comparison on a path that already performed a fallible
`offset_of`.

**Keep `Layout::new` as a `const fn` alongside `try_new`.** Rejected: a
non-validating public constructor is exactly the hole, and no call site in the
stack used `Layout::new` in a `const` or `static` context, so the `const`
capability was unused. One consumer (`athena-leto`'s `vector_layout`) lost its
`const fn` status as a consequence; validation requires runtime arithmetic and
`try_new` cannot be `const` while returning `Result` with the current
`checked_*` kernels.

**Validate inside `from_parts_unchecked`.** Rejected on the hot-path cost above,
and unnecessary: every caller inherits a proven invariant.

## Consequences

Safe construction of a self-inconsistent `Layout` is now impossible outside the
crate, and the demonstrated safe-code out-of-bounds read and write both panic
with a message naming the violated bound.

The perimeter is *not* fully closed. `ArrayView::new` and `ArrayViewMut::new`
remain safe and non-validating, so a caller can still pair a valid layout with
too small a buffer. Every consuming path now either bounds-checks at access or
validates at dispatch, so no known route reaches UB — but that is an
enumeration over the current tree, not a type-system guarantee, and it decays as
code is added. Converting those two constructors to `unsafe fn new_unchecked`
with `try_new` as the safe path is filed as the follow-on `[major]` item; until
it lands, a new `unsafe` accessor added without a bounds check reopens the hole.

Migration for consumers is mechanical: `layout.shape` → `layout.shape()` (and
likewise `strides`, `offset`); `Layout::new(s, t, o)` → `Layout::try_new(s, t, o)`
with `?`, `.expect("invariant: …")`, or `Layout::c_contiguous(shape)` where the
layout is dense row-major. Struct literals must become constructor calls.

## Verification

- 21 adversarial cases in `crates/leto/tests/core/layout_validation.rs`, one per
  invalid class (shape-product overflow, stride-extent overflow, accumulated
  offset overflow, base offset past `isize::MAX`, negative-stride underrun,
  mixed-sign minimum, zero-extent and unit-extent edge cases, hostile
  deserialization), each asserting the typed `LetoError` variant. Positive
  controls accompany each rejection class so the checks cannot degenerate into
  blanket refusal.
- The historical proof of concept is pinned as three `#[should_panic]`
  regression tests.
- 6 cases in `crates/leto-ops/tests/ops/storage_bounds.rs` covering `trace`,
  `kron` and `matmul`'s `copy_back_to_out`, each with an in-bounds positive
  control.
- `cargo semver-checks` reports 4 major breaks — `inherent_method_missing`,
  `struct_marked_non_exhaustive`, `struct_pub_field_missing`,
  `struct_pub_field_now_doc_hidden` — which is the intended and declared
  classification.
