# ADR 0020: Generic tuple source sets for multi-input zips

- Status: Accepted
- Date: 2026-07-31
- Class: [major] [arch]

## Context

`leto-ops` had separate mutable zip implementations for two, three, and five
read-only inputs, plus separate indexed implementations for two and four
inputs. Each function repeated shape validation, storage validation, dense
slice handling, and strided row traversal. The repeated arity-specific bodies
made the traversal contract harder to extend and increased the number of
public symbols without adding a distinct operation.

## Decision

Use one sealed `ZipSources<N>` source-set contract for multi-input zips.
Implement it for read-only `ArrayView` references and statically typed tuples
of two through twelve sources. The source set supplies its associated value,
offset, and dense-slice types; one generic traversal kernel handles validation,
the contiguous fast path, and strided row walking.

Expose:

- `zip_many_mut_with(lhs, sources, closure)` for non-indexed traversal.
- `indexed_zip_many_mut_with(lhs, sources, closure)` for logical-index
  traversal.

The closure receives the source values as one tuple, for example
`|output, (&a, &b, &c)| ...`. Tuple element types remain statically known, so
the source set and closure monomorphize without dynamic dispatch or a
per-element allocation. The existing optimized single-source
`zip_mut_with` and `indexed_zip_mut_with` contracts remain unchanged.

## Migration

Replace `zip2_mut_with`, `zip3_mut_with`, and `zip5_mut_with` with
`zip_many_mut_with` and pass the read-only views as one tuple. Replace
`indexed_zip2_mut_with` and `indexed_zip4_mut_with` with
`indexed_zip_many_mut_with`. Destructure the tuple in the closure. The old
arity-specific functions are removed; no forwarding wrappers or aliases are
retained.

## Alternatives

- Keep the arity-specific functions: rejected because each new arity would
  recreate the same validation and traversal implementation.
- Store heterogeneous sources behind dynamic trait objects: rejected because
  it adds vtable dispatch and erases the compile-time element types.
- Use a homogeneous slice of source views: rejected because multi-input zips
  legitimately combine different element types.
- Use a higher-ranked associated-value closure: rejected because the current
  Rust type system promotes local view borrows toward `'static` at this public
  boundary. The associated value type carries the view-data lifetime directly
  without changing ownership or introducing a copy.

## Verification

The conformance cases cover two, three, four, and five source tuples,
heterogeneous source element types, dense and transposed strided layouts,
indexed logical coordinates, and the existing finite-difference consumers.
`cargo check -p leto-ops --all-features` and
`cargo nextest run -p leto-ops --all-features` pass; the latter runs 448/448
tests. No runtime or benchmark improvement claim is made by this ADR.
