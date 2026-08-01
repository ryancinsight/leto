# ADR 0021: Generic mutable zip output families

- Status: Accepted
- Date: 2026-07-31
- Class: [major] [arch]
- Board item: `LETO-MUTABLE-ZIP-PROVIDER-1`

## Context

Kwavers carried separate mutable zip implementations for one, two, and three
outputs. The bodies repeated shape validation, storage validation, and logical
row traversal while the output arity was the only structural variation. The
provider already owns the read-source tuple family through `ZipSources<N>`.

## Decision

`leto-ops` owns the output family through two sealed, statically dispatched
operation traits:

- `ZipMutOutputs<S, F, N>` for row-major mutation without indices;
- `IndexedZipMutOutputs<S, F, N>` for row-major mutation with `[usize; N]`.

Each operation supports one, two, or three mutable `ArrayViewMut` outputs. A
zero-source operation passes `()` as the source set. Read-only sources remain
`ZipSources<N>` values or heterogeneous tuples, so output and source element
types are inferred and monomorphized at the callsite. No output allocation,
type erasure, or per-element dynamic dispatch is introduced.

The public entrypoints are `zip_mut_with` and `indexed_zip_mut_with`; output
arity is not encoded in an identifier. Kwavers consumers call these provider
functions directly and no longer carry local mutable-output traits or wrappers.

Separate operation traits are used instead of a higher-ranked generic
associated-value callback because the current stable compiler promotes that
HRTB/GAT combination to a `'static` requirement at this boundary. The split
keeps the operation monomorphic without requiring a copy or an owned callback
value.

## Alternatives

- Keep `zip_two_mut_with` and `zip_three_mut_with`: rejected because each new
  output arity would duplicate the traversal and validation kernel.
- Use `Cow` to combine mutable and immutable views: rejected because mutability
  is an ownership and aliasing contract, not a copy-on-write policy; `Cow` would
  obscure whether writes target caller storage and could allocate on mutation.
- Put array/view semantics in Moirai: rejected because Moirai owns scheduling
  primitives, while Leto owns array layout, view validation, and traversal
  semantics.
- Use dynamic output trait objects: rejected because vtable dispatch and type
  erasure are unnecessary on this operation path.

## Migration and verification

Kwavers root, physics, therapy, analysis, solver, and core consumers now call
the Leto entrypoints directly. The superseded local root, therapy, physics,
and core mutable zip/indexed-source wrappers are deleted. Provider tests cover
one/two/three outputs, zero sources, heterogeneous read sources, dense and
strided views, and logical indices. Consumer verification is package-scoped
compile plus focused nextest; no runtime or benchmark gain is claimed without
controlled measurements.
