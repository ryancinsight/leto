# ADR 0026: Retire the `LendingIterator` seam

- Status: Accepted
- Date: 2026-08-17
- Class: [major] (removes the public trait `leto::LendingIterator` and the
  public module `leto::application::iter::lending`)

## Context

`leto` published a GAT-based streaming-iterator trait:

```rust
pub trait LendingIterator {
    type Item<'this> where Self: 'this;
    fn next(&mut self) -> Option<Self::Item<'_>>;
    fn count_remaining(&mut self) -> usize { /* consumes self */ }
}
```

It was introduced alongside `Tiles`, the non-overlapping tile-view iterator,
on the premise that a tile view borrows from the iterator and therefore cannot
be expressed as a plain `Iterator`.

That premise was wrong. A `Tiles` item is an `ArrayView<'a, T, N>` into the
*parent slice*; `view_at` borrows `&self` only to read `Copy` grid state, and
the returned view carries the parent lifetime `'a`, not the iterator's. `Tiles`
has accordingly implemented `Iterator`, `DoubleEndedIterator` and
`ExactSizeIterator` since the tile iterator was corrected, and its own tests
exercise `for` loops, `zip`, `enumerate`, `rev`, `map`/`collect` and
`size_hint` — all of which the GAT signature would have precluded.

That left `LendingIterator` a public seam with no implementor. A stack-wide
search (`leto`, `leto-ops`, `kwavers`, `CFDrs`, and the remaining atlas
members) found:

- no `impl LendingIterator` anywhere outside leto's own test module, whose
  `ScratchLender` fixture existed only to prove the trait compiled;
- no call to `count_remaining` outside that same fixture's test;
- one consumer import, `kwavers`'s `tiled_kspace_processing` example, which
  was dead — `Tiles::next` resolved to `Iterator::next`, not the trait — and
  which has since been migrated to `Iterator`/`ExactSizeIterator`
  (kwavers `8c232e4a8`);
- two documentation claims in the CFDrs book asserting that
  `cfd-2d::stencil`/`cfd-3d::stencil` consume `LendingIterator` and a
  `TileStreaming` trait. Neither symbol occurs anywhere in CFDrs, and
  `TileStreaming` has never existed in leto. Both claims were fabricated.

A published trait whose only implementor is a fixture written to justify it is
speculative generality, not an extension seam: there is no current implementor
to validate the contract against, and no documented next one.

## Decision

Remove `LendingIterator` and rename the module that housed it from
`application::iter::lending` to `application::iter::tiles`, which is what it
now contains. leto publishes no lending-iterator abstraction.

Rejected alternatives:

- **Keep it as a declared seam.** Rejected: seam-first extensibility requires
  a present requirement and validation against a real implementor plus a
  documented next one. Neither exists, and the trait's own doc comment
  conceded it becomes redundant when RFC 3301 lands — a seam whose stated
  future is deletion is not one to publish now.
- **Keep it and deprecate.** Rejected: a `#[deprecated]` re-export is the
  compatibility shim the anti-shim rule forbids. The consumer was migrated
  first, so nothing needs bridging.
- **Keep the module name `lending`.** Rejected: after the removal the module
  holds only `Tiles`, so `lending` names a concern the module no longer has.
  The rename costs nothing beyond the major bump the trait removal already
  requires.

## Consequences

`cargo semver-checks` reports `trait_missing` for all four prior public paths
(`leto::LendingIterator`, `leto::application::LendingIterator`,
`leto::application::iter::LendingIterator`,
`leto::application::iter::lending::LendingIterator`) and `module_missing` for
`leto::application::iter::lending`. Both are major; leto's `Unreleased`
section already carries the ADR 0025 `Layout` major, so this ships in the same
major release.

Migration for downstream code is an import deletion. `Tiles` needs no trait in
scope for `.next()`, `for`, or any adaptor. The single method with no
`Iterator` counterpart, `count_remaining()`, maps to `ExactSizeIterator::len()`
where the length is exact — for `Tiles` it is, because `Tiles::new` rejects a
layout addressing outside its backing slice, so iteration cannot terminate
early — and unlike `count_remaining` it does not consume the iterator. Where a
length is not exactly known, `Iterator::count()` is the consuming equivalent.

A type that genuinely lends — one whose item borrows the iterator itself, such
as a reused scratch buffer — should declare its own trait locally until RFC
3301 stabilizes. leto has no such type.

The remaining GAT in the leto workspace is
`leto_ops::StatefulUpdateRule::State<'a>`, which is re-parameterized per
lifetime by each update rule and is unaffected.
