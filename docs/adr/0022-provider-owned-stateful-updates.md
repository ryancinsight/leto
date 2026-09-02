# ADR 0022: Provider-owned CPU stateful updates

- Status: Accepted
- Date: 2026-08-01
- Refs: `backlog.md#leto-stateful-update-1`; Hephaestus ADR 0045; Coeus
  optimizer provider migration.

## Context

Coeus owns CPU formulas for SGD, Adam, AdamW, RMSProp, and AdaGrad. This leaves
one mathematical family in a consumer even though Leto owns CPU array views,
validated layouts, scalar contracts, and zero-copy mutable zip traversal.
Hephaestus now owns the corresponding accelerator contract. Keeping the CPU
formulas in Coeus would preserve a second source of truth and prevent direct
backend dispatch through provider APIs.

Each update mutates a parameter view and one or two state views from their
pre-update values and a borrowed gradient view. Invalid parameters or array
contracts must fail before any output changes.

## Decision

1. `leto-ops` owns one `stateful_update` entry point parameterized by a sealed
   zero-sized rule marker, scalar `T`, and const rank `N`. The marker supplies
   the parameter and state family through associated types; static dispatch
   monomorphizes the complete rule without trait objects or per-element
   capability checks.
2. The public request borrows `ArrayViewMut` parameter/state views and an
   `ArrayView` gradient. It reuses the existing `zip_mut_with` traversal, dense
   fast path, and arbitrary-stride index mapping, so execution allocates no
   tensor-sized temporary and copies no operand.
3. Parameters are validating scalar-preserving types. Arithmetic, bias
   correction, square root, and accumulation execute in `T: RealScalar`;
   f32/f64 are initial conformance instantiations. No widen-compute-narrow path
   is admitted.
4. Validation checks parameter domains, exact shape equality, storage spans,
   exact writable-layout injectivity, and rule state cardinality before calling the
   mutable traversal. Safe Rust borrowing makes cross-output aliasing
   unrepresentable; no runtime pointer comparison substitutes for ownership.
5. Coeus CPU adapters call this API directly. Hephaestus conformance uses it as
   the differential CPU oracle. Coeus then deletes its local formulas while
   selecting Leto for CPU and Hephaestus for accelerator backends.

## Alternatives

- Keep formulas in Coeus: rejected because the consumer remains the CPU
  provider and duplicates Hephaestus rule ownership.
- Add five public update functions: rejected because traversal, validation,
  and execution differ only by a statically representable rule dimension.
- Return newly allocated arrays: rejected because optimizer state is mutable,
  long-lived storage and caller-owned views express the required zero-copy
  contract.
- Reimplement traversal in the new module: rejected because mutable zip is the
  existing canonical provider for dense and strided multi-output traversal.

## Consequences

Leto gains an additive public CPU operation family and becomes the single CPU
owner for optimizer updates. The associated state family adds type complexity,
but invalid state cardinality becomes a compile-time construction error and
each rule remains a zero-cost specialization. Coeus must adopt fallible direct
provider calls in a later breaking migration; no compatibility adapter is
retained.
