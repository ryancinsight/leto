# ADR 0017: Retire public ndarray compatibility

- Status: Accepted
- Date: 2026-07-21
- Class: [major] [arch]

## Context

Leto owns Atlas-native strided arrays. Its public `ndarray-compat` feature
instead admitted `ndarray` into the production graph, re-exported that crate,
and implemented conversions for owned and borrowed arrays. The borrowed-view
implementations reconstructed physical spans with unsafe raw-slice operations.

No live Atlas manifest or Rust caller uses this feature. Apollo now consumes
native Leto arrays and retains independent mathematical references at its
validation boundary. Leto's canonical tests independently cover every retained
array behavior that the conversion-only contract suite also exercised.

## Decision

- Remove the `ndarray-compat` feature, optional production dependency, module,
  public re-export, conversion implementations, and conversion-only tests.
- Keep `ndarray` as a dev-dependency oracle for differential tests and
  benchmarks. Oracle code does not define or leak into the public contract.
- Language, FFI, and external-library consumers construct native Leto arrays at
  their ownership boundary. Leto does not own consumer-specific adapters.
- Preserve one canonical test owner for constructors, storage, transpose,
  broadcast, mutable views, signed strides, slicing, and reshape semantics.

No compatibility alias, forwarding module, or replacement conversion shim is
retained.

## Public migration

- Remove `features = ["ndarray-compat"]` from Leto dependencies.
- Replace `leto::ndarray_compat` and `leto::ndarray` imports with native Leto
  `Array`, `ArrayView`, or `ArrayViewMut` contracts.
- At an unavoidable third-party boundary, validate shape and storage once and
  construct the native owner there. Do not add that boundary back to Leto.
- Keep third-party arrays only in consumer-local differential tests when they
  serve as independent oracles.

## Coverage argument

Deleting a conversion implementation removes its cross-representation contract;
it does not remove the underlying Leto operation. The retained suites verify:

- constructors, storage bounds, and Mnemosyne ownership in `core/storage`;
- transpose, broadcast, reshape, and logical-order materialization in
  `core/transform`;
- mutable axis views in `core/indexing`;
- signed-stride slicing and bounds in `core/slicing` and
  `layout_property_tests`.

Therefore the deleted tests do not create an unverified Leto operation. They
remove only contracts for a public boundary that no supported consumer uses.

## Rejected alternatives

- Retaining the feature for hypothetical external consumers preserves a second
  array owner and two avoidable unsafe blocks.
- Moving conversions into another Leto module changes the path but not the
  dependency-direction defect.
- Keeping a deprecated re-export or forwarding shim preserves the breaking
  surface indefinitely and violates the single-owner migration policy.

## Consequences

- This is a breaking public change: feature selection and conversion impls no
  longer compile. The changelog and this ADR provide the migration.
- Leto's production graph is independent of `ndarray`; oracle coverage remains
  available in dev targets.
- The public unsafe surface decreases, and FFI ownership remains with the
  consumer that can validate the actual boundary contract.

## Verification

- scan production manifests and Rust sources for removed-surface residue;
- inspect Cargo's normal and dev dependency graphs separately;
- run configured Nextest, doctests, warning-denied Clippy, and Rustdoc;
- run Rust public SemVer classification against the pre-removal revision;
- refresh and verify Apollo against the merged Leto revision.
