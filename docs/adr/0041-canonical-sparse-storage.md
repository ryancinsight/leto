# ADR 0041: Own compressed sparse storage in Leto and numerical kernels in Leto Ops

- Status: Proposed
- Date: 2026-11-22
- Class: [arch] [major]

## Context

Leto exposes CsrArray, CscArray and CooArray. Leto Ops has a second CSR, CSC and COO family with validated parts and borrowed kernel views. Leto Ops depends on Leto. Prometheus constructs COO and calls CscArray::from_coo to preserve the species-by-reaction orientation. [ADR 0028](0028-sparse-array-support.md) establishes the storage boundary. The [index-width proposal](0038-csr-index-width.md) depends on choosing one representation.

## Proposed decision

Place canonical owned CSR, CSC and COO storage and borrowed views in Leto. Preserve the validated compressed-matrix invariants and provide the assembly, insertion, orientation-preserving conversion and traversal behavior required by current callers. Keep numerical kernels in Leto Ops over that storage. Select final public names during API and downstream review. Changing width or scalar bounds is a separate decision with its own proof and benchmarks.

For an m-by-n CSR matrix with k stored entries, the row pointer has length m+1, starts at zero, ends at k, and is nondecreasing. Values and column indices both have length k. Canonical columns in each row are strictly increasing and lie in [0,n). CSC applies the corresponding column invariant. Specify how COO duplicates and explicit zeros are handled at *each constructor*: callers may observe stored count, accumulation order and rounding. Validate raw parts before exposing views.

## Alternatives

Owning storage in Leto Ops forces base-crate consumers upward or introduces a reverse dependency. Keeping two families with adapters leaves two validators and risks allocation or transposition on the hot path. Moving one family by name alone could lose mutation, scalar-domain or constructor behavior.

## Review and migration gates

Inventory both families' methods, bounds, errors, ownership and all in-stack callers before sign-off. Port both test suites. Use rectangular asymmetric inputs, empty rows/columns, duplicate coordinates, explicit zeros and malformed parts. Compare dense reconstructions, products and stored counts, check borrowed operations without copying, and run the Prometheus stoichiometric product against its independent reference. Compare allocation and SpMV benchmarks with the current versions. Migrate consumers, supply a breaking-change guide and obtain the required provider review before removing the superseded surface.
