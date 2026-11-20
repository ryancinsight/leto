# 0040 — Share the published scalar provider in Leto

## Status

Accepted (2026-11-19)

## Context

Leto accepts typed Aequitas quantities in public operations. Leto Ops
uses Hermes kernels for Eunomia scalar types. Aequitas 0.4.0 uses
Eunomia from crates.io. A graph that combines the old scalar source
with the published provider fails to satisfy the quantity and SIMD
trait bounds even when the numbers have the same representation.

The already-published Leto 0.50.0 manifest permits older Aequitas and
Eunomia versions. The release guard rejects rewriting those public
dependency requirements under the same version.

## Decision

Release Leto and Leto Ops together at 0.51.0. Select Aequitas 0.4.0 and
Eunomia 0.12.1 from crates.io and require Hermes 0.10.0 for SIMD
operations. Cargo must resolve one Eunomia crate and one derive crate
through the active public boundaries. Existing storage, solver, and
unit algorithms retain their behavior.

A second scalar alias or a graph-local override would leave ordinary
consumers with two identities. The new release line records the
dependency contract in the distributed manifests.

## Verification

Require the native, Miri, Python parity, and package gates at the same
candidate. Verify Hermes before publishing Leto, and Leto before
publishing Leto Ops. A fresh registry consumer must exchange a typed
quantity and use Leto Ops to compute values with the shared scalar
provider.


## Aequitas 0.5 adoption

The 0.52.0 candidate advances the public stencil spacing type to Aequitas
0.5.0. A consumer can pass a length directly when its graph resolves that
Aequitas identity. The native stencil and Eunomia 0.12.1 contract stay
the same. The bump guard requires a new release line for this change.
