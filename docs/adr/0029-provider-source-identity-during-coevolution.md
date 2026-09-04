# ADR 0029: Provider Source Identity During Co-evolution

## Status

Accepted on 2026-09-03.

## Context

Leto consumes first-party Hermes, Mnemosyne, Eunomia, and Moirai packages.
During coordinated changes, Cargo can resolve the same package name from
different commits when a transitive provider still follows its default branch.
Those nominally identical Rust types then belong to distinct crate instances,
which prevents values such as layouts and buffers from crossing provider
boundaries. The owning dependency edge must be corrected at the provider, not
worked around in consumers.

This decision is tracked by
[LETO-MNEMOSYNE-DEFAULT-2026-09-04](../../backlog.md#leto-mnemosyne-default-2026-09-04).

## Decision

Leto follows reviewed provider default branches, while `Cargo.lock` records
the exact revisions that define its reproducible source graph:

- Hermes PR #155 is merged; its temporary `rev` pin is removed and the
  standalone lockfile records the merged default-branch source;
- Mnemosyne `main` already contains the Eunomia source correction, so obsolete
  PR #123 is closed and its temporary `rev` pin is removed;
- Eunomia PR #87 is merged; its temporary `rev` pin is removed and the
  standalone lockfile records the merged default-branch source;
- Aequitas PR #51 is merged; its temporary `rev` pin is removed and the
  standalone lockfile records the merged default-branch source;
- Moirai PR #256 at `70d201a`; its temporary `rev` pin is now removed and the
  standalone lockfile records the merged default-branch source.

Consumers follow the resulting Leto revision rather than adding an override of
their own.

## Alternatives Rejected

- A downstream `[patch]` or path override would move ownership of the source
  graph into each consumer and allow Leto's published dependency graph to
  drift.
- Converting values between duplicate crate instances would be a compatibility
  adapter and would preserve the duplicate graph instead of correcting it.
- Before PR #256 merged, leaving Moirai on its default branch reproduced the
  older Mnemosyne source and defeated the Hermes/Mnemosyne identity correction;
  the merged provider now carries the corrected source identity.

## Verification

At the decision revision, Leto's standalone lock check, workspace check,
warning-denied Clippy, nextest, doctests, rustdoc, and diff checks pass. The
nextest run executes 923 tests. The lockfile resolves provider defaults through
one Eunomia identity without a compatibility layer.

## Revision Note

2026-09-03: Added the Moirai source edge to the existing Hermes identity
increment after the dependency graph audit found that Moirai's default-branch
edge still selected the pre-Mnemosyne-identity graph.

2026-09-04: Removed Moirai's temporary revision pin after PR #256 merged at
`70d201a` and regenerated the standalone lockfile; the remaining provider pins
stay until their corresponding upstream increments merge.

2026-09-04: Removed the Eunomia and Aequitas temporary revision pins after
Eunomia PR #87 and Aequitas PR #51 merged. This restores one Eunomia trait
identity for Leto and unpinned Gaia consumers. Hermes PR #155 was then merged
and its temporary pin removed; Mnemosyne remains pinned while PR #123 is open.

2026-09-04: Advanced the retained Mnemosyne pin to PR #123's current head,
`a07f999`, after dependency-tree inspection showed the previous `da5c6be`
revision still selected pre-merge Eunomia types. That head removes Mnemosyne's
obsolete Eunomia PR #87 revision after the provider merge.

2026-09-04: Closed obsolete Mnemosyne PR #123 after confirming its source-only
correction already exists on `main`; independent review rejected the stale PR's
unrelated conflicting allocator changes. Removed Leto's temporary Mnemosyne
revision pin and restored provider-default resolution.
