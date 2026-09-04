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
[LETO-HERMES-IDENTITY-2026-09-03](../../backlog.md#leto-hermes-identity-2026-09-03).

## Decision

During the current co-evolution window, Leto pins the exact revisions that
define its source graph:

- Hermes PR #155 at `5a399ee`;
- Mnemosyne PR #123 at `da5c6be`;
- Eunomia PR #87 at `fdbf122`; and
- Moirai PR #256 at `773c117`.

The pins remain in the workspace manifest as temporary, documented
co-evolution state. After each upstream PR merges, its `rev` is removed and
the standalone lockfile is regenerated. Consumers then follow the resulting
Leto revision rather than adding an override of their own.

## Alternatives Rejected

- A downstream `[patch]` or path override would move ownership of the source
  graph into each consumer and allow Leto's published dependency graph to
  drift.
- Converting values between duplicate crate instances would be a compatibility
  adapter and would preserve the duplicate graph instead of correcting it.
- Leaving Moirai on its default branch reproduces the older Mnemosyne source
  and defeats the Hermes/Mnemosyne identity correction.

## Verification

At the decision revision, Leto's standalone lock check, workspace check,
warning-denied Clippy, nextest, doctests, rustdoc, and diff checks pass. The
nextest run executes 923 tests. The lockfile resolves the four pinned provider
edges without requiring a compatibility layer.

## Revision Note

2026-09-03: Added the Moirai source edge to the existing Hermes identity
increment after the dependency graph audit found that Moirai's default-branch
edge still selected the pre-Mnemosyne-identity graph.
