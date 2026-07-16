# ADR 0013: Provider-default MSRV alignment

Status: accepted

## Context

Leto's direct Atlas dependencies were pinned to individual revisions after the
providers had merged their default-source convergence. The pins block normal
stack integration and hide Mnemosyne 0.5/Core 0.2's Rust 1.95 requirement.
The workspace previously had no declared MSRV, and its packages did not inherit
one from the workspace metadata.

## Decision

Leto 0.37.0 removes direct revision pins for Mnemosyne, Moirai, Hermes,
Eunomia, and Themis. The providers resolve from their merged default branches;
`Cargo.lock` remains the reproducibility pin. The workspace declares Rust 1.95
and each published member inherits that value with `rust-version.workspace`.

## Consequences

Consumers must use Rust 1.95. The release is a pre-1.0 minor transition because
the supported compiler range changes. No adapter, local patch, or alternate
provider source remains.

## Verification

Rust 1.95 compiles `leto-ops` and Rust 1.94 rejects the declared graph.
Formatter, explicit-nightly warning-denied release Clippy, configured release
Nextest (568/568), doctests (9/9), rustdoc, and provider-source identity pass.
Offline rustdoc SemVer comparisons against the clean 0.36 baseline pass all 196
applicable checks for `leto` and `leto-ops`.
