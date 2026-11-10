# Milestone ladder ([ADR 0037](../docs/adr/0037-release-scope-1-0.md))
<a id="M-array-core"></a> **M-array-core** (0.49.0) needs: (none) — array layout, storage, and view contracts.
- exit: run array basics example (`cargo run -p leto --example book_array_basics --locked --all-features` exits successfully).
- exit: run statistics example (`cargo run -p leto --example book_statistics --locked --all-features` exits successfully).
- exit: pass core array tests (`cargo test -p leto --locked --all-features` passes).
- standing: no open correctness item.
<a id="M-ops-linalg"></a> **M-ops-linalg** (0.50.0) needs: M-array-core — array operations and linear algebra.
- exit: pass operations and linear algebra tests (`cargo test -p leto-ops --locked --all-features` passes).
- exit: run sparse LU accuracy example (`cargo run -p leto-ops --example poisson_sparse_lu_accuracy --locked --all-features` exits successfully).
- exit: run scalar reference parity example (`cargo run -p leto-ops --example scalar_reference_parity --locked --all-features` exits successfully).
- standing: no open correctness item.
<a id="M-1.0"></a> **M-1.0** (1.0.0) needs: M-ops-linalg — stable array substrate release (authority-gated).
- standing: `cargo fmt --all -- --check`, `cargo clippy --workspace --locked --all-targets --all-features -- -D warnings`, `cargo test --workspace --locked --all-features`, `cargo test --workspace --locked --doc --all-features`, and `cargo doc --workspace --locked --no-deps --all-features` pass; SemVer and publish dry-run gates required for declared releases pass; README and rustdoc are current; no open correctness item.
