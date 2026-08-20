# Position in the Atlas Stack

Leto is the shared host-array substrate between tensor/autodiff consumers and
transform consumers:

`Mnemosyne` / `Moirai` → `Leto` → `Coeus` and `Apollo` → `Hephaestus`

The arrow is a dependency direction, not a runtime execution claim.

## Boundary ownership

- **Leto** owns shapes, strides, storage, views, slicing, broadcasting,
  structural operations, dense representations, and the CPU array vocabulary.
- **Leto-ops** owns numerical kernels, reductions, matrix products, dense and
  sparse linear algebra, and the generic scalar/execution seams built on Leto.
- **Coeus** owns tensors, autodiff graphs, neural-network orchestration, and
  optimizers. It consumes Leto arrays rather than reimplementing layout logic.
- **Apollo** owns Fourier and other spectral transforms. It consumes Leto views
  and contiguous blocks at transform boundaries.
- **Hephaestus** owns accelerator kernels. A GPU operation may use Leto layout
  metadata, but device buffers and dispatch remain outside Leto's domain types.
- **Mnemosyne**, **Moirai**, **Hermes**, **Themis**, and **Melinoe** provide
  memory, scheduling, SIMD, placement, and capability contracts through the
  dependency direction selected by each provider. Leto's core stays free of
  concrete accelerator and scheduler state.

## Why the boundary matters

An array view is a data-plane contract: shape, stride, offset, element type,
and lifetime. A scheduler or GPU queue is control-plane infrastructure. Keeping
them separate lets Apollo and Coeus share one layout implementation, lets
`leto-ops` select a policy at kernel granularity, and keeps the core testable
with deterministic CPU values.

The thin `leto-python` crate exposes this Rust contract to Python. Bindings
convert arguments and release the Python GIL around Rust computation; domain
logic remains in the Rust crates. The book and Rustdoc describe the same
provider-owned APIs, so an example is a contract check rather than a separate
Python implementation.
