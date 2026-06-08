# Leto: Systems-Optimized N-Dimensional Strided Array Library

Leto is a high-performance, N-dimensional strided array library written in Rust from first principles. It is engineered to replace `ndarray` as the shared memory and vocabulary layer across the Atlas simulation and learning stack.

It resides architecturally between low-level memory allocation (`mnemosyne`), SIMD kernels (`hermes-simd`), parallel iteration (`moirai`), and the downstream domains of spectral transforms (`apollo`) and differentiable deep learning (`coeus`).

---

## Mythological Context

In classical Greek mythology, **Leto** is a Titaness, daughter of **Coeus** and Phoebe, and mother of the twin deities **Apollo** and Artemis. 

Architecturally:
- `leto` acts as the direct vocabulary and memory layout bridge between `coeus` (intellect, tensors, and autodiff) and `apollo` (harmony, Fourier, and spectral transforms).
- It breaks the potential circular dependency loop between `coeus` and `apollo` by providing a shared, non-differentiable strided array definition that both can depend on.

---

## Workspace Structure

- **`leto`**: Core multidimensional array and view primitives. Defines layout offsets, shapes, strides, slicing, transposition, broadcasting, and storage abstractions (including `mnemosyne` alignment).
- **`leto-ops`**: High-performance mathematical kernels, element-wise maps, and reductions leveraging `hermes-simd` and parallelized via the `moirai` work-stealing thread scheduler.
- **`leto-python`**: Lightweight PyO3 and NumPy binding layer exposing zero-copy views to Python.

---

## Design Invariants

1. **Zero-Copy Layout Traversal**
   Layout manipulations (slicing, transposition, broadcasting, axis rolling) compute stride arithmetic and offset shifts without copying underlying element buffers.
2. **Explicit Storage Abstraction**
   Unlike general-purpose array libraries, Leto decouples strided layouts from the memory allocation backing. Arrays can be backed by standard heaps (`Vec`), borrowed memory slices (`&[T]`), or aligned heap blocks from the `mnemosyne` arena allocator.
3. **Monomorphization and Zero Overhead**
   Methods are generic over `<T: Scalar, S: Storage<T>, const N: usize>`, compiling directly to specialized inlined assembly blocks with zero dynamic dispatch.
