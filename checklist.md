# Leto Development Checklist

## 1. Setup and Infrastructure
- [ ] Initialize git repository in `repos/leto`.
- [ ] Establish directory structure (`crates/leto`, `crates/leto-ops`, `crates/leto-python`).
- [ ] Setup workspace-level lint rules and compiler profiles in `Cargo.toml`.

## 2. Core Library Crate (`leto`)
- [ ] Define errors (`LayoutError`, `StorageError`) in `domain/error.rs`.
- [ ] Implement `Layout<const N: usize>` with strides, offset, contiguity, and transpose in `domain/layout.rs`.
- [ ] Implement `Storage<T>` and `StorageMut<T>` with standard and `mnemosyne` backings in `infrastructure/storage.rs`.
- [ ] Implement `Array<T, S, const N: usize>` core struct in `application/array.rs`.
- [ ] Implement `ArrayView` and `ArrayViewMut` slicing and views in `application/view.rs`.

## 3. Operations Crate (`leto-ops`)
- [ ] Define `ExecutionStrategy` markers in `domain/strategy.rs`.
- [ ] Write contiguous and strided SIMD implementations in `infrastructure/simd.rs` using `hermes-simd`.
- [ ] Setup `moirai` loop scheduling in `infrastructure/parallel.rs`.
- [ ] Write arithmetic mapping and reductions in `application/map.rs`.

## 4. Python FFI Crate (`leto-python`)
- [ ] Setup PyO3 bindings and NumPy pointer conversions in `src/lib.rs`.

## 5. Verification and Validation
- [ ] Write unit tests for layout calculations.
- [ ] Verify zero-copy slicing and transposes.
- [ ] Benchmark operations comparing contiguous vs. non-contiguous layouts.
- [ ] Verify clippy correctness across all crates.
