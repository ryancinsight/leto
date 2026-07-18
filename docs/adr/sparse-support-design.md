# ADR: Sparse Array Support in Leto and Hephaestus

## Status
Proposed

## Context
The Atlas Physics Simulation Suite requires sparse array/tensor formats (CSR, CSC, COO, block-sparse) for efficient representation of sparse matrices in numerical methods (FEM, BEM, finite difference stencils). Current leto and hephaestus only support dense storage.

## Decision

### Architecture Principles
1. **API Symmetry**: Sparse APIs mirror dense APIs where possible
2. **Zero-Copy Conversion**: Format conversions without data copying
3. **Backend Agnostic**: Same trait surface for CPU (leto) and GPU (hephaestus)
4. **Typestate Patterns**: Compile-time format guarantees
5. **GAT-Driven Lending**: Iterator patterns for sparse traversal

### Storage Formats

#### CSR (Compressed Sparse Row)
- Efficient for row-major operations
- Structure: `data: Vec<T>`, `col_indices: Vec<usize>`, `row_ptr: Vec<usize>`
- Suitable for: Sparse matrix-vector multiplication, row-wise operations

#### CSC (Compressed Sparse Column)
- Efficient for column-major operations  
- Structure: `data: Vec<T>`, `row_indices: Vec<usize>`, `col_ptr: Vec<usize>`
- Suitable for: Column-wise operations, transpose operations

#### COO (Coordinate Format)
- Efficient for construction and format conversion
- Structure: `data: Vec<T>`, `row_indices: Vec<usize>`, `col_indices: Vec<usize>`
- Suitable for: Matrix assembly, format conversion

#### Block-Sparse
- Efficient for structured sparsity patterns
- Structure: Block-level sparse format with dense blocks
- Suitable for: Multi-scale methods, block-structured problems

### Module Structure (leto)

```
leto/src/infrastructure/sparse/
├── mod.rs              # Public API and re-exports
├── traits.rs           # SparseStorage, SparseStorageMut traits
├── csr.rs              # CSR implementation
├── csc.rs              # CSC implementation  
├── coo.rs              # COO implementation
├── block.rs            # Block-sparse implementation
├── convert.rs          # Zero-copy format conversions
└── ops.rs              # Sparse-dense arithmetic operations
```

### Trait Design

```rust
pub trait SparseStorage<T> {
    fn nnz(&self) -> usize;  // Number of non-zero elements
    fn format(&self) -> SparseFormat;
    fn to_csr(&self) -> CsrArray<T>;
    fn to_csc(&self) -> CscArray<T>;
    fn to_coo(&self) -> CooArray<T>;
}

pub trait SparseStorageMut<T>: SparseStorage<T> {
    fn add_entry(&mut self, row: usize, col: usize, value: T);
    fn remove_entry(&mut self, row: usize, col: usize);
}
```

### Integration with Dense Arrays

```rust
// Sparse-dense multiplication
impl<T> Mul<&Array<T>> for &CsrArray<T> {
    type Output = Array<T>;
    fn mul(self, dense: &Array<T>) -> Array<T> { ... }
}

// Dense-sparse multiplication  
impl<T> Mul<&CsrArray<T>> for &Array<T> {
    type Output = Array<T>;
    fn mul(self, sparse: &CsrArray<T>) -> Array<T> { ... }
}
```

### Hephaestus GPU Support

GPU sparse kernels will follow the same trait surface but use:
- WGSL compute shaders for sparse operations
- cuSPARSE-compatible kernels for CUDA backend
- Metal sparse frameworks for Apple GPU backend

### Solvers

Iterative solvers (CPU):
- Conjugate Gradient (CG)
- GMRES
- BiCGSTAB

Direct solvers (where feasible):
- Sparse LU for small structured systems

### Coeus Integration

Sparse-aware autodiff primitives:
- Gradient tracking through sparse operations
- Sparse Jacobian/Hessian representations
- Efficient backpropagation for sparse networks

## Consequences

### Positive
- Efficient sparse matrix operations for physics simulations
- Unified API across CPU and GPU backends
- Zero-copy format conversions reduce memory overhead
- Typestate patterns prevent format errors at compile time

### Negative  
- Increased API surface complexity
- Additional maintenance burden for sparse implementations
- GPU sparse kernel development overhead

### Mitigations
- Sparse features behind feature flags to reduce binary size
- Extensive testing and benchmarking to ensure correctness
- Documentation and examples for sparse usage patterns

## Implementation Phases

1. **Phase 1**: Core sparse storage traits and COO format (construction)
2. **Phase 2**: CSR and CSC formats with conversions
3. **Phase 3**: Sparse-dense arithmetic operations
4. **Phase 4**: Iterative solvers (CG, GMRES)
5. **Phase 5**: Block-sparse format
6. **Phase 6**: GPU sparse kernels (hephaestus)
7. **Phase 7**: Coeus autodiff integration

## References
- SuiteSparse documentation
- cuSPARSE API reference
- PETSc sparse matrix design
- Eigen sparse module architecture
