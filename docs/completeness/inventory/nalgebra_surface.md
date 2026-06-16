# Oracle API Inventory: nalgebra 0.35

This document catalogs the public API surface of `nalgebra 0.35` (focusing on dense matrices, vectors, and linear algebra routines), serving as the Stage 1 oracle reference for Leto.

## 1. Matrix Representation and Typenames
- `Matrix<T, R, C, S>`: Core matrix type parameterized by scalar `T`, static or dynamic row dimension `R`, static or dynamic column dimension `C`, and storage buffer type `S`.
- `DMatrix<T>`: Alias for dynamic-dimension owned matrix `Matrix<T, Dyn, Dyn, VecStorage<T, Dyn, Dyn>>`.
- `DVector<T>`: Alias for dynamic-dimension owned vector `Matrix<T, Dyn, Const<1>, VecStorage<T, Dyn, Const<1>>>`.
- `OMatrix<T, R, C>`: Alias for generic owned matrix structure.
- `MatrixView<T, R, C, RStride, CStride>`: Borrowed shared read-only view of sub-matrix.
- `MatrixViewMut<T, R, C, RStride, CStride>`: Borrowed exclusive mutable view of sub-matrix.
- Type aliases for static dimensions: `Matrix1` to `Matrix6`, `Vector1` to `Vector6`, `RowVector1` to `RowVector6`.

## 2. Constructors
- `Matrix::zeros()` / `DMatrix::zeros(rows, cols)`: Construct matrix filled with zero.
- `Matrix::repeat(elem)` / `DMatrix::repeat(rows, cols, elem)`: Construct matrix filled with clones of `elem`.
- `Matrix::from_element(elem)`: Constructor alias for `repeat`.
- `Matrix::identity()` / `DMatrix::identity(rows, cols)`: Construct identity matrix.
- `Matrix::from_fn(f)` / `DMatrix::from_fn(rows, cols, f)`: Construct matrix using element coordinate generator closure.
- `Matrix::from_iterator(iter)`: Construct from iterator.
- `Matrix::from_row_slice(rows, cols, slice)`: Construct from flat row-major slice.
- `Matrix::from_column_slice(rows, cols, slice)`: Construct from flat column-major slice.
- `Matrix::from_vec(vec)`: Construct matrix from owned vector.

## 3. Properties and Accessors
- `.nrows()`: Return number of rows.
- `.ncols()`: Return number of columns.
- `.shape()`: Return shape as `(rows, cols)` tuple.
- `.len()`: Return total element count.
- `.is_empty()`: Return true if matrix contains no elements.
- `.as_slice()` / `.as_mut_slice()`: Borrow elements as flat column-major slice.
- `.as_ptr()` / `.as_mut_ptr()`: Return raw pointers to first element.

## 4. Indexing, Slicing and Views
- `Index` and `IndexMut` traits: `matrix[(row, col)]` or `matrix[index]` (returns element/ref).
- `.slice(start, shape)` / `.slice_mut(start, shape)`: Borrow dynamic-size sub-matrix view.
- `.rows(start, nrows)` / `.columns(start, ncols)`: Borrow dynamic-size slice of consecutive rows or columns.
- `.fixed_slice::<R, C>(start)` / `.fixed_slice_mut::<R, C>(start)`: Borrow static-size sub-matrix view.
- `.diagonal()` / `.diagonal_mut()`: Return diagonal elements as a vector / mutable view.

## 5. Basic Operators & Transformations
- `Add`, `Sub`, `Mul`, `Div` implementations for:
  - `Matrix op Matrix` (addition and subtraction are elementwise; multiplication is matrix product).
  - `Matrix op Scalar` / `Scalar op Matrix`
  - Reference and mutable variants of the above.
- `Neg` trait for element negation.
- `.transpose()` / `.transpose_mut()`: Return transposed matrix / transpose in place.
- `.adjoint()` / `.conjugate()`: Hermetian adjoint and elementwise conjugation.

## 6. Reductions and Norms
- `.sum()` / `.product()`: Accumulate all elements.
- `.norm()`: L2 (Frobenius) norm.
- `.norm_squared()`: Sum of squared element absolute values.
- `.lp_norm(p)`: Lp norm.
- `.metric_distance(other)`: Euclidean distance between two matrices.

## 7. Matrix Functions
- `.pow(k)`: Compute matrix integer power `A^k` using exponentiation by squaring.
- `.exp()`: Compute matrix exponential `e^A` via scaling and squaring combined with Padé approximant.

## 8. Linear Algebra and Decompositions
- `.determinant()`: Compute determinant of square matrix.
- `.try_inverse()` / `.try_inverse_to(out)`: Compute matrix inverse; returns `None` if singular.
- `.pseudo_inverse(epsilon)`: Compute Moore-Penrose pseudo-inverse.
- `.lu()` / `.full_piv_lu()`: LU decomposition with partial / complete pivoting.
- `.qr()` / `.col_piv_qr()`: QR decomposition with column pivoting.
- `.cholesky()`: Cholesky decomposition of symmetric positive-definite matrix.
- `.svd(compute_u, compute_v)`: Singular Value Decomposition.
- `.symmetric_eigen()`: Symmetric eigenvalue decomposition (Jacobi method).
- `.complex_eigenvalues()`: Non-symmetric eigenvalues (shifted QR).
- `.schur()`: Schur decomposition (Francis double-shift QR).
- `.hessenberg()`: Hessenberg reduction.
- `.bidiagonal()`: Bidiagonalization (Golub-Kahan Householder).
- `.udu()` / `.bunch_kaufman()`: Symmetric indefinite LDL decompositions.

## 9. Geometry Module
- `Rotation<T, D>` / `Rotation3<T>`: Orthonormal rotation matrices.
- `Translation<T, D>` / `Translation3<T>`: Translation vectors.
- `Isometry<T, D>` / `Isometry3<T>`: Rigid transformations (translation + rotation).
- `Similarity<T, D>` / `Similarity3<T>`: Similarity transformations (translation + rotation + scale).
- `Quaternion<T>` / `UnitQuaternion<T>`: Quaternions for 3D rotations.
- `Perspective3<T>` / `Orthographic3<T>`: 3D projection transformations.
