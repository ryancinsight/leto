#![warn(missing_docs)]
//! Leto Operations contains performance-tuned math and reduction kernels for Leto arrays.

/// Application-level operation entry points.
pub mod application;
/// Operation scalar and strategy contracts.
pub mod domain;
/// SIMD and parallel execution infrastructure.
pub mod infrastructure;

pub use domain::real::RealScalar;
pub use domain::rng::Xorshift64;
pub use domain::scalar::Scalar;
pub use domain::strategy::{ExecutionStrategy, ScalarStrategy};
pub use infrastructure::cache::{
    cache_geometry, cached_cache_geometry, CacheGeometry, MatmulTilePolicy,
};

pub use domain::strategy::SimdStrategy;

#[cfg(feature = "parallel")]
pub use domain::strategy::ParallelStrategy;

#[cfg(feature = "parallel")]
pub use infrastructure::parallel::{for_each_task_partition_mut, for_each_task_partition_mut_with};

pub use application::attention::{
    scaled_dot_product_attention_backward_accumulate, scaled_dot_product_attention_into,
    AttentionError, AttentionGradients, AttentionMask, AttentionOperand, AttentionResult,
    GroupedKeepMask,
};
pub use application::convolution::{
    convolution_backward_accumulate, convolution_forward_into,
    convolution_transposed_backward_accumulate, convolution_transposed_forward_into,
    ConvolutionParameters, TransposedConvolutionGradients, TransposedConvolutionParameters,
};
pub use application::layout::{
    transpose_complex_matrices, transpose_square_inplace, SquareTransposeError,
};
pub use application::linalg::{
    bidiagonalize,
    bunch_kaufman,
    cholesky_decompose,
    cholesky_det,
    cholesky_inv,
    cholesky_solve,
    col_piv_qr,
    // ── Complex linear algebra ────────────────────────────────────────────────
    complex_inv,
    complex_solve,
    det,
    eigenvalues,
    full_piv_lu,
    // ── Hermitian eigensolvers ─────────────────────────────────────────────────
    hermitian_eigen_jacobi,
    hermitian_eigen_qr,
    hessenberg,
    inv,
    kron,
    l2_normalize,
    l2_normalize_into,
    lu_decompose,
    matexp,
    matpow,
    matrix_rank,
    matrix_rank_with_tolerance,
    nnls,
    norm,
    norm_l1,
    norm_l2,
    norm_max,
    pinv,
    qr_decompose,
    schur,
    singular_values,
    solve,
    solve_least_squares,
    svd_decompose,
    symmetric_eigen_jacobi,
    symmetric_eigen_jacobi_with_tolerance,
    symmetric_eigenvalues_jacobi,
    symmetric_eigenvalues_jacobi_with_tolerance,
    trace,
    udu_decompose,
    AsMatrixView,

    BidiagonalDecomposition,
    BunchKaufmanDecomposition,
    CholeskyDecomposition,
    ColPivQrDecomposition,

    FullPivLuDecomposition,
    HermitianEigenConfig,
    HermitianEigenResult,
    HessenbergDecomposition,

    LuDecomposition,
    MatrixDecompose,
    MatrixFunction,
    MatrixNorm,
    MatrixProduct,
    MatrixProperties,
    MatrixSolve,
    NnlsConfig,
    NnlsResult,
    NormKind,
    NormL1,
    NormL2,
    NormMax,

    QrDecomposition,
    RealSchur,

    SvdDecomposition,
    SymmetricEigenDecomposition,
    UduDecomposition,
};
pub use application::loss::{
    cross_entropy_backward_accumulate, cross_entropy_forward_into, CrossEntropyError,
    CrossEntropyOperand, CrossEntropyResult,
};
pub use application::map::{
    add, binary_map, binary_map_with_cache_geometry, div, mul, scalar_map, scalar_map_into, sub,
    sum, AddOp, BinaryOp, DivOp, EqOp, GeOp, GtOp, LeOp, LtOp, MulOp, NeOp, SubOp,
};
pub use application::matrix::{batched_matmul, matmul, matmul_accumulate, matmul_with_tile_policy};
pub use application::nonlinear::{AndersonAccelerator, AndersonConfig, AndersonMethod};
pub use application::optimization::{minimize, LbfgsConfig, LbfgsMemory, LbfgsResult};
pub use application::pooling::{pooling_backward_accumulate, pooling_forward_into, PoolingMode};
pub use application::random::{
    normal_with_seed, normal_with_seed_into, uniform_with_seed, uniform_with_seed_into,
};
pub use application::reduction::{
    max, max_axis, max_axis_into, mean_axis, mean_axis_into, min, min_axis, min_axis_into,
    product_axis, product_axis_into, reduce_all, reduce_axis, reduce_axis_into, sum_axis,
    sum_axis_into, AxisReduction, MaxAxis, MeanAxis, MinAxis, ProductAxis, SumAxis,
};
pub use application::scan::{
    cumsum, cumsum_into, scan_axis, scan_axis_into, CumProdOp, CumSumOp, ScanDirection, ScanOp,
};
pub use application::signal::{blackman, hamming, hann, tukey, wrap_to_pi};
pub use application::sliding_window::{fold_into, unfold_into};
pub use application::sparse::{
    csc_spmv, csc_spmv_into, csr_to_dense, factor_numeric, factor_symbolic,
    factor_symbolic_with_ordering, sparse_lu_solve, spgemm, spmm, spmm_into, spmv, spmv_into,
    CooMatrix, CscColumn, CscMatrix, CsrMatrix, CsrRow, NumericLu, OrderingStrategy,
    OwnedNumericLu, SparseLuSolver, SymbolicLu, DENSE_LIMIT_DEFAULT,
};
/// Special mathematical functions (sinc, erf, Bessel J₀/J₁/Jₙ, Legendre polynomials).
pub use application::special::{erf, j0, j1, jn, sinc};
pub use application::special_legendre::{legendre_poly, legendre_poly_and_deriv};
pub use application::stateful_update::{
    stateful_update, AdaGrad, AdaGradParameters, Adam, AdamParameters, AdamW, AdamWParameters,
    RmsProp, RmsPropParameters, Sgd, SgdParameters, StatefulUpdateRule,
};
pub use application::statistics::{
    normalized_rmse, nrmse, pearson, percentile_range, phase_error_degrees_for_correlation,
    phase_shift_correlation_curve, psnr, rmse, validation_psnr_from_relative_rmse,
};
pub use application::stencil::laplacian_2d_into;
pub use application::unary::{
    map, map_inplace, map_into, map_into_with_cache_geometry, mapv, unary_map, unary_map_into,
    AbsOp, CosOp, ErfOp, ErfcOp, ExpOp, LgammaOp, LnOp, NegOp, PowfOp, RecipOp, SinOp, SqrtOp,
    UnaryOp,
};
pub use application::vector::{dot, hamming_distance, jaccard_distance, matvec};
pub use application::zip::{
    coordinate_map_inplace, coordinate_map_plan, coordinate_map_plan_inplace, indexed_fold,
    indexed_fold_fortran, indexed_map4_inplace, indexed_map_inplace, indexed_zip_mut_with,
    zip_fold, zip_mut_with, CoordinateMapPlan, IndexedZipMutOutputs, ZipMutOutputs, ZipSources,
};

// ── Interpolation (SSOT) ──────────────────────────────────────────────────────
/// 2-D and 3-D spatial interpolation (bilinear, trilinear) in index space.
pub use application::interpolation::{
    bilinear, bilinear_index_space, trilinear, trilinear_index_space,
};
/// 1-D interpolation trait and implementations.
pub use application::interpolation::{
    CubicSplineInterpolation, Interpolation1D, LagrangeInterpolation, LinearInterpolation,
};

// ── Finite-difference differentiation (SSOT) ─────────────────────────────────
/// Derived staggered/collocated first-derivative taps to arbitrary even order.
pub use application::diff::{
    central_first_derivative_coefficients, staggered_first_derivative_coefficients,
    TapCoefficients, MAX_HALF_ORDER,
};
/// Arbitrary-even-order staggered gradient/divergence pair for a Yee leapfrog.
pub use application::diff::{Axis, StaggeredLeapfrog3D};
/// Generic 1-D finite-difference operator.
pub use application::diff::{FiniteDifference, FiniteDifferenceScheme};
/// Generic 3-D finite-difference operator (provider-SSOT for kwavers/CFDrs/helios
/// first-derivative kernels: central 2nd/4th/6th + Yee staggered forward/backward).
pub use application::diff::{FiniteDifference3D, FiniteDifference3DScheme};

// ── Quadrature (SSOT) ─────────────────────────────────────────────────────────
/// Numerical quadrature (integration) rules.
pub use application::quadrature::{
    gauss_legendre_nodes_weights, CompositeQuadrature, GaussLegendre2, GaussLegendre3,
    GaussLegendre5, GaussLegendreN, Quadrature, SimpsonsRule, TrapezoidalRule, GL3_NODES,
    GL3_NODES_UNIT, GL3_WEIGHTS, GL3_WEIGHTS_UNIT,
};
