/// CPU cache geometry for cache-aware kernel policy.
pub mod cache;

#[cfg(feature = "parallel")]
/// Parallel loop scheduling.
pub mod parallel;
/// SIMD operation forwarding.
pub mod simd;
