/// CPU cache geometry for cache-aware kernel policy.
pub mod cache;

#[cfg(feature = "parallel")]
/// Parallel loop scheduling.
pub mod parallel;
#[cfg(feature = "simd")]
/// SIMD operation forwarding.
pub mod simd;
