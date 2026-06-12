//! Cache geometry discovery for cache-aware kernel policy.

/// Conservative L1 data-cache byte count used when topology detection is unavailable.
pub const FALLBACK_L1_BYTES: usize = 32 * 1024;
/// Conservative L2 data-cache byte count used when topology detection is unavailable.
pub const FALLBACK_L2_BYTES: usize = 256 * 1024;
/// Cache-line byte count used by current line-tiling kernels.
pub const FALLBACK_CACHE_LINE_BYTES: usize = 64;

/// CPU cache geometry used to derive cache-aware kernel tile shapes.
///
/// Evidence tier: type-level contract plus value-semantic unit tests. With the
/// `topology` feature enabled, L1/L2 capacities are read from `themis`
/// `CacheLevel` values; otherwise the documented
/// fallback constants are returned. Current hot kernels keep their existing
/// compile-time tile constants until a benchmarked blocking policy consumes
/// this geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheGeometry {
    l1_bytes: usize,
    l2_bytes: usize,
    cache_line_bytes: usize,
}

impl CacheGeometry {
    /// Returns conservative fallback geometry with no runtime topology query.
    #[must_use]
    pub const fn fallback() -> Self {
        Self {
            l1_bytes: FALLBACK_L1_BYTES,
            l2_bytes: FALLBACK_L2_BYTES,
            cache_line_bytes: FALLBACK_CACHE_LINE_BYTES,
        }
    }

    /// Returns detected cache geometry when `topology` is enabled, otherwise fallbacks.
    #[must_use]
    pub fn detect() -> Self {
        detect_cache_geometry()
    }

    /// L1 data-cache capacity in bytes.
    #[must_use]
    pub const fn l1_bytes(self) -> usize {
        self.l1_bytes
    }

    /// L2 cache capacity in bytes.
    #[must_use]
    pub const fn l2_bytes(self) -> usize {
        self.l2_bytes
    }

    /// Cache-line width in bytes.
    #[must_use]
    pub const fn cache_line_bytes(self) -> usize {
        self.cache_line_bytes
    }
}

/// Returns the active cache geometry policy.
#[must_use]
pub fn cache_geometry() -> CacheGeometry {
    CacheGeometry::detect()
}

#[cfg(feature = "topology")]
fn detect_cache_geometry() -> CacheGeometry {
    let Some(topology) = themis::CpuTopology::detect() else {
        return CacheGeometry::fallback();
    };
    geometry_from_cache_levels(topology.cache_levels())
}

#[cfg(not(feature = "topology"))]
fn detect_cache_geometry() -> CacheGeometry {
    CacheGeometry::fallback()
}

#[cfg(feature = "topology")]
fn geometry_from_cache_levels(levels: &[themis::CacheLevel]) -> CacheGeometry {
    let mut geometry = CacheGeometry::fallback();
    for level in levels {
        match level.level {
            1 if level.size_bytes > 0 => geometry.l1_bytes = level.size_bytes,
            2 if level.size_bytes > 0 => geometry.l2_bytes = level.size_bytes,
            _ => {}
        }
    }
    geometry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_geometry_is_conservative() {
        let geometry = CacheGeometry::fallback();

        assert_eq!(geometry.l1_bytes(), 32 * 1024);
        assert_eq!(geometry.l2_bytes(), 256 * 1024);
        assert_eq!(geometry.cache_line_bytes(), 64);
    }

    #[cfg(feature = "topology")]
    #[test]
    fn cache_levels_override_l1_l2_without_allocating_copies() {
        let levels = [
            themis::CacheLevel {
                level: 1,
                size_bytes: 48 * 1024,
                shared_processors: [0, 1].into(),
            },
            themis::CacheLevel {
                level: 2,
                size_bytes: 1024 * 1024,
                shared_processors: [0, 1].into(),
            },
        ];

        let geometry = geometry_from_cache_levels(&levels);

        assert_eq!(geometry.l1_bytes(), 48 * 1024);
        assert_eq!(geometry.l2_bytes(), 1024 * 1024);
        assert_eq!(geometry.cache_line_bytes(), FALLBACK_CACHE_LINE_BYTES);
    }

    #[cfg(feature = "topology")]
    #[test]
    fn zero_sized_or_unknown_cache_levels_keep_fallbacks() {
        let levels = [
            themis::CacheLevel {
                level: 1,
                size_bytes: 0,
                shared_processors: [0].into(),
            },
            themis::CacheLevel {
                level: 3,
                size_bytes: 8 * 1024 * 1024,
                shared_processors: [0].into(),
            },
        ];

        let geometry = geometry_from_cache_levels(&levels);

        assert_eq!(geometry, CacheGeometry::fallback());
    }
}
