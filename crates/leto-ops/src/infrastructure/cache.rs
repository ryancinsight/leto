//! Cache geometry discovery for cache-aware kernel policy.

use std::sync::LazyLock;

/// Conservative L1 data-cache byte count used when topology detection is unavailable.
pub const FALLBACK_L1_BYTES: usize = 32 * 1024;
/// Conservative L2 data-cache byte count used when topology detection is unavailable.
pub const FALLBACK_L2_BYTES: usize = 256 * 1024;
/// Fallback shared last-level-cache byte count used when topology detection is
/// unavailable. Unlike the per-core L1/L2 fallbacks (deliberately small so tile
/// kernels fit the *smallest* likely cache), this is a mid-range desktop LLC
/// (8 MiB): the parallel-threshold policy keeps a bandwidth-bound op serial
/// until its working set exceeds this, and a mid estimate avoids both the
/// over-parallelization of a too-small value and the missed parallelism of a
/// too-large one when detection fails. Real topology overrides it.
pub const FALLBACK_L3_BYTES: usize = 8 * 1024 * 1024;
/// Cache-line byte count used by current line-tiling kernels.
pub const FALLBACK_CACHE_LINE_BYTES: usize = 64;

/// CPU cache geometry used to derive cache-aware kernel tile shapes.
///
/// Evidence tier: type-level contract plus value-semantic unit tests. With the
/// `topology` feature enabled, L1/L2 capacities are read from `themis`
/// `CacheLevel` values; otherwise the documented
/// fallback constants are returned. The production convenience route retains
/// the measured 32-row specialization; the explicit geometry-derived policy is
/// consumed by both dense C×C and generic row-block routes and remains available
/// for hardware-specific evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheGeometry {
    l1_bytes: usize,
    l2_bytes: usize,
    l3_bytes: usize,
    cache_line_bytes: usize,
}

impl CacheGeometry {
    /// Returns conservative fallback geometry with no runtime topology query.
    #[must_use]
    pub const fn fallback() -> Self {
        Self {
            l1_bytes: FALLBACK_L1_BYTES,
            l2_bytes: FALLBACK_L2_BYTES,
            l3_bytes: FALLBACK_L3_BYTES,
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

    /// Shared last-level (L3) cache capacity in bytes — the granularity at which
    /// additional cores contribute aggregate memory bandwidth, so the threshold
    /// past which a bandwidth-bound elementwise op benefits from parallelism.
    #[must_use]
    pub const fn l3_bytes(self) -> usize {
        self.l3_bytes
    }

    /// Cache-line width in bytes.
    #[must_use]
    pub const fn cache_line_bytes(self) -> usize {
        self.cache_line_bytes
    }
}

/// Selects the row-block shape for the existing matmul kernel family.
///
/// The selector is deliberately conservative: it uses at most one quarter of
/// the detected L2 capacity for output rows, caps the block at the measured
/// `32`-row specialization, and rounds down to a power of two. This preserves
/// the existing kernel model while preventing very wide output rows from
/// overfilling a small L2. It is a policy contract, not a performance claim;
/// callers must benchmark a changed policy before claiming a speedup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatmulTilePolicy {
    row_block: usize,
}

impl MatmulTilePolicy {
    /// Construct a validated explicit row-block policy.
    ///
    /// Only the existing power-of-two kernel specializations are accepted.
    #[must_use]
    pub const fn fixed(row_block: usize) -> Option<Self> {
        match row_block {
            1 | 2 | 4 | 8 | 16 | 32 => Some(Self { row_block }),
            _ => None,
        }
    }

    /// Select the current automatic policy from the process topology cache.
    #[must_use]
    pub fn automatic(element_bytes: usize, cols: usize) -> Self {
        Self::for_geometry(cached_cache_geometry(), element_bytes, cols)
    }

    /// Select a conservative row block for `cols` output columns.
    #[must_use]
    pub fn for_geometry(geometry: CacheGeometry, element_bytes: usize, cols: usize) -> Self {
        let row_bytes = cols.saturating_mul(element_bytes.max(1));
        if row_bytes == 0 {
            return Self { row_block: 1 };
        }

        let budget = geometry.l2_bytes() / 4;
        let capacity = (budget / row_bytes).clamp(1, 32);
        let row_block = if capacity >= 32 {
            32
        } else {
            let next_power = capacity.next_power_of_two();
            if next_power == capacity {
                capacity
            } else {
                next_power / 2
            }
        }
        .max(1);

        Self { row_block }
    }

    /// Number of output rows processed by one kernel block.
    #[must_use]
    pub const fn row_block(self) -> usize {
        self.row_block
    }
}

/// Returns the active cache geometry policy (re-detects on every call).
#[must_use]
pub fn cache_geometry() -> CacheGeometry {
    CacheGeometry::detect()
}

/// The process-wide cache geometry, detected once on first use.
///
/// [`cache_geometry`] re-detects on every call (a topology syscall / sysfs
/// read); hot-path policy such as the parallel-threshold decision reads this
/// cached value instead. Cache geometry is fixed for the process lifetime, so a
/// single detection suffices.
#[must_use]
pub fn cached_cache_geometry() -> CacheGeometry {
    static GEOMETRY: LazyLock<CacheGeometry> = LazyLock::new(CacheGeometry::detect);
    *GEOMETRY
}

#[cfg(feature = "topology")]
fn detect_cache_geometry() -> CacheGeometry {
    let Some(topology) = themis::CpuTopology::detect() else {
        return CacheGeometry::fallback();
    };
    geometry_from_cache_levels(topology.cache_levels().unwrap_or(&[]))
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
            3 if level.size_bytes > 0 => geometry.l3_bytes = level.size_bytes,
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
        assert_eq!(geometry.l3_bytes(), 8 * 1024 * 1024);
        assert_eq!(geometry.cache_line_bytes(), 64);
    }

    #[test]
    fn matmul_policy_preserves_measured_default_for_common_shapes() {
        let geometry = CacheGeometry::fallback();

        assert_eq!(
            MatmulTilePolicy::for_geometry(geometry, 8, 64).row_block(),
            32
        );
        assert_eq!(
            MatmulTilePolicy::for_geometry(geometry, 8, 256).row_block(),
            32
        );
    }

    #[test]
    fn matmul_policy_downsizes_wide_rows_without_exceeding_bounds() {
        let geometry = CacheGeometry::fallback();

        assert_eq!(
            MatmulTilePolicy::for_geometry(geometry, 8, 1_024).row_block(),
            8
        );
        assert_eq!(
            MatmulTilePolicy::for_geometry(geometry, 8, 2_048).row_block(),
            4
        );
        assert_eq!(
            MatmulTilePolicy::for_geometry(geometry, 8, 1).row_block(),
            32
        );
    }

    #[test]
    fn fixed_matmul_policy_accepts_only_supported_specializations() {
        assert_eq!(
            MatmulTilePolicy::fixed(32).map(MatmulTilePolicy::row_block),
            Some(32)
        );
        assert_eq!(MatmulTilePolicy::fixed(3), None);
        assert_eq!(MatmulTilePolicy::fixed(64), None);
    }

    #[test]
    fn matmul_policy_handles_empty_and_tiny_cache_inputs() {
        assert_eq!(
            MatmulTilePolicy::for_geometry(CacheGeometry::fallback(), 0, 0).row_block(),
            1
        );

        let tiny = CacheGeometry {
            l1_bytes: 4 * 1024,
            l2_bytes: 32 * 1024,
            l3_bytes: 256 * 1024,
            cache_line_bytes: 64,
        };
        assert_eq!(MatmulTilePolicy::for_geometry(tiny, 8, 256).row_block(), 4);
    }

    #[cfg(feature = "topology")]
    #[test]
    fn cache_levels_override_l1_l2_l3_without_allocating_copies() {
        let levels = [
            themis::CacheLevel {
                level: 1,
                size_bytes: 48 * 1024,
                line_bytes: Some(FALLBACK_CACHE_LINE_BYTES),
                shared_processors: [0, 1].into(),
            },
            themis::CacheLevel {
                level: 2,
                size_bytes: 1024 * 1024,
                line_bytes: Some(FALLBACK_CACHE_LINE_BYTES),
                shared_processors: [0, 1].into(),
            },
            themis::CacheLevel {
                level: 3,
                size_bytes: 32 * 1024 * 1024,
                line_bytes: Some(FALLBACK_CACHE_LINE_BYTES),
                shared_processors: [0, 1, 2, 3].into(),
            },
        ];

        let geometry = geometry_from_cache_levels(&levels);

        assert_eq!(geometry.l1_bytes(), 48 * 1024);
        assert_eq!(geometry.l2_bytes(), 1024 * 1024);
        assert_eq!(geometry.l3_bytes(), 32 * 1024 * 1024);
        assert_eq!(geometry.cache_line_bytes(), FALLBACK_CACHE_LINE_BYTES);
    }

    #[cfg(feature = "topology")]
    #[test]
    fn zero_sized_or_unknown_cache_levels_keep_fallbacks() {
        let levels = [
            // Zero-sized L1 (reported but unusable) keeps the L1 fallback.
            themis::CacheLevel {
                level: 1,
                size_bytes: 0,
                line_bytes: None,
                shared_processors: [0].into(),
            },
            // A level the policy does not model (4) is ignored, keeping every
            // fallback — including the L3 fallback the geometry still reports.
            themis::CacheLevel {
                level: 4,
                size_bytes: 128 * 1024 * 1024,
                line_bytes: None,
                shared_processors: [0].into(),
            },
        ];

        let geometry = geometry_from_cache_levels(&levels);

        assert_eq!(geometry, CacheGeometry::fallback());
    }
}
