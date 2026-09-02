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
/// Cache-line byte count used when no cache level reports a line width.
///
/// 64 bytes is the *narrowest* line width in current mainstream targets
/// (x86-64, and the aarch64 parts that do not use 128), which is the
/// conservative direction for the line-tiling this value feeds: a micro-tile
/// derived from a 64-byte line is fully consumed on a 64-byte part and merely
/// touches each 128-byte line twice on a wider one, whereas assuming 128 on a
/// 64-byte part quadruples a two-dimensional tile's working set and can
/// overflow the L1 budget. It is not a false-sharing padding width — a
/// consumer needing that must read the platform value and handle its absence
/// explicitly rather than inherit this floor.
pub const FALLBACK_CACHE_LINE_BYTES: usize = 64;

/// CPU cache geometry used to derive cache-aware kernel tile shapes.
///
/// Evidence tier: type-level contract plus value-semantic unit tests. With the
/// `topology` feature enabled, L1/L2/L3 capacities and the cache-line width are
/// read from `themis` `CacheLevel` values; otherwise the documented
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

    /// Returns fallback capacities with an explicitly known cache-line width.
    ///
    /// This is useful when a caller has topology information from outside
    /// [`themis`] or when evaluating a cache-line policy on a target that is
    /// not the current host. The capacity fields remain conservative fallbacks
    /// because this constructor only establishes the line-width policy.
    #[must_use]
    pub const fn with_cache_line_bytes(cache_line_bytes: usize) -> Option<Self> {
        if cache_line_bytes == 0 {
            return None;
        }

        let fallback = Self::fallback();
        Some(Self {
            cache_line_bytes,
            ..fallback
        })
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
    // `themis` reports a per-level line width as typed absence: `None` when the
    // platform exposes no `coherency_line_size` / `CacheRelationship::LineSize`.
    // That absence is carried here as an `Option` and resolved exactly once,
    // after the scan, so the fallback constant cannot silently stand in for a
    // reported value at any individual level.
    let mut reported_line_bytes: Option<usize> = None;
    // Smallest capacity reported at each level, carried as typed absence for the
    // same reason the line width is: a fallback constant must not compete with
    // reported values during the scan. Taking `min` against the fallback
    // directly would clamp every real cache down to it — `fallback`'s 256 KiB L2
    // would beat this host's 3 MiB.
    //
    // Smallest, rather than the last entry scanned, because a hybrid part
    // reports one entry per distinct cache and a level therefore arrives
    // several times with different sizes: this host reports a 48 KiB P-core L1d
    // beside a 32 KiB E-core L1d, and a 3 MiB private L2 beside a 4 MiB
    // per-cluster one. Assigning unconditionally made the result depend on
    // enumeration order, which is not a policy.
    //
    // A task does not choose its core — under work stealing it may run on
    // either class — so a tile sized to the larger cache overflows on the
    // smaller, while the smallest fits everywhere. This mirrors the widest-line
    // rule below: both resolve a multi-valued report by stating which value the
    // policy wants.
    let mut reported: [Option<usize>; 3] = [None; 3];
    for level in levels {
        let slot = level
            .level
            .checked_sub(1)
            .and_then(|index| usize::try_from(index).ok())
            .and_then(|index| reported.get_mut(index));
        if let Some(slot) = slot.filter(|_| level.size_bytes > 0) {
            *slot = Some(slot.map_or(level.size_bytes, |smallest| smallest.min(level.size_bytes)));
        }
        // Widest line reported by any level, including levels whose capacity
        // this policy does not model. A tile sized to the widest line is fully
        // consumed at every level; reading only L1 under-tiles the parts that
        // motivate this at all — Apple M-series and several aarch64 server
        // parts report 128 bytes, at outer levels on some of them.
        if let Some(line) = level.line_bytes.filter(|&line| line > 0) {
            reported_line_bytes = Some(reported_line_bytes.map_or(line, |widest| line.max(widest)));
        }
    }
    if let Some(l1) = reported[0] {
        geometry.l1_bytes = l1;
    }
    if let Some(l2) = reported[1] {
        geometry.l2_bytes = l2;
    }
    if let Some(l3) = reported[2] {
        geometry.l3_bytes = l3;
    }
    geometry.cache_line_bytes = reported_line_bytes.unwrap_or(FALLBACK_CACHE_LINE_BYTES);
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
        // The narrowest mainstream line, not a guess at the local part: an
        // under-estimate keeps a derived micro-tile inside the L1 budget on
        // every target, while an over-estimate does not.
        assert_eq!(geometry.cache_line_bytes(), 64);
    }

    #[test]
    fn explicit_line_geometry_requires_a_positive_width() {
        assert_eq!(CacheGeometry::with_cache_line_bytes(0), None);

        let geometry = CacheGeometry::with_cache_line_bytes(128)
            .expect("positive cache-line width is a valid explicit policy");
        assert_eq!(geometry.cache_line_bytes(), 128);
        assert_eq!(geometry.l1_bytes(), FALLBACK_L1_BYTES);
        assert_eq!(geometry.l2_bytes(), FALLBACK_L2_BYTES);
        assert_eq!(geometry.l3_bytes(), FALLBACK_L3_BYTES);
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
    fn cache_level(level: u32, size_bytes: usize, line_bytes: Option<usize>) -> themis::CacheLevel {
        themis::CacheLevel {
            level,
            size_bytes,
            line_bytes,
            shared_processors: [0, 1].into(),
        }
    }

    #[cfg(feature = "topology")]
    #[test]
    fn heterogeneous_levels_resolve_to_the_smallest_reported_capacity() {
        // A hybrid part reports one entry per distinct cache, so a level
        // arrives several times with different sizes. This is the shape of a
        // real Arrow Lake report: a 48 KiB P-core L1d beside a 32 KiB E-core
        // L1d, and a 3 MiB private L2 beside a 4 MiB per-cluster one.
        //
        // The larger value is listed last in each pair, so a last-wins scan
        // would return it and this case would fail.
        let levels = [
            cache_level(1, 32 * 1024, Some(64)),
            cache_level(2, 3 * 1024 * 1024, Some(64)),
            cache_level(1, 48 * 1024, Some(64)),
            cache_level(2, 4 * 1024 * 1024, Some(64)),
            cache_level(3, 36 * 1024 * 1024, Some(64)),
        ];

        let geometry = geometry_from_cache_levels(&levels);

        // A tile sized to the larger cache overflows on the smaller core, and
        // a task does not choose its core under work stealing.
        assert_eq!(geometry.l1_bytes(), 32 * 1024);
        assert_eq!(geometry.l2_bytes(), 3 * 1024 * 1024);
        assert_eq!(geometry.l3_bytes(), 36 * 1024 * 1024);
    }

    #[cfg(feature = "topology")]
    #[test]
    fn a_reported_capacity_is_never_clamped_to_the_fallback() {
        // Selecting the smallest must range over reported values only. Folding
        // the fallback into the same comparison would clamp every real cache
        // down to it -- the fallback L2 is 256 KiB, far below any real one.
        let levels = [cache_level(2, 3 * 1024 * 1024, Some(64))];

        let geometry = geometry_from_cache_levels(&levels);

        assert_eq!(geometry.l2_bytes(), 3 * 1024 * 1024);
        // Levels absent from the report still fall back rather than collapsing.
        assert_eq!(geometry.l1_bytes(), CacheGeometry::fallback().l1_bytes());
        assert_eq!(geometry.l3_bytes(), CacheGeometry::fallback().l3_bytes());
    }

    #[cfg(feature = "topology")]
    #[test]
    fn cache_levels_override_capacities_and_line_width() {
        // A 128-byte-line part (Apple M-series class). Every reported field is
        // distinct from its fallback, so no assertion can pass by coincidence.
        let levels = [
            cache_level(1, 48 * 1024, Some(128)),
            cache_level(2, 1024 * 1024, Some(128)),
            cache_level(3, 32 * 1024 * 1024, Some(128)),
        ];

        let geometry = geometry_from_cache_levels(&levels);

        assert_eq!(geometry.l1_bytes(), 48 * 1024);
        assert_eq!(geometry.l2_bytes(), 1024 * 1024);
        assert_eq!(geometry.l3_bytes(), 32 * 1024 * 1024);
        assert_eq!(geometry.cache_line_bytes(), 128);
    }

    #[cfg(feature = "topology")]
    #[test]
    fn widest_reported_line_width_wins_across_levels() {
        // Mixed widths, absence, and a level the capacity policy ignores: the
        // widest *reported* line still governs, so a tile derived from it is
        // fully consumed at every level.
        let levels = [
            cache_level(1, 64 * 1024, Some(64)),
            cache_level(2, 4 * 1024 * 1024, None),
            cache_level(4, 128 * 1024 * 1024, Some(128)),
        ];

        let geometry = geometry_from_cache_levels(&levels);

        assert_eq!(geometry.cache_line_bytes(), 128);
        assert_eq!(geometry.l1_bytes(), 64 * 1024);
        assert_eq!(geometry.l2_bytes(), 4 * 1024 * 1024);
    }

    #[cfg(feature = "topology")]
    #[test]
    fn reported_line_width_narrower_than_the_fallback_is_honoured() {
        // The fallback is a floor for *absence*, never a floor for a reported
        // value: a platform reporting 32 bytes gets 32, not 64.
        let levels = [cache_level(1, 16 * 1024, Some(32))];

        let geometry = geometry_from_cache_levels(&levels);

        assert_eq!(geometry.cache_line_bytes(), 32);
    }

    #[cfg(feature = "topology")]
    #[test]
    fn zero_sized_or_unknown_cache_levels_keep_fallbacks() {
        let levels = [
            // Zero-sized L1 (reported but unusable) keeps the L1 fallback, and
            // a zero line width is absence in numeric disguise — also fallback.
            cache_level(1, 0, Some(0)),
            // A level the policy does not model (4) is ignored, keeping every
            // fallback — including the L3 fallback the geometry still reports.
            cache_level(4, 128 * 1024 * 1024, None),
        ];

        let geometry = geometry_from_cache_levels(&levels);

        assert_eq!(geometry, CacheGeometry::fallback());
        assert_eq!(geometry.cache_line_bytes(), FALLBACK_CACHE_LINE_BYTES);
    }
}
