/// Deterministic `xorshift64*` pseudo-random generator.
///
/// A small reproducible PRNG for seeded array initialization. It is not
/// cryptographically secure; it exists so model/weight initialization is
/// deterministic given a seed, matching Coeus init semantics. The state is a
/// single non-zero `u64`; a seed of zero is remapped to a fixed non-zero
/// constant because `xorshift` degenerates at zero.
#[derive(Clone, Copy, Debug)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Golden-ratio-derived constant used to remap a zero seed.
    const SEED_FALLBACK: u64 = 0x9E37_79B9_7F4A_7C15;

    /// Create a generator from a seed. Seed `0` is remapped to a fixed non-zero
    /// constant so the stream is always well-defined.
    #[inline]
    pub const fn new(seed: u64) -> Self {
        let state = if seed == 0 { Self::SEED_FALLBACK } else { seed };
        Self { state }
    }

    /// Advance the state and return the next 64-bit value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Return the next uniform `f64` in the half-open interval `[0, 1)`.
    ///
    /// Uses the top 53 bits so every representable `f64` mantissa is reachable.
    #[inline]
    pub fn next_unit_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        bits as f64 * (1.0 / (1u64 << 53) as f64)
    }
}
