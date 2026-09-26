//! The plane rotations of the sweep: the scale-safe Givens rotation and its
//! contiguous application to a transposed factor.

use crate::application::linalg::scaling::KernelWindow;
use crate::domain::real::RealScalar;

/// Givens rotation `(c, s, r)` with `c·a + s·b = r`, `−s·a + c·b = 0`.
///
/// Returns the rotated lead component `r = c·a + s·b` (the 2-norm `√(a²+b²)`
/// when `b ≠ 0`, else `a`) so the caller uses it directly instead of recomputing
/// `c·a + s·b` — `givens` already formed `√(a²+b²)` to normalize, so re-deriving
/// it at the call site is pure redundant arithmetic (the leto `cancel_y`
/// pattern, which returns the norm alongside the rotation).
///
/// Scale-safe as LAPACK `dlartg` (3.10, Anderson 2017): `a² + b² ≤ 2m²`,
/// `m = max(|a|, |b|)`, is formed unscaled while `m` lies in
/// `[√safmin, √(Ω/2)]` — `dlartg`'s `rtmin`/`rtmax` — and otherwise from
/// `a, b` divided by the power of two bringing `m` into `[1, 2)`. `c`, `s`
/// are scale-invariant and `r` is multiplied back, all exactly, so inside the
/// window the result is bit-for-bit the unscaled formula. `window` is
/// [`SweepWindows::rotation`](super::sweep::SweepWindows::rotation).
#[inline]
pub(super) fn givens<T: RealScalar>(a: T, b: T, window: KernelWindow<T>) -> (T, T, T) {
    if b == T::ZERO {
        return (T::ONE, T::ZERO, a);
    }
    let exponent = window.exponent(&[a, b]);
    let (a, b) = (a.scale_binary(-exponent), b.scale_binary(-exponent));
    // One reciprocal + two mults rather than two divides by the same `r`
    // (division is several× a multiply; the sweep calls this O(n²) times). The
    // extra rounding of `1/r` is within the SVD's differential tolerance.
    let r = a.mul(a).add(b.mul(b)).sqrt();
    let inv_r = T::ONE.div(r);
    (a.mul(inv_r), b.mul(inv_r), r.scale_binary(exponent))
}

/// Apply the Givens `first' = c·first + s·second`, `second' = c·second − s·first`
/// to the two rows `(first, second)` of length `len` in a row-major matrix.
///
/// The pair need not be adjacent and need not be ordered (`first > second` is a
/// rotation with the roles swapped, which the zero-diagonal chase below needs);
/// each row is a contiguous `len`-slice either way.
///
/// # Theorem (transposed accumulation is bitwise-identical, and contiguous)
/// Accumulating `U` (or `V`) as its transpose `Uᵀ` while rotating two **rows** of
/// `Uᵀ` produces exactly the same factor `U` as rotating two **columns** of `U`,
/// bit for bit, and turns the strided column update into a contiguous one.
///
/// *Proof.* A plane rotation applied to columns `(k, k+1)` of `U` is the
/// right-multiplication `U ← U G` with `G` the embedded `2×2` Givens. Transposing,
/// `(U G)ᵀ = Gᵀ Uᵀ`, i.e. `Uᵀ ← Gᵀ Uᵀ`, which is a left-multiplication mixing
/// **rows** `(k, k+1)` of `Uᵀ` with the same scalar coefficients `(c, s)`. The two
/// updated entries in each position are the identical floating-point expression
/// `c·a + s·b` / `c·b − s·a` of the identical operands `a, b`, evaluated in the
/// identical order — so no rounding differs: the stored `Uᵀ` is the exact transpose
/// of the column-accumulated `U`. In row-major storage a row of `Uᵀ` is a
/// contiguous length-`len` slice, so the rotation is two contiguous disjoint
/// slices (cache-friendly, auto-vectorizable) instead of a stride-`len` column walk
/// (a cache line per element). The factors are recovered by reading rows of
/// `Uᵀ`/`Vᵀ` as columns of `U`/`V` at the `O(n²)` thin-extraction step — no
/// separate transpose pass — negligible against the `O(n³)` sweep. ∎
///
/// `U`/`V` are accumulated as `Uᵀ`/`Vᵀ` so every rotation hits this path.
#[inline]
pub(super) fn rotate_row_pair<T: RealScalar>(
    mat: &mut [T],
    len: usize,
    first: usize,
    second: usize,
    c: T,
    s: T,
) {
    debug_assert!(first != second, "a Givens rotation mixes two distinct rows");
    let split = first.max(second);
    let (head, tail) = mat.split_at_mut(split * len);
    let low = &mut head[first.min(second) * len..][..len];
    let high = &mut tail[..len];
    let (row_a, row_b) = if first < second {
        (low, high)
    } else {
        (high, low)
    };
    for (a, b) in row_a.iter_mut().zip(row_b.iter_mut()) {
        let (va, vb) = (*a, *b);
        *a = c.mul(va).add(s.mul(vb));
        *b = c.mul(vb).sub(s.mul(va));
    }
}
