//! The direction a sweep chases its bulge through a block: LAPACK `dbdsqr`'s
//! `IDIR`, as an index reflection so each sweep is written once.
//!
//! # Theorem (a bottom-to-top chase is a top-to-bottom chase of a reflection)
//! Let `P` reverse the indices of the block `[p, q]` (`j ↦ p + q − j`). Then
//! `B' = P Bᵀ P` is upper bidiagonal with `d'ⱼ = d_{p+q−j}` and
//! `e'ⱼ = e_{p+q−1−j}`, and `σ(B') = σ(B)`. A right rotation of columns
//! `(a, b)` of `B'` (`B' ← B' G`) is `Bᵀ ← P B' G P`, i.e. `B ← (P Gᵀ P) B`:
//! a left rotation of rows `(p+q−a, p+q−b)` of `B` with the same `(c, s)`
//! and the same `first' = c·first + s·second` form; symmetrically a left
//! rotation of `B'` is a right rotation of `B`. So `dbdsqr`'s bottom-to-top
//! loops (LAPACK 3.12.0 `dbdsqr.f`: loop 130, lines 661–695, loop 150,
//! lines 751–790, and the backward convergence test, loop 110, lines
//! 556–580) are its top-to-bottom loops (120, lines 623–657; 140, lines
//! 703–745; 100, lines 528–552)
//! run on `B'`, with the `U`/`V` roles swapped — which is how `dbdsqr`
//! applies them: its bottom-to-top sweeps update `VT` with the second
//! (`OLDCS`/`COSL`) rotation of each pair and `U` with the first. ∎
//!
//! [`Down`] is the identity orientation, [`Up`] the reflection; the sweeps
//! and convergence tests index the block only through [`Oriented`] and rotate
//! the factors only through [`Chase::rotate_columns`] /
//! [`Chase::rotate_rows`], and monomorphize once per direction.

use super::rotation::TransposedFactors;
use crate::domain::real::RealScalar;
use core::marker::PhantomData;

/// The two chase directions, selected at run time once per block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Direction {
    /// Top to bottom (`IDIR = 1`): the block's top end is the larger.
    Down,
    /// Bottom to top (`IDIR = 2`): the block's bottom end is the larger.
    Up,
}

impl Direction {
    /// `dbdsqr`'s choice "from larger end diagonal element towards smaller":
    /// down when `|d_p| ≥ |d_q|`, else up.
    pub(super) fn of_block<T: RealScalar>(d: &[T], p: usize, q: usize) -> Self {
        if d[p].abs() >= d[q].abs() {
            Self::Down
        } else {
            Self::Up
        }
    }
}

/// An orientation of the block `[p, q]`: the map from the oriented
/// bidiagonal's indices to the stored ones, and the factor each oriented
/// rotation accumulates into.
pub(super) trait Chase {
    /// Stored index of the oriented diagonal entry `j` (`p ≤ j ≤ q`).
    fn diagonal(p: usize, q: usize, j: usize) -> usize;
    /// Stored index of the oriented superdiagonal entry `j` (`p ≤ j < q`).
    fn superdiagonal(p: usize, q: usize, j: usize) -> usize;
    /// Accumulate a rotation of oriented columns `(a, b)`.
    fn rotate_columns<T: RealScalar>(
        factors: &mut TransposedFactors<'_, T>,
        p: usize,
        q: usize,
        a: usize,
        b: usize,
        rotation: (T, T),
    );
    /// Accumulate a rotation of oriented rows `(a, b)`.
    fn rotate_rows<T: RealScalar>(
        factors: &mut TransposedFactors<'_, T>,
        p: usize,
        q: usize,
        a: usize,
        b: usize,
        rotation: (T, T),
    );
}

/// Top to bottom: the stored block itself.
pub(super) struct Down;

/// Bottom to top: the reflection `P Bᵀ P` of the stored block.
pub(super) struct Up;

impl Chase for Down {
    #[inline]
    fn diagonal(_p: usize, _q: usize, j: usize) -> usize {
        j
    }
    #[inline]
    fn superdiagonal(_p: usize, _q: usize, j: usize) -> usize {
        j
    }
    #[inline]
    fn rotate_columns<T: RealScalar>(
        factors: &mut TransposedFactors<'_, T>,
        _p: usize,
        _q: usize,
        a: usize,
        b: usize,
        (c, s): (T, T),
    ) {
        factors.rotate_right(a, b, c, s);
    }
    #[inline]
    fn rotate_rows<T: RealScalar>(
        factors: &mut TransposedFactors<'_, T>,
        _p: usize,
        _q: usize,
        a: usize,
        b: usize,
        (c, s): (T, T),
    ) {
        factors.rotate_left(a, b, c, s);
    }
}

impl Chase for Up {
    #[inline]
    fn diagonal(p: usize, q: usize, j: usize) -> usize {
        p + q - j
    }
    #[inline]
    fn superdiagonal(p: usize, q: usize, j: usize) -> usize {
        p + q - 1 - j
    }
    #[inline]
    fn rotate_columns<T: RealScalar>(
        factors: &mut TransposedFactors<'_, T>,
        p: usize,
        q: usize,
        a: usize,
        b: usize,
        (c, s): (T, T),
    ) {
        factors.rotate_left(p + q - a, p + q - b, c, s);
    }
    #[inline]
    fn rotate_rows<T: RealScalar>(
        factors: &mut TransposedFactors<'_, T>,
        p: usize,
        q: usize,
        a: usize,
        b: usize,
        (c, s): (T, T),
    ) {
        factors.rotate_right(p + q - a, p + q - b, c, s);
    }
}

/// The block `[p, q]` of `(d, e)` seen in orientation `C`, `p < q`: the
/// oriented superdiagonal is defined on `p ≤ j < q` only, and a one-row
/// block has none (for [`Up`] its index map would leave the block).
pub(super) struct Oriented<'s, T, C> {
    d: &'s mut [T],
    e: &'s mut [T],
    p: usize,
    q: usize,
    chase: PhantomData<C>,
}

impl<'s, T: RealScalar, C: Chase> Oriented<'s, T, C> {
    pub(super) fn new(d: &'s mut [T], e: &'s mut [T], p: usize, q: usize) -> Self {
        assert!(p < q, "invariant: an oriented block has at least two rows");
        Self {
            d,
            e,
            p,
            q,
            chase: PhantomData,
        }
    }

    /// Oriented diagonal entry `j`.
    #[inline]
    pub(super) fn d(&self, j: usize) -> T {
        self.d[C::diagonal(self.p, self.q, j)]
    }

    /// Oriented superdiagonal entry `j`.
    #[inline]
    pub(super) fn e(&self, j: usize) -> T {
        self.e[C::superdiagonal(self.p, self.q, j)]
    }

    #[inline]
    pub(super) fn set_d(&mut self, j: usize, value: T) {
        self.d[C::diagonal(self.p, self.q, j)] = value;
    }

    #[inline]
    pub(super) fn set_e(&mut self, j: usize, value: T) {
        self.e[C::superdiagonal(self.p, self.q, j)] = value;
    }

    /// The largest `|dᵢ|, |eᵢ|` over the active block `[p, q]`
    /// (orientation-independent: the oriented accessors visit exactly the
    /// stored entries of `[p, q]`, in a block-local permutation of them).
    /// `dbdsqr` takes its `SMAX` over this same scan (LAPACK 3.12.0
    /// `dbdsqr.f` lines 453–462, `DO 90 LLL = LL, M`).
    pub(super) fn largest_entry(&self, p: usize, q: usize) -> T {
        let mut largest = self.d(p).abs();
        for j in p..q {
            let e = self.e(j).abs();
            if e > largest {
                largest = e;
            }
            let d = self.d(j + 1).abs();
            if d > largest {
                largest = d;
            }
        }
        largest
    }

    /// Stored index of the oriented superdiagonal entry `j`.
    #[inline]
    pub(super) fn stored_superdiagonal(&self, j: usize) -> usize {
        C::superdiagonal(self.p, self.q, j)
    }

    /// Accumulate a rotation of oriented columns `(a, b)` (`V` when `Down`).
    #[inline]
    pub(super) fn rotate_columns(
        &self,
        factors: &mut TransposedFactors<'_, T>,
        a: usize,
        b: usize,
        rotation: (T, T),
    ) {
        C::rotate_columns(factors, self.p, self.q, a, b, rotation);
    }

    /// Accumulate a rotation of oriented rows `(a, b)` (`U` when `Down`).
    #[inline]
    pub(super) fn rotate_rows(
        &self,
        factors: &mut TransposedFactors<'_, T>,
        a: usize,
        b: usize,
        rotation: (T, T),
    ) {
        C::rotate_rows(factors, self.p, self.q, a, b, rotation);
    }
}
