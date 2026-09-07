use super::CtcError;
use crate::RealScalar;
use eunomia::{FloatElement, NumericElement};

/// A log weight and its rounding residual, both in the selected scalar.
///
/// Keeping the residual preserves path multiplicities such as `-A + ln(2)`
/// when `ln(2)` lies below the spacing of `A`. Converting that message to one
/// scalar before posterior normalization would discard the multiplicity.
#[derive(Clone, Copy, Debug)]
pub(super) struct Weight<T> {
    high: T,
    low: T,
}

impl<T: RealScalar> Weight<T> {
    pub(super) fn scalar(high: T) -> Self {
        Self {
            high,
            low: <T as NumericElement>::ZERO,
        }
    }

    pub(super) fn unreachable() -> Self {
        Self::scalar(-<T as NumericElement>::INFINITY)
    }

    pub(super) fn is_unreachable(self) -> bool {
        self.high == -<T as NumericElement>::INFINITY
    }

    pub(super) fn value(self) -> T {
        self.high + self.low
    }

    // Knuth's TwoSum, Ogita et al. (2005), slide 7:
    // https://ogilab.w.waseda.jp/ogita/math/presen/Dag2005_Ogita.pdf
    // Round-to-nearest with gradual underflow retains the sum's residual
    // when intermediates do not overflow. No widening occurs. Combining
    // residuals still rounds; this is not an exact arbitrary-length expansion.
    fn sum(a: T, b: T) -> Self {
        let high = a + b;
        let virtual_b = high - a;
        let low = (a - (high - virtual_b)) + (b - virtual_b);
        Self { high, low }
    }

    pub(super) fn add(self, other: Self, sample: usize) -> Result<Self, CtcError> {
        if self.is_unreachable() || other.is_unreachable() {
            return Ok(Self::unreachable());
        }
        let leading = Self::sum(self.high, other.high);
        if !NumericElement::is_finite(leading.high) {
            return Err(CtcError::Arithmetic { sample });
        }
        let result = Self::sum(leading.high, leading.low + (self.low + other.low));
        if !NumericElement::is_finite(result.high) || !NumericElement::is_finite(result.low) {
            return Err(CtcError::Arithmetic { sample });
        }
        Ok(result)
    }

    pub(super) fn subtract(self, other: Self, sample: usize) -> Result<Self, CtcError> {
        self.add(
            Self {
                high: -other.high,
                low: -other.low,
            },
            sample,
        )
    }

    pub(super) fn merge(self, other: Self, sample: usize) -> Result<Self, CtcError> {
        if self.is_unreachable() {
            return Ok(other);
        }
        if other.is_unreachable() {
            return Ok(self);
        }
        let (large, small) =
            if self.high > other.high || (self.high == other.high && self.low >= other.low) {
                (self, other)
            } else {
                (other, self)
            };
        // A difference below the scalar range has an exponential rounded to
        // zero, so its contribution is exactly absent at this precision.
        if small.high - large.high == -<T as NumericElement>::INFINITY {
            return Ok(large);
        }
        let difference = small.subtract(large, sample)?.value();
        let correction =
            FloatElement::ln(<T as NumericElement>::ONE + FloatElement::exp(difference));
        large.add(Self::scalar(correction), sample)
    }
}
