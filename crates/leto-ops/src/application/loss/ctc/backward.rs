use super::{CtcError, CtcState};
use crate::RealScalar;
use eunomia::{FloatElement, NumericElement};
use leto::ArrayViewMut;

impl<T: RealScalar> CtcState<T> {
    /// Adds the seeded derivative with respect to independent log-probabilities.
    ///
    /// Each active coordinate receives `-upstream * posterior / (batch *
    /// max(target_length, 1))`. Padded frames receive no update. Composing this
    /// derivative with log-softmax produces the normalized logit derivative
    /// `probability - posterior`; this method does not add the probability term.
    ///
    /// The destination may be strided or offset but must be injective. A scratch
    /// row of `classes` scalars is allocated once; no input probabilities are
    /// needed. A read-only pass checks every update before an identical pass
    /// writes them, preserving the entire destination on any returned error.
    ///
    /// # Errors
    /// Returns [`CtcError::ImpossibleAlignment`] if any sample has zero alignment
    /// probability, even for a zero seed. Invalid destination layouts/shapes,
    /// nonfinite seeds or active destination values, overflowing updates, and
    /// allocation failures also return typed errors before any write.
    ///
    /// # Examples
    /// ```
    /// use leto::{Array, Layout, Storage, VecStorage};
    /// use leto_ops::ctc::CtcState;
    /// let input = Array::new(Layout::c_contiguous([1, 1, 2])?,
    ///     VecStorage::new(vec![-std::f32::consts::LN_2; 2]))?;
    /// let state = CtcState::forward(&input.view(), &[], &[1], &[0], 0)?;
    /// let mut gradient = Array::new(Layout::c_contiguous([1, 1, 2])?,
    ///     VecStorage::new(vec![3.0_f32; 2]))?;
    /// state.backward_accumulate(2.0, &mut gradient.view_mut())?;
    /// assert_eq!(gradient.storage().as_slice(), &[1.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn backward_accumulate(
        &self,
        upstream: T,
        gradient: &mut ArrayViewMut<'_, T, 3>,
    ) -> Result<(), CtcError> {
        if gradient.shape() != self.shape {
            return Err(CtcError::GradientShape {
                expected: self.shape,
                actual: gradient.shape(),
            });
        }
        gradient
            .layout()
            .validate_storage_len(gradient.data().len())
            .map_err(CtcError::Layout)?;
        if !gradient.layout().is_injective().map_err(CtcError::Layout)? {
            return Err(CtcError::AliasedGradient);
        }
        for (sample, state) in self.samples.iter().enumerate() {
            if state.log_likelihood.is_unreachable() {
                return Err(CtcError::ImpossibleAlignment { sample });
            }
        }
        if !NumericElement::is_finite(upstream) {
            return Err(CtcError::Arithmetic { sample: 0 });
        }
        let mut row = Vec::new();
        row.try_reserve_exact(self.shape[2])
            .map_err(CtcError::Allocation)?;
        row.resize(self.shape[2], <T as NumericElement>::ZERO);
        self.increments(upstream, &mut row, |index, update| {
            let previous = *gradient
                .get(index)
                .expect("invariant: validated gradient coordinate is reachable");
            if !NumericElement::is_finite(previous) || !NumericElement::is_finite(previous + update)
            {
                return Err(CtcError::Arithmetic { sample: index[1] });
            }
            Ok(())
        })?;
        self.increments(upstream, &mut row, |index, update| {
            *gradient
                .get_mut(index)
                .expect("invariant: validated gradient coordinate is reachable") += update;
            Ok(())
        })
    }

    fn increments(
        &self,
        upstream: T,
        row: &mut [T],
        mut visit: impl FnMut([usize; 3], T) -> Result<(), CtcError>,
    ) -> Result<(), CtcError> {
        for (sample, state) in self.samples.iter().enumerate() {
            let labels = &self.labels[state.labels_offset..state.labels_offset + state.states];
            for time in 0..state.frames {
                row.fill(<T as NumericElement>::ZERO);
                let start = state.offset + time * state.states;
                for (position, &label) in labels.iter().enumerate() {
                    let alpha = self.alpha[start + position];
                    let beta = self.beta[start + position];
                    if alpha.is_unreachable() || beta.is_unreachable() {
                        continue;
                    }
                    // Subtract before adding to avoid overflowing alpha+beta
                    // when the normalized posterior is representable.
                    row[label] += FloatElement::exp(
                        alpha
                            .add(beta.subtract(state.log_likelihood, sample)?, sample)?
                            .value(),
                    );
                }
                for (class, &posterior) in row.iter().enumerate() {
                    let update =
                        ((-posterior * upstream) / state.target_divisor) / self.batch_divisor;
                    if !NumericElement::is_finite(update) {
                        return Err(CtcError::Arithmetic { sample });
                    }
                    visit([time, sample, class], update)?;
                }
            }
        }
        Ok(())
    }
}
