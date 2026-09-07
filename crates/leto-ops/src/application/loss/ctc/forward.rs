use super::state::Sample;
use super::weight::Weight;
use super::{CtcError, CtcState};
use crate::RealScalar;
use eunomia::NumericElement;
use leto::ArrayView;

fn reserve<T>(length: usize) -> Result<Vec<T>, CtcError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(CtcError::Allocation)?;
    Ok(values)
}

fn extent<T: RealScalar>(value: usize) -> Result<T, CtcError> {
    let scalar = T::from_usize(value);
    if !NumericElement::is_finite(scalar) {
        return Err(CtcError::ScalarExtent { extent: value });
    }
    Ok(scalar)
}

impl<T: RealScalar> CtcState<T> {
    /// Evaluates temporal alignment loss from `[frames, batch, classes]` log-probabilities.
    ///
    /// Targets are concatenated, exclude `blank`, and have lengths given by
    /// `target_lengths`. Input lengths select valid prefixes; padding is ignored.
    /// Views may be strided or offset. Active values must be nonpositive (negative
    /// infinity represents zero probability). Normalization is the caller's
    /// responsibility; independent log weights use the same path-sum definition.
    ///
    /// Empty targets accept only the all-blank path. Zero frames and an empty
    /// target have probability one; other impossible paths have probability zero.
    /// The mean divides each sample loss by `max(target_length, 1)` before the
    /// batch mean. All recurrences and normalization execute in `T`.
    ///
    /// Alpha follows Graves et al., [CTC, equations 2–8](https://www.cs.toronto.edu/~graves/icml_2006.pdf).
    /// Beta excludes the current emission: its terminal accepted states are zero
    /// in log space, and each step adds the *next* state's emission. Hence
    /// `exp(alpha + beta - log_likelihood)` is posterior state occupancy.
    /// Messages retain a native-`T` offset and residual using compensated sums;
    /// this preserves path multiplicity when the offset's spacing exceeds the
    /// log-count correction. Compensation assumes round-to-nearest and gradual
    /// underflow; it does not remove elementary-function or residual rounding.
    /// Storage is `O(sum(input_length * (2*target_length+1)))` and every size
    /// product is checked before allocation. State is returned only on success.
    ///
    /// # Errors
    /// Returns [`CtcError`] for invalid lengths, labels, layouts, active values,
    /// unrepresentable dimensions/arithmetic, or allocation failure.
    ///
    /// # Examples
    /// ```
    /// use leto::{Array, Layout, VecStorage};
    /// use leto_ops::ctc::CtcState;
    /// let input = Array::new(Layout::c_contiguous([1, 1, 2])?,
    ///     VecStorage::new(vec![-std::f32::consts::LN_2; 2]))?;
    /// let state = CtcState::forward(&input.view(), &[], &[1], &[0], 0)?;
    /// assert_eq!(state.loss(), std::f32::consts::LN_2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn forward(
        log_probs: &ArrayView<'_, T, 3>,
        targets: &[usize],
        input_lengths: &[usize],
        target_lengths: &[usize],
        blank: usize,
    ) -> Result<Self, CtcError> {
        let shape = log_probs.shape();
        let [frames, batch, classes] = shape;
        log_probs
            .layout()
            .validate_storage_len(log_probs.data().len())
            .map_err(CtcError::Layout)?;
        if batch == 0 || classes == 0 {
            return Err(CtcError::EmptyDimension { shape });
        }
        if input_lengths.len() != batch || target_lengths.len() != batch {
            return Err(CtcError::LengthCount {
                batch,
                inputs: input_lengths.len(),
                targets: target_lengths.len(),
            });
        }
        if blank >= classes {
            return Err(CtcError::Label {
                label: blank,
                blank,
                classes,
            });
        }
        let batch_divisor = extent::<T>(batch)?;
        let overflow = || CtcError::SizeOverflow {
            frames,
            targets: targets.len(),
        };
        let mut target_count = 0usize;
        let mut state_count = 0usize;
        let mut label_count = 0usize;
        for (sample, (&length, &target_length)) in
            input_lengths.iter().zip(target_lengths).enumerate()
        {
            if length > frames {
                return Err(CtcError::InputLength {
                    sample,
                    actual: length,
                    maximum: frames,
                });
            }
            extent::<T>(target_length.max(1))?;
            target_count = target_count
                .checked_add(target_length)
                .ok_or_else(overflow)?;
            let states = target_length
                .checked_mul(2)
                .and_then(|n| n.checked_add(1))
                .ok_or_else(overflow)?;
            label_count = label_count.checked_add(states).ok_or_else(overflow)?;
            state_count = length
                .checked_mul(states)
                .and_then(|n| state_count.checked_add(n))
                .ok_or_else(overflow)?;
            for time in 0..length {
                for class in 0..classes {
                    let index = [time, sample, class];
                    let value = *log_probs
                        .get(index)
                        .expect("invariant: validated input coordinate is reachable");
                    if NumericElement::is_nan(value) || value > <T as NumericElement>::ZERO {
                        return Err(CtcError::LogProbability { index });
                    }
                }
            }
        }
        if target_count != targets.len() {
            return Err(CtcError::TargetCount {
                expected: target_count,
                actual: targets.len(),
            });
        }
        for &label in targets {
            if label == blank || label >= classes {
                return Err(CtcError::Label {
                    label,
                    blank,
                    classes,
                });
            }
        }
        let negative_infinity = Weight::unreachable();
        let mut result = Self {
            shape,
            batch_divisor,
            loss: <T as NumericElement>::ZERO,
            alpha: reserve(state_count)?,
            beta: reserve(state_count)?,
            labels: reserve(label_count)?,
            samples: reserve(batch)?,
        };
        result.alpha.resize(state_count, negative_infinity);
        result.beta.resize(state_count, negative_infinity);
        let mut target_offset = 0;
        let mut offset = 0;
        for (sample, (&length, &target_length)) in
            input_lengths.iter().zip(target_lengths).enumerate()
        {
            let labels_offset = result.labels.len();
            result.labels.push(blank);
            for &label in &targets[target_offset..target_offset + target_length] {
                result.labels.extend([label, blank]);
            }
            target_offset += target_length;
            let states = 2 * target_length + 1;
            let labels = &result.labels[labels_offset..labels_offset + states];
            let target_divisor = extent::<T>(target_length.max(1))?;
            let log_likelihood = if length == 0 {
                if target_length == 0 {
                    Weight::scalar(<T as NumericElement>::ZERO)
                } else {
                    negative_infinity
                }
            } else {
                let alpha = &mut result.alpha[offset..offset + length * states];
                let beta = &mut result.beta[offset..offset + length * states];
                recurrence(log_probs, sample, labels, length, alpha, beta)?
            };
            let contribution = (-log_likelihood.value() / target_divisor) / batch_divisor;
            if !log_likelihood.is_unreachable() && !NumericElement::is_finite(contribution) {
                return Err(CtcError::Arithmetic { sample });
            }
            let previous_loss = result.loss;
            result.loss += contribution;
            if NumericElement::is_nan(result.loss)
                || (NumericElement::is_finite(previous_loss)
                    && NumericElement::is_finite(contribution)
                    && !NumericElement::is_finite(result.loss))
            {
                return Err(CtcError::Arithmetic { sample });
            }
            result.samples.push(Sample {
                frames: length,
                states,
                offset,
                labels_offset,
                log_likelihood,
                target_divisor,
            });
            offset += length * states;
        }
        Ok(result)
    }
}

fn recurrence<T: RealScalar>(
    input: &ArrayView<'_, T, 3>,
    sample: usize,
    labels: &[usize],
    frames: usize,
    alpha: &mut [Weight<T>],
    beta: &mut [Weight<T>],
) -> Result<Weight<T>, CtcError> {
    let states = labels.len();
    let read = |time, state| {
        Weight::scalar(
            *input
                .get([time, sample, labels[state]])
                .expect("invariant: validated active emission is reachable"),
        )
    };
    alpha[0] = read(0, 0);
    if states > 1 {
        alpha[1] = read(0, 1);
    }
    for time in 1..frames {
        for state in 0..states {
            let previous = (time - 1) * states;
            let mut value = alpha[previous + state];
            if state > 0 {
                value = value.merge(alpha[previous + state - 1], sample)?;
            }
            if state > 1 && state % 2 == 1 && labels[state] != labels[state - 2] {
                value = value.merge(alpha[previous + state - 2], sample)?;
            }
            alpha[time * states + state] = value.add(read(time, state), sample)?;
        }
    }
    let terminal = (frames - 1) * states;
    beta[terminal + states - 1] = Weight::scalar(<T as NumericElement>::ZERO);
    let mut likelihood = alpha[terminal + states - 1];
    if states > 1 {
        beta[terminal + states - 2] = Weight::scalar(<T as NumericElement>::ZERO);
        likelihood = likelihood.merge(alpha[terminal + states - 2], sample)?;
    }
    for time in (0..frames - 1).rev() {
        for state in 0..states {
            let next = (time + 1) * states;
            let mut value = read(time + 1, state).add(beta[next + state], sample)?;
            if state + 1 < states {
                value = value.merge(
                    read(time + 1, state + 1).add(beta[next + state + 1], sample)?,
                    sample,
                )?;
            }
            if state + 2 < states && state % 2 == 1 && labels[state] != labels[state + 2] {
                value = value.merge(
                    read(time + 1, state + 2).add(beta[next + state + 2], sample)?,
                    sample,
                )?;
            }
            beta[time * states + state] = value;
        }
    }
    Ok(likelihood)
}
