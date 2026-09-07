use super::weight::Weight;

/// Saved recurrences for a mean-reduced connectionist temporal classification loss.
///
/// State is constructed by [`Self::forward`]. It owns no input log-probabilities:
/// alpha includes the current emission, while beta includes only later emissions.
/// Their sum minus the sample log-likelihood is a posterior log occupancy.
/// Each log message keeps an offset and rounding residual, both in `T`, so
/// path-count corrections survive until posterior normalization.
#[derive(Debug)]
pub struct CtcState<T> {
    pub(super) shape: [usize; 3],
    pub(super) loss: T,
    pub(super) batch_divisor: T,
    pub(super) alpha: Vec<Weight<T>>,
    pub(super) beta: Vec<Weight<T>>,
    pub(super) labels: Vec<usize>,
    pub(super) samples: Vec<Sample<T>>,
}

#[derive(Debug)]
pub(super) struct Sample<T> {
    pub(super) frames: usize,
    pub(super) states: usize,
    pub(super) offset: usize,
    pub(super) labels_offset: usize,
    pub(super) log_likelihood: Weight<T>,
    pub(super) target_divisor: T,
}

impl<T: Copy> CtcState<T> {
    /// Returns `sum(loss[n] / max(target_length[n], 1)) / batch`.
    ///
    /// An impossible alignment yields positive infinity, including a nonempty
    /// target with zero valid frames. Empty input and target contribute zero.
    #[must_use]
    pub fn loss(&self) -> T {
        self.loss
    }
}
