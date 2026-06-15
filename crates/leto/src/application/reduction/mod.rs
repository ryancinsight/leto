//! Element-wise reduction operations over N-dimensional strided arrays.
//!
//! # Submodules
//! - [`sum`](crate::application::reduction::sum) — [`sum_all`], [`sum_axis`]
//! - [`mean`](crate::application::reduction::mean) — [`mean_all`], [`mean_axis`]
//! - [`min_max`](crate::application::reduction::min_max) — [`min_all`], [`max_all`], [`min_axis`], [`max_axis`], [`argmin`], [`argmax`], [`argmin_all`], [`argmax_all`]
//! - [`variance`](crate::application::reduction::variance) — [`var_all`], [`std_all`], [`var_axis`], [`std_axis`]
//! - [`quantile`](crate::application::reduction::quantile) — [`quantile_all`], [`median_all`], [`quantile_axis`], [`median_axis`]
//!
//! # Shared infrastructure
//! `iter_elements` is a `pub(crate)` helper that yields `&T` references for
//! every logical element of an [`ArrayView`](crate::application::view::ArrayView),
//! respecting arbitrary strides.
//! All leaf modules import it via `crate::application::reduction::iter_elements`.

pub mod mean;
pub mod min_max;
pub mod quantile;
pub mod sum;
pub mod variance;

pub use mean::{mean_all, mean_axis};
pub use min_max::{argmax, argmax_all, argmin, argmin_all, max_all, max_axis, min_all, min_axis};
pub use quantile::{median_all, median_axis, quantile_all, quantile_axis, Interpolation};
pub use sum::{sum_all, sum_axis};
pub use variance::{std_all, std_axis, var_all, var_axis};

use crate::application::index::index_from_flat;
use crate::application::view::ArrayView;

/// Iterate every logical element of `view` in row-major index order.
///
/// Fast-paths to a direct slice scan for C-contiguous layouts are not needed
/// here because `index_from_flat` + `offset_of` is branch-free per element.
/// The function is `pub(crate)` so leaf modules can import it without
/// re-implementing the contiguity check.
#[inline]
pub(crate) fn iter_elements<'a, T, const N: usize>(
    view: &'a ArrayView<'a, T, N>,
) -> impl Iterator<Item = &'a T> + 'a {
    let data = view.data();
    let layout = view.layout();
    let size = layout.size();
    let shape = layout.shape;

    (0..size).map(move |flat| {
        let index = index_from_flat(flat, &shape);
        let offset = layout
            .offset_of(index)
            .expect("index_from_flat produced a valid index");
        &data[offset]
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::application::array::Array;
    use crate::application::reduction::{
        argmax, argmax_all, argmin, argmin_all, max_all, max_axis, mean_all, mean_axis, min_all,
        min_axis, sum_all, sum_axis,
    };
    use crate::infrastructure::storage::VecStorage;

    fn arr2(shape: [usize; 2], data: Vec<f32>) -> Array<f32, VecStorage<f32>, 2> {
        Array::from_vec(shape, data).expect("arr2 construction")
    }

    fn arr1(shape: [usize; 1], data: Vec<f32>) -> Array<f32, VecStorage<f32>, 1> {
        Array::from_vec(shape, data).expect("arr1 construction")
    }

    // ── sum_all ───────────────────────────────────────────────────────────────

    #[test]
    fn sum_all_rank1() {
        let a = arr1([4], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(sum_all(&a).unwrap(), 10.0f32);
    }

    #[test]
    fn sum_all_rank2() {
        let a = arr2([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(sum_all(&a).unwrap(), 10.0f32);
    }

    #[test]
    fn sum_all_empty_errors() {
        let a: Array<f32, VecStorage<f32>, 1> = Array::from_vec([0], vec![]).unwrap();
        assert!(sum_all(&a).is_err());
    }

    // ── sum_axis ──────────────────────────────────────────────────────────────

    #[test]
    fn sum_axis0_rank2() {
        // [[1, 2], [3, 4]] summed along axis 0 → [4, 6]
        let a = arr2([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let s = sum_axis::<f32, _, 2, 1>(&a, 0).unwrap();
        assert_eq!(s.into_vec(), vec![4.0f32, 6.0]);
    }

    #[test]
    fn sum_axis1_rank2() {
        // [[1, 2], [3, 4]] summed along axis 1 → [3, 7]
        let a = arr2([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let s = sum_axis::<f32, _, 2, 1>(&a, 1).unwrap();
        assert_eq!(s.into_vec(), vec![3.0f32, 7.0]);
    }

    // ── mean_all ──────────────────────────────────────────────────────────────

    #[test]
    fn mean_all_rank1() {
        let a = arr1([4], vec![1.0, 2.0, 3.0, 4.0]);
        assert!((mean_all(&a).unwrap() - 2.5f32).abs() < 1e-6);
    }

    #[test]
    fn mean_all_empty_errors() {
        let a: Array<f32, VecStorage<f32>, 1> = Array::from_vec([0], vec![]).unwrap();
        assert!(mean_all(&a).is_err());
    }

    // ── mean_axis ─────────────────────────────────────────────────────────────

    #[test]
    fn mean_axis0_rank2() {
        // [[1, 2], [3, 4]] mean along axis 0 → [2, 3]
        let a = arr2([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let m = mean_axis::<f32, _, 2, 1>(&a, 0).unwrap();
        let v = m.into_vec();
        assert!((v[0] - 2.0f32).abs() < 1e-6);
        assert!((v[1] - 3.0f32).abs() < 1e-6);
    }

    // ── min_all / max_all ─────────────────────────────────────────────────────

    #[test]
    fn min_all_rank2() {
        let a = arr2([2, 2], vec![3.0, 1.0, 4.0, 2.0]);
        assert_eq!(min_all(&a).unwrap(), 1.0f32);
    }

    #[test]
    fn max_all_rank2() {
        let a = arr2([2, 2], vec![3.0, 1.0, 4.0, 2.0]);
        assert_eq!(max_all(&a).unwrap(), 4.0f32);
    }

    // ── min_axis / max_axis ───────────────────────────────────────────────────

    #[test]
    fn min_axis0_rank2() {
        // [[3, 1], [4, 2]] min along axis 0 → [3, 1]
        let a = arr2([2, 2], vec![3.0, 1.0, 4.0, 2.0]);
        let m = min_axis::<f32, _, 2, 1>(&a, 0).unwrap();
        assert_eq!(m.into_vec(), vec![3.0f32, 1.0]);
    }

    #[test]
    fn max_axis1_rank2() {
        // [[3, 1], [4, 2]] max along axis 1 → [3, 4]
        let a = arr2([2, 2], vec![3.0, 1.0, 4.0, 2.0]);
        let m = max_axis::<f32, _, 2, 1>(&a, 1).unwrap();
        assert_eq!(m.into_vec(), vec![3.0f32, 4.0]);
    }

    // ── argmin / argmax ───────────────────────────────────────────────────────

    #[test]
    fn argmin_axis0_rank2() {
        // [[3, 1], [4, 2]]: along axis 0 each col min is at row 0 → [0, 0]
        let a = arr2([2, 2], vec![3.0, 1.0, 4.0, 2.0]);
        let idx = argmin::<f32, _, 2, 1>(&a, 0).unwrap();
        assert_eq!(idx.into_vec(), vec![0usize, 0]);
    }

    #[test]
    fn argmax_axis0_rank2() {
        // [[3, 1], [4, 2]]: along axis 0 each col max is at row 1 → [1, 1]
        let a = arr2([2, 2], vec![3.0, 1.0, 4.0, 2.0]);
        let idx = argmax::<f32, _, 2, 1>(&a, 0).unwrap();
        assert_eq!(idx.into_vec(), vec![1usize, 1]);
    }

    #[test]
    fn argmin_axis1_rank2() {
        // [[3, 1], [4, 2]]: along axis 1 row mins at col 1, col 1 → [1, 1]
        let a = arr2([2, 2], vec![3.0, 1.0, 4.0, 2.0]);
        let idx = argmin::<f32, _, 2, 1>(&a, 1).unwrap();
        assert_eq!(idx.into_vec(), vec![1usize, 1]);
    }

    #[test]
    fn argmax_axis1_rank2() {
        // [[3, 1], [4, 2]]: along axis 1 row maxes at col 0, col 0 → [0, 0]
        let a = arr2([2, 2], vec![3.0, 1.0, 4.0, 2.0]);
        let idx = argmax::<f32, _, 2, 1>(&a, 1).unwrap();
        assert_eq!(idx.into_vec(), vec![0usize, 0]);
    }

    #[test]
    fn argmin_rank1() {
        // [5, 2, 8, 1, 9] → argmin = 3
        let a = arr1([5], vec![5.0, 2.0, 8.0, 1.0, 9.0]);
        let idx = argmin::<f32, _, 1, 0>(&a, 0).unwrap();
        assert_eq!(idx.size(), 1);
        assert_eq!(*idx.get([]).unwrap(), 3usize);
    }

    #[test]
    fn argmax_rank1() {
        // [5, 2, 8, 1, 9] → argmax = 4
        let a = arr1([5], vec![5.0, 2.0, 8.0, 1.0, 9.0]);
        let idx = argmax::<f32, _, 1, 0>(&a, 0).unwrap();
        assert_eq!(*idx.get([]).unwrap(), 4usize);
    }

    // ── argmin_all / argmax_all (whole-array multi-index) ───────────────────────

    #[test]
    fn argmin_all_rank2_multi_index() {
        // [[3, 1], [4, 2]] global min is 1 at [0, 1]; max is 4 at [1, 0].
        let a = arr2([2, 2], vec![3.0, 1.0, 4.0, 2.0]);
        assert_eq!(argmin_all(&a).unwrap(), [0usize, 1]);
        assert_eq!(argmax_all(&a).unwrap(), [1usize, 0]);
    }

    #[test]
    fn argmin_all_first_occurrence_on_ties() {
        // Two minima (1.0) at flat 1 and flat 2; first row-major wins → [0, 1].
        let a = arr2([2, 2], vec![3.0, 1.0, 1.0, 2.0]);
        assert_eq!(argmin_all(&a).unwrap(), [0usize, 1]);
        // Two maxima (4.0) at flat 0 and flat 3; first wins → [0, 0].
        let b = arr2([2, 2], vec![4.0, 1.0, 2.0, 4.0]);
        assert_eq!(argmax_all(&b).unwrap(), [0usize, 0]);
    }

    #[test]
    fn argmin_all_rank1() {
        // [5, 2, 8, 1, 9] → min at index 3, max at index 4.
        let a = arr1([5], vec![5.0, 2.0, 8.0, 1.0, 9.0]);
        assert_eq!(argmin_all(&a).unwrap(), [3usize]);
        assert_eq!(argmax_all(&a).unwrap(), [4usize]);
    }

    #[test]
    fn argmin_all_value_agrees_with_min_all() {
        // The element at argmin_all equals min_all (and likewise for max).
        let a = arr2([3, 3], vec![6.0, 2.0, 1.0, 9.0, 5.0, 2.0, 1.0, 8.0, 4.0]);
        let imin = argmin_all(&a).unwrap();
        assert_eq!(*a.get(imin).unwrap(), min_all(&a).unwrap());
        let imax = argmax_all(&a).unwrap();
        assert_eq!(*a.get(imax).unwrap(), max_all(&a).unwrap());
    }

    #[test]
    fn argmin_all_empty_errors() {
        let a: Array<f32, VecStorage<f32>, 1> = Array::from_vec([0], vec![]).unwrap();
        assert!(argmin_all(&a).is_err());
        assert!(argmax_all(&a).is_err());
    }
}
