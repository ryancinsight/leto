use leto::application::array::AssignSource;
use leto::{Array2, ArrayView, ArrayViewMut, Complex, Layout, LetoError, Result, SliceArg};

mod transpose;

fn complex_input(shape: [usize; 2]) -> Array2<Complex<f64>> {
    Array2::from_shape_fn(shape, |[row, column]| {
        let linear = row * shape[1] + column;
        Complex::new(linear as f64 + 0.5, linear as f64 * -0.25)
    })
}

#[test]
fn rectangular_transpose_assignment_matches_analytical_mapping() {
    for shape in [[2, 3], [3, 2], [1, 7], [7, 1], [35, 67], [67, 35]] {
        let [rows, columns] = shape;
        let source = complex_input(shape);
        let transposed = source
            .transpose([1, 0])
            .expect("rank-2 transpose permutation is valid");
        let mut target = Array2::zeros([columns, rows]);

        target
            .view_mut()
            .try_assign(&transposed)
            .expect("rectangular transpose assignment is valid");

        for row in 0..rows {
            for column in 0..columns {
                assert_eq!(target[[column, row]], source[[row, column]]);
            }
        }
    }
}

#[test]
fn empty_and_singleton_assignments_preserve_shapes_and_values() {
    let empty = Array2::<i32>::zeros([0, 4]);
    let empty_transposed = empty
        .transpose([1, 0])
        .expect("empty rank-2 transpose permutation is valid");
    let mut empty_target = Array2::<i32>::zeros([4, 0]);
    empty_target
        .view_mut()
        .try_assign(&empty_transposed)
        .expect("empty transpose assignment is valid");
    assert_eq!(empty_target.shape(), [4, 0]);
    assert_eq!(empty_target.size(), 0);

    let singleton =
        Array2::from_shape_vec([1, 1], vec![37_i32]).expect("singleton storage matches its shape");
    let mut singleton_target = Array2::zeros([1, 1]);
    singleton_target
        .view_mut()
        .try_assign(&singleton.view())
        .expect("singleton assignment is valid");
    assert_eq!(singleton_target[[0, 0]], 37);
}

#[test]
fn shape_rejection_is_failure_atomic() {
    let source =
        Array2::from_shape_vec([3, 2], (0..6).collect()).expect("source storage matches its shape");
    let mut target =
        Array2::from_shape_vec([2, 3], vec![91_i32; 6]).expect("target storage matches its shape");
    let before = target.view().data().to_vec();

    let error = target
        .view_mut()
        .try_assign(&source.view())
        .expect_err("shape mismatch must be rejected before mutation");

    assert_eq!(
        error,
        LetoError::ShapeMismatch {
            lhs: vec![2, 3],
            rhs: vec![3, 2],
        }
    );
    assert_eq!(target.view().data(), before);
}

#[test]
fn aliased_destination_preserves_logical_overwrite_order() {
    let source = Array2::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4])
        .expect("source storage matches its shape");
    let layout = Layout::try_new([2, 2], [0, 1], 0).expect("aliased layout arithmetic is valid");
    let mut storage = vec![17_i32, 19];

    ArrayViewMut::try_new(layout, &mut storage)
        .expect("aliased layout remains storage-reachable")
        .try_assign(&source.view())
        .expect("aliased assignment retains the checked fallback");

    assert_eq!(storage, vec![3, 4]);
}

#[test]
fn nonzero_stride_alias_preserves_logical_overwrite_order() {
    let source = Array2::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4])
        .expect("source storage matches its shape");
    let layout = Layout::try_new([2, 2], [1, 1], 0).expect("overlapping nonzero strides are valid");
    let mut storage = vec![17_i32, 19, 23];

    ArrayViewMut::try_new(layout, &mut storage)
        .expect("aliased layout remains storage-reachable")
        .try_assign(&source.view())
        .expect("aliased assignment retains the checked fallback");

    assert_eq!(storage, vec![1, 3, 4]);
}

#[test]
fn invalid_builtin_source_is_rejected_before_mutation() {
    let source_layout = Layout::c_contiguous([2, 2]).expect("small dense source layout is valid");
    let source_storage = vec![1_i32, 2];
    let source = ArrayView::new(source_layout, &source_storage);
    let mut target = Array2::from_shape_vec([2, 2], vec![17_i32, 19, 23, 29])
        .expect("target storage matches its shape");
    let before = target.view().data().to_vec();

    let error = target
        .view_mut()
        .try_assign(&source)
        .expect_err("invalid built-in source storage must be rejected");

    assert!(matches!(error, LetoError::StorageError { .. }));
    assert_eq!(target.view().data(), before);
}

#[test]
fn gapped_negative_injective_strides_copy_in_logical_order() {
    let source_storage = Array2::from_shape_vec([4, 6], (0..24).collect())
        .expect("source storage matches its shape");
    let source = source_storage
        .view()
        .slice_with::<2>(&[
            SliceArg::range(None, None, 2),
            SliceArg::range(None, None, 2),
        ])
        .expect("step-2 source slice is valid");
    let destination_layout =
        Layout::try_new([2, 3], [4, -1], 2).expect("destination layout is injective");
    let mut destination_storage = vec![-1_i32; 7];

    ArrayViewMut::try_new(destination_layout, &mut destination_storage)
        .expect("destination layout fits its storage")
        .try_assign(&source)
        .expect("arbitrary injective stride assignment is valid");

    assert_eq!(destination_storage, vec![4, 2, 0, -1, 16, 14, 12]);
}

#[test]
fn offset_fortran_source_copies_into_offset_c_destination() {
    let shape = [3, 5];
    let source_layout = Layout::try_new(shape, [1, 3], 2).expect("offset F layout is valid");
    let mut source_storage = vec![-1_i32; 19];
    for row in 0..shape[0] {
        for column in 0..shape[1] {
            source_storage[2 + column * shape[0] + row] =
                i32::try_from(row * 10 + column).expect("fixture values fit in i32");
        }
    }
    let source = ArrayView::try_new(source_layout, &source_storage)
        .expect("offset F source fits its storage");
    let destination_layout =
        Layout::try_new(shape, [5, 1], 4).expect("offset C destination layout is valid");
    let mut destination_storage = vec![-7_i32; 21];

    ArrayViewMut::try_new(destination_layout, &mut destination_storage)
        .expect("offset C destination fits its storage")
        .try_assign(&source)
        .expect("offset F-to-C assignment is valid");

    assert_eq!(&destination_storage[..4], &[-7; 4]);
    assert_eq!(
        &destination_storage[4..19],
        &[0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
    );
    assert_eq!(&destination_storage[19..], &[-7; 2]);
}

#[test]
fn offset_dense_views_copy_only_their_reachable_blocks() {
    let source_storage = vec![-1_i32, -1, 2, 3, 5, 7, -1, -1];
    let source_layout = Layout::try_new([2, 2], [2, 1], 2).expect("offset source layout is valid");
    let source = ArrayView::try_new(source_layout, &source_storage)
        .expect("offset source layout fits its storage");
    let mut destination_storage = vec![11_i32; 9];
    let destination_layout =
        Layout::try_new([2, 2], [2, 1], 3).expect("offset destination layout is valid");

    ArrayViewMut::try_new(destination_layout, &mut destination_storage)
        .expect("offset destination layout fits its storage")
        .try_assign(&source)
        .expect("offset dense assignment is valid");

    assert_eq!(destination_storage, vec![11, 11, 11, 2, 3, 5, 7, 11, 11]);
}

struct ExternalSource {
    values: Array2<i32>,
}

impl AssignSource<i32, 2> for ExternalSource {
    fn assign_shape(&self) -> [usize; 2] {
        self.values.shape()
    }

    fn assign_get(&self, index: [usize; 2]) -> Result<&i32> {
        self.values.get(index)
    }
}

#[test]
fn external_assign_source_retains_checked_fallback() {
    let source = ExternalSource {
        values: Array2::from_shape_vec([2, 3], vec![1, 2, 3, 5, 8, 13])
            .expect("external source storage matches its shape"),
    };
    let mut target = Array2::zeros([2, 3]);

    target
        .try_assign(&source)
        .expect("external checked assignment is valid");

    assert_eq!(target.view().data(), &[1, 2, 3, 5, 8, 13]);
}

struct InconsistentViewSource {
    indexed: Array2<i32>,
    fast_view: Array2<i32>,
}

impl AssignSource<i32, 2> for InconsistentViewSource {
    fn assign_shape(&self) -> [usize; 2] {
        self.indexed.shape()
    }

    fn assign_get(&self, index: [usize; 2]) -> Result<&i32> {
        self.indexed.get(index)
    }

    fn assign_view(&self) -> Option<ArrayView<'_, i32, 2>> {
        Some(self.fast_view.view())
    }
}

#[test]
fn inconsistent_external_fast_view_is_rejected_before_mutation() {
    let source = InconsistentViewSource {
        indexed: Array2::from_shape_vec([2, 2], vec![1, 2, 3, 4])
            .expect("indexed source storage matches its shape"),
        fast_view: Array2::from_shape_vec([4, 1], vec![5, 6, 7, 8])
            .expect("fast-view source storage matches its shape"),
    };
    let mut target = Array2::from_shape_vec([2, 2], vec![17, 19, 23, 29])
        .expect("target storage matches its shape");
    let before = target.view().data().to_vec();

    let error = target
        .try_assign(&source)
        .expect_err("inconsistent fast-view shape must be rejected");

    assert_eq!(
        error,
        LetoError::ShapeMismatch {
            lhs: vec![2, 2],
            rhs: vec![4, 1],
        }
    );
    assert_eq!(target.view().data(), before);
}
