#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array, Layout, LetoError, SliceArg, Storage, VecStorage};
use leto_ops::{embedding_gather, embedding_gather_into};

fn contract_table() -> Array<f32, VecStorage<f32>, 2> {
    Array::from_shape_vec(
        [4, 3],
        vec![
            0.0, 1.0, 2.0, // row 0
            10.0, 11.0, 12.0, // row 1
            20.0, 21.0, 22.0, // row 2
            30.0, 31.0, 32.0, // row 3
        ],
    )
    .unwrap()
}

/// The hephaestus embedding conformance fixture, verbatim: gathering rows
/// `[2, 0, 3]` must reproduce those rows contiguously, in order.
#[test]
fn gather_matches_gpu_contract_fixture() {
    let table = contract_table();
    let indices = Array::from_shape_vec([3], vec![2u32, 0, 3]).unwrap();
    let layout = Layout::c_contiguous([3, 3]).unwrap();
    let mut output = Array::new(layout, VecStorage::fill(9, 0.0)).unwrap();
    embedding_gather_into(&table.view(), &indices.view(), &mut output.view_mut()).unwrap();
    assert_eq!(
        output.storage().as_slice(),
        &[20.0, 21.0, 22.0, 0.0, 1.0, 2.0, 30.0, 31.0, 32.0]
    );
}

/// The seam's second clause: index 4 is out of range for 4 embeddings and
/// must be a typed rejection, never a clamped or wrapped read. Rows before
/// the failing index are already written, exactly as the host reference.
#[test]
fn gather_rejects_out_of_range_index() {
    let table = contract_table();
    let indices = Array::from_shape_vec([2], vec![0u32, 4]).unwrap();
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let mut output = Array::new(layout, VecStorage::fill(6, -1.0)).unwrap();
    let err = embedding_gather_into(&table.view(), &indices.view(), &mut output.view_mut())
        .expect_err("index 4 is out of range for 4 embeddings");
    assert_eq!(
        err,
        LetoError::OutOfBounds {
            index: vec![4],
            shape: vec![4],
        }
    );
    assert_eq!(
        output.storage().as_slice(),
        &[0.0, 1.0, 2.0, -1.0, -1.0, -1.0]
    );
}

#[test]
fn gather_serves_strided_views() {
    // Even columns carry the contract rows; odd columns are sentinel gaps.
    let table = Array::from_shape_vec(
        [4, 6],
        vec![
            0.0, -1.0, 1.0, -1.0, 2.0, -1.0, //
            10.0, -1.0, 11.0, -1.0, 12.0, -1.0, //
            20.0, -1.0, 21.0, -1.0, 22.0, -1.0, //
            30.0, -1.0, 31.0, -1.0, 32.0, -1.0, //
        ],
    )
    .unwrap();
    let selected = table
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(Some(0), None, 2)])
        .unwrap();
    assert_eq!(selected.shape(), [4, 3]);
    let indices = Array::from_shape_vec([5], vec![2u32, 99, 0, 99, 3]).unwrap();
    let stepped = indices
        .view()
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .unwrap();
    assert_eq!(stepped.shape(), [3]);
    let mut backing = Array::from_shape_vec([3, 6], vec![-1.0f32; 18]).unwrap();
    let mut out = backing
        .slice_with_mut::<2>(&[SliceArg::All, SliceArg::range(Some(0), None, 2)])
        .unwrap();
    embedding_gather_into(&selected, &stepped, &mut out).unwrap();
    assert_eq!(
        backing.storage().as_slice(),
        &[
            20.0, -1.0, 21.0, -1.0, 22.0, -1.0, //
            0.0, -1.0, 1.0, -1.0, 2.0, -1.0, //
            30.0, -1.0, 31.0, -1.0, 32.0, -1.0, //
        ]
    );
}

#[test]
fn gather_rejects_shape_mismatch() {
    let table = contract_table();
    let indices = Array::from_shape_vec([3], vec![2u32, 0, 3]).unwrap();
    for bad_shape in [[4, 3], [3, 4], [2, 3]] {
        let bad = Layout::c_contiguous(bad_shape).unwrap();
        let mut output =
            Array::new(bad, VecStorage::fill(bad_shape[0] * bad_shape[1], 0.0)).unwrap();
        let err = embedding_gather_into(&table.view(), &indices.view(), &mut output.view_mut())
            .expect_err("output shape must be [n, embedding_dim]");
        assert_eq!(
            err,
            LetoError::ShapeMismatch {
                lhs: vec![3, 3],
                rhs: bad_shape.to_vec(),
            }
        );
    }
}

#[test]
fn owned_gather_allocates_and_repeats_rows() {
    let table = contract_table();
    let indices = Array::from_shape_vec([2], vec![1u32, 1]).unwrap();
    let output = embedding_gather(&table.view(), &indices.view()).unwrap();
    assert_eq!(output.shape(), [2, 3]);
    assert_eq!(
        output.storage().as_slice(),
        &[10.0, 11.0, 12.0, 10.0, 11.0, 12.0]
    );
}

#[test]
fn gather_rejects_u32_max_and_accepts_empty() {
    let table = contract_table();
    let huge = Array::from_shape_vec([1], vec![u32::MAX]).unwrap();
    let layout = Layout::c_contiguous([1, 3]).unwrap();
    let mut output = Array::new(layout, VecStorage::fill(3, 0.0)).unwrap();
    assert!(embedding_gather_into(&table.view(), &huge.view(), &mut output.view_mut()).is_err());
    let empty = Array::from_shape_vec([0], Vec::<u32>::new()).unwrap();
    let layout = Layout::c_contiguous([0, 3]).unwrap();
    let mut output = Array::new(layout, VecStorage::fill(0, 0.0)).unwrap();
    embedding_gather_into(&table.view(), &empty.view(), &mut output.view_mut()).unwrap();
}
