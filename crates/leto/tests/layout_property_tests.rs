use leto::{Array, Layout, LetoError, SliceArg};
use proptest::prelude::*;

proptest! {
    #[test]
    fn c_contiguous_offsets_match_row_major_formula(
        rows in 1usize..8,
        cols in 1usize..8,
        depth in 1usize..8,
        row in 0usize..8,
        col in 0usize..8,
        lane in 0usize..8,
    ) {
        let row = row % rows;
        let col = col % cols;
        let lane = lane % depth;
        let layout = Layout::c_contiguous([rows, cols, depth]).unwrap();

        prop_assert_eq!(
            layout.offset_of([row, col, lane]).unwrap(),
            (row * cols * depth) + (col * depth) + lane
        );
        prop_assert_eq!(layout.checked_size().unwrap(), rows * cols * depth);
        prop_assert!(layout.validate_storage_len(rows * cols * depth).is_ok());
    }

    #[test]
    fn f_contiguous_offsets_match_column_major_formula(
        rows in 1usize..8,
        cols in 1usize..8,
        depth in 1usize..8,
        row in 0usize..8,
        col in 0usize..8,
        lane in 0usize..8,
    ) {
        let row = row % rows;
        let col = col % cols;
        let lane = lane % depth;
        let layout = Layout::f_contiguous([rows, cols, depth]).unwrap();

        prop_assert_eq!(
            layout.offset_of([row, col, lane]).unwrap(),
            row + (col * rows) + (lane * rows * cols)
        );
        prop_assert_eq!(layout.checked_size().unwrap(), rows * cols * depth);
        prop_assert!(layout.validate_storage_len(rows * cols * depth).is_ok());
    }

    #[test]
    fn transposed_views_preserve_physical_values(
        rows in 1usize..7,
        cols in 1usize..7,
        depth in 1usize..7,
        a in 0usize..7,
        b in 0usize..7,
        c in 0usize..7,
    ) {
        let a = a % depth;
        let b = b % rows;
        let c = c % cols;
        let len = rows * cols * depth;
        let array = Array::from_shape_vec([rows, cols, depth], (0..len).collect::<Vec<_>>()).unwrap();

        let transposed = array.transpose([2, 0, 1]).unwrap();

        prop_assert_eq!(transposed.shape(), [depth, rows, cols]);
        prop_assert_eq!(
            *transposed.get([a, b, c]).unwrap(),
            *array.get([b, c, a]).unwrap()
        );
    }

    #[test]
    fn reverse_slices_preserve_ndarray_style_value_order(
        rows in 1usize..8,
        cols in 1usize..8,
        row in 0usize..8,
        col in 0usize..8,
    ) {
        let row = row % rows;
        let col = col % cols;
        let len = rows * cols;
        let array = Array::from_shape_vec([rows, cols], (0..len).collect::<Vec<_>>()).unwrap();

        let view = array
            .slice_with::<2>(&[
                SliceArg::range(Some(-1), None, -1),
                SliceArg::All,
            ])
            .unwrap();

        prop_assert_eq!(view.shape(), [rows, cols]);
        prop_assert_eq!(
            *view.get([row, col]).unwrap(),
            *array.get([rows - 1 - row, col]).unwrap()
        );
    }

    #[test]
    fn broadcasted_singleton_axis_preserves_source_values(
        rows in 0usize..8,
        cols in 1usize..8,
        row in 0usize..8,
        col in 0usize..8,
    ) {
        let col = col % cols;
        let source = Array::from_shape_vec([1, cols], (0..cols).collect::<Vec<_>>()).unwrap();
        let broadcast = source.broadcast([rows, cols]).unwrap();

        prop_assert_eq!(broadcast.shape(), [rows, cols]);
        let expected_row_stride = if rows == 1 { cols as isize } else { 0 };
        prop_assert_eq!(broadcast.strides(), [expected_row_stride, 1]);

        if rows > 0 {
            let row = row % rows;
            prop_assert_eq!(*broadcast.get([row, col]).unwrap(), *source.get([0, col]).unwrap());
        }
    }

    #[test]
    fn negative_stride_layout_storage_span_is_checked(len in 1usize..32) {
        let valid = Layout::new([len], [-1], len - 1);

        prop_assert_eq!(valid.checked_min_max_offsets().unwrap(), (0, len - 1));
        prop_assert!(valid.validate_storage_len(len).is_ok());

        if len > 1 {
            let invalid = Layout::new([len], [-1], 0);
            let rejects_negative_span = matches!(
                invalid.checked_min_max_offsets(),
                Err(LetoError::StorageError { .. })
            );
            prop_assert!(rejects_negative_span);
        }
    }
}
