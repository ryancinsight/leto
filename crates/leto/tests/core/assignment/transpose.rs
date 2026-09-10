use std::cell::Cell;
use std::fmt::Debug;
use std::rc::Rc;

use leto::{transpose_copy, transpose_copy_strided, LetoError};

fn assert_coordinate_mapping<T: Clone + Debug + PartialEq>(make: impl Fn(usize) -> T) {
    for (rows, columns) in [
        (0, 0),
        (0, 5),
        (5, 0),
        (1, 1),
        (1, 129),
        (129, 1),
        (31, 33),
        (33, 31),
        (63, 65),
        (65, 63),
        (127, 129),
        (129, 127),
    ] {
        let count = rows * columns;
        for (source_offset, destination_offset) in [(0, 0), (1, 3), (3, 1)] {
            let source: Vec<T> = (0..source_offset + count + 3).map(&make).collect();
            let before = source.clone();
            let mut destination: Vec<T> = (0..destination_offset + count + 3)
                .map(|_| make(usize::MAX))
                .collect();
            transpose_copy(
                &source[source_offset..source_offset + count],
                &mut destination[destination_offset..destination_offset + count],
                rows,
                columns,
            )
            .expect("exact dense matrix extents are valid");

            for row in 0..rows {
                for column in 0..columns {
                    assert_eq!(
                        destination[destination_offset + column * rows + row],
                        source[source_offset + row * columns + column],
                        "shape {rows}x{columns}, element ({row}, {column})"
                    );
                }
            }
            for guard in destination[..destination_offset]
                .iter()
                .chain(&destination[destination_offset + count..])
            {
                assert_eq!(*guard, make(usize::MAX));
            }
            assert_eq!(source, before);
        }
    }
}

#[test]
fn dense_transpose_preserves_coordinates_and_guards() {
    assert_coordinate_mapping(|value| value);
    assert_coordinate_mapping(|value| value.to_string());
}

#[derive(Debug)]
struct CloneRecord {
    value: usize,
    clones: Rc<Cell<usize>>,
}

impl Clone for CloneRecord {
    fn clone(&self) -> Self {
        self.clones.set(self.clones.get() + 1);
        Self {
            value: self.value,
            clones: Rc::clone(&self.clones),
        }
    }
}

#[test]
fn dense_transpose_clones_each_element_once() {
    for (rows, columns) in [(3, 7), (7, 3), (35, 67), (67, 35)] {
        let clones = Rc::new(Cell::new(0));
        let source: Vec<_> = (0..rows * columns)
            .map(|value| CloneRecord {
                value,
                clones: Rc::clone(&clones),
            })
            .collect();
        let mut destination: Vec<_> = (0..rows * columns)
            .map(|_| CloneRecord {
                value: usize::MAX,
                clones: Rc::clone(&clones),
            })
            .collect();
        transpose_copy(&source, &mut destination, rows, columns)
            .expect("non-Copy elements support dense transpose");
        assert_eq!(clones.get(), rows * columns);
        for row in 0..rows {
            for column in 0..columns {
                assert_eq!(
                    destination[column * rows + row].value,
                    row * columns + column
                );
            }
        }
        for (index, record) in source.iter().enumerate() {
            assert_eq!(record.value, index);
        }
        let last = destination
            .last()
            .expect("the clone workload is nonempty")
            .value;
        assert_eq!(
            transpose_copy(&source, &mut destination[..1], rows, columns),
            Err(LetoError::StorageError {
                reason: format!(
                    "dense transpose destination length 1 does not match expected {}",
                    rows * columns
                ),
            })
        );
        assert_eq!(clones.get(), rows * columns, "rejection must not clone");
        assert_eq!(destination[0].value, 0);
        assert_eq!(
            destination.last().expect("storage is unchanged").value,
            last
        );
    }
}

#[test]
fn dense_transpose_rejects_lengths_in_source_then_destination_order() {
    for (source_len, destination_len, role, actual) in [
        (5, 6, "source", 5),
        (7, 6, "source", 7),
        (6, 5, "destination", 5),
        (6, 7, "destination", 7),
        (5, 7, "source", 5),
    ] {
        let source = vec![13; source_len];
        let mut destination = vec![91; destination_len];
        let error = transpose_copy(&source, &mut destination, 2, 3)
            .expect_err("inexact source or destination extent must fail");
        assert_eq!(
            error,
            LetoError::StorageError {
                reason: format!("dense transpose {role} length {actual} does not match expected 6")
            }
        );
        assert_eq!(source, vec![13; source_len]);
        assert_eq!(destination, vec![91; destination_len]);
    }
}

#[test]
fn dense_transpose_product_overflow_precedes_storage_validation() {
    for (rows, columns) in [(usize::MAX, 2), (2, usize::MAX), (usize::MAX, usize::MAX)] {
        let source = [13];
        let mut destination = [91];
        assert_eq!(
            transpose_copy(&source, &mut destination, rows, columns),
            Err(LetoError::Overflow {
                reason: "dense transpose element count"
            })
        );
        assert_eq!(source, [13]);
        assert_eq!(destination, [91]);
    }
}

#[test]
fn dense_transpose_empty_shapes_validate_exact_storage() {
    for (rows, columns) in [(0, usize::MAX), (usize::MAX, 0), (0, 0)] {
        let mut destination = [91];
        transpose_copy::<usize>(&[], &mut destination[..0], rows, columns)
            .expect("empty shapes require no signed dimension conversion");
        assert_eq!(destination, [91]);
        assert_eq!(
            transpose_copy(&[13], &mut destination[..0], rows, columns),
            Err(LetoError::StorageError {
                reason: "dense transpose source length 1 does not match expected 0".to_owned()
            })
        );
        assert_eq!(
            transpose_copy::<usize>(&[], &mut destination, rows, columns),
            Err(LetoError::StorageError {
                reason: "dense transpose destination length 1 does not match expected 0".to_owned()
            })
        );
        assert_eq!(destination, [91]);
    }
}

#[test]
fn dense_transpose_rejects_unrepresentable_zero_sized_extents() {
    // Vec<()> represents these lengths without allocation or element work.
    // Non-ZST safe slices cannot reach this signed-layout boundary.
    let count = usize::try_from(isize::MAX).expect("isize::MAX fits usize") + 1;
    let source = vec![(); count];
    let mut destination = vec![(); count];
    for (rows, columns) in [(count, 1), (1, count), (count / 2, 2), (2, count / 2)] {
        assert_eq!(
            transpose_copy(&source, &mut destination, rows, columns),
            Err(LetoError::Overflow {
                reason: "dense transpose signed layout extent"
            })
        );
    }
}

#[test]
fn dense_transpose_preserves_observable_zero_sized_clones() {
    thread_local! { static CLONES: Cell<usize> = const { Cell::new(0) }; }
    struct Element;
    impl Clone for Element {
        fn clone(&self) -> Self {
            CLONES.with(|count| count.set(count.get() + 1));
            Self
        }
    }
    for (rows, columns) in [(3, 7), (7, 3), (0, 7), (7, 0)] {
        let source: Vec<_> = (0..rows * columns).map(|_| Element).collect();
        let mut destination: Vec<_> = (0..rows * columns).map(|_| Element).collect();
        CLONES.with(|count| count.set(0));
        transpose_copy(&source, &mut destination, rows, columns)
            .expect("zero-sized Clone elements have observable clone semantics");
        CLONES.with(|count| assert_eq!(count.get(), rows * columns));
    }
}

/// Splits `columns` into windows whose widths cycle through a few tile
/// relations: below, at and across the traversal's tile edges.
fn column_windows(columns: usize) -> Vec<(usize, usize)> {
    let mut windows = Vec::new();
    let mut start = 0;
    for width in [1, 7, 16, 33, 5].into_iter().cycle() {
        if start >= columns {
            break;
        }
        let end = (start + width).min(columns);
        windows.push((start, end));
        start = end;
    }
    windows
}

#[test]
fn strided_windows_assemble_the_dense_transpose() {
    for (rows, columns) in [
        (1, 1),
        (1, 129),
        (129, 1),
        (31, 33),
        (33, 31),
        (63, 65),
        (65, 63),
        (127, 129),
        (129, 127),
    ] {
        // The matrix sits inside a wider row-major buffer so that the pitch
        // exceeds the column count and the trailing row ends before its pitch.
        for pitch in [columns, columns + 5] {
            let buffer: Vec<usize> = (0..(rows - 1) * pitch + columns).collect();
            let source: Vec<usize> = (0..rows)
                .flat_map(|row| buffer[row * pitch..row * pitch + columns].iter().copied())
                .collect();
            let mut oracle = vec![usize::MAX; rows * columns];
            transpose_copy(&source, &mut oracle, rows, columns).expect("dense oracle");

            let mut assembled = vec![usize::MAX; rows * columns];
            for (start, end) in column_windows(columns) {
                let width = end - start;
                let span = (rows - 1) * pitch + width;
                transpose_copy_strided(
                    &buffer[start..start + span],
                    pitch,
                    &mut assembled[start * rows..end * rows],
                    rows,
                    width,
                )
                .expect("a column window over its exact span is valid");
                for guard in &assembled[end * rows..] {
                    assert_eq!(
                        *guard,
                        usize::MAX,
                        "shape {rows}x{columns}, window {start}..{end}"
                    );
                }
            }
            assert_eq!(assembled, oracle, "shape {rows}x{columns}, pitch {pitch}");
        }
    }
}

#[test]
fn strided_transpose_validates_product_destination_pitch_span_then_extent() {
    let source = [1, 2, 3, 4, 5, 6];
    let mut destination = [0; 4];

    assert_eq!(
        transpose_copy_strided(&source, 3, &mut destination, usize::MAX, 2),
        Err(LetoError::Overflow {
            reason: "strided transpose element count"
        })
    );
    assert_eq!(
        transpose_copy_strided(&source, 3, &mut destination[..3], 2, 2),
        Err(LetoError::StorageError {
            reason: "strided transpose destination length 3 does not match expected 4".to_owned()
        })
    );
    assert_eq!(
        transpose_copy_strided(&source, 1, &mut destination, 2, 2),
        Err(LetoError::StorageError {
            reason: "strided transpose source pitch 1 is narrower than 2 columns".to_owned()
        })
    );
    // Rows 0..2 of a pitch-3 matrix with a 2-wide window span 5 elements.
    assert_eq!(
        transpose_copy_strided(&source[..4], 3, &mut destination, 2, 2),
        Err(LetoError::StorageError {
            reason:
                "strided transpose source length 4 is shorter than the 5 elements the window spans"
                    .to_owned()
        })
    );
    assert_eq!(
        destination, [0; 4],
        "rejection leaves the destination unchanged"
    );

    transpose_copy_strided(&source[1..], 3, &mut destination, 2, 2)
        .expect("the exact span is valid");
    assert_eq!(destination, [2, 5, 3, 6]);

    // An empty window needs no source at all and no pitch relation beyond the
    // column count.
    let mut empty: [usize; 0] = [];
    for (rows, columns, pitch) in [(0, 5, 5), (3, 0, 0), (0, usize::MAX, usize::MAX)] {
        transpose_copy_strided::<usize>(&[], pitch, &mut empty, rows, columns)
            .expect("an empty window reads nothing");
    }
    assert_eq!(
        transpose_copy_strided::<usize>(&[], 4, &mut empty, 0, 5),
        Err(LetoError::StorageError {
            reason: "strided transpose source pitch 4 is narrower than 5 columns".to_owned()
        })
    );
}

#[test]
fn strided_transpose_rejects_unrepresentable_zero_sized_extents() {
    // Vec<()> represents these lengths without allocation or element work.
    let count = usize::try_from(isize::MAX).expect("isize::MAX fits usize") + 1;
    let source = vec![(); count];
    let mut destination = vec![(); count];
    assert_eq!(
        transpose_copy_strided(&source, 1, &mut destination, count, 1),
        Err(LetoError::Overflow {
            reason: "strided transpose signed layout extent"
        })
    );
    // At a pitch of three the leading rows alone overflow `usize`, before the
    // extent is reached; at two they fit, and the span is merely unmet.
    assert_eq!(
        transpose_copy_strided(&source, 2, &mut destination, count, 1),
        Err(LetoError::StorageError {
            reason: format!(
                "strided transpose source length {count} is shorter than the {} elements the window spans",
                (count - 1) * 2 + 1
            )
        })
    );
    assert_eq!(
        transpose_copy_strided(&source, 3, &mut destination, count, 1),
        Err(LetoError::Overflow {
            reason: "strided transpose source span"
        })
    );
}
