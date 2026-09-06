use std::cell::Cell;
use std::fmt::Debug;
use std::rc::Rc;

use leto::{transpose_copy, LetoError};

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
