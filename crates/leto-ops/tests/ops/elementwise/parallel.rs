//! Parallel-route parity tests above the element-count gate.

use super::*;

/// Values `0.5, 1.0, 1.5, …`, built without a cast so the test carries no
/// precision-losing conversion.
fn ramp(len: usize, step: f64) -> Vec<f64> {
    let mut value = 0.0;
    (0..len)
        .map(|_| {
            value += step;
            value
        })
        .collect()
}

/// Above the element-count gate of the unary paths (`PARALLEL_THRESHOLD`), so
/// `map_into` and `map_inplace` take their parallel dense route.
const PARALLEL_LEN: usize = 1 << 17;

#[test]
fn parallel_unary_dense_routes_match_the_serial_result() {
    let source_values = ramp(PARALLEL_LEN, 0.5);
    let source = Array::from_shape_vec([PARALLEL_LEN], source_values.clone()).unwrap();

    let mut mapped = Array::from_shape_vec([PARALLEL_LEN], vec![0.0_f64; PARALLEL_LEN]).unwrap();
    map_into(&source.view(), &mut mapped.view_mut(), |x| {
        x.mul_add(3.0, -1.0)
    })
    .unwrap();
    for (index, (&got, &x)) in mapped
        .storage()
        .as_slice()
        .iter()
        .zip(&source_values)
        .enumerate()
    {
        assert_eq!(
            got.to_bits(),
            x.mul_add(3.0, -1.0).to_bits(),
            "map_into[{index}]"
        );
    }

    let mut in_place = Array::from_shape_vec([PARALLEL_LEN], source_values.clone()).unwrap();
    map_inplace(&mut in_place.view_mut(), |x| x.mul_add(-2.0, 0.25)).unwrap();
    for (index, (&got, &x)) in in_place
        .storage()
        .as_slice()
        .iter()
        .zip(&source_values)
        .enumerate()
    {
        assert_eq!(
            got.to_bits(),
            x.mul_add(-2.0, 0.25).to_bits(),
            "map_inplace[{index}]"
        );
    }
}

#[test]
fn parallel_binary_dense_route_matches_the_serial_result() {
    // The binary gate is a working set past the last-level cache, so the
    // length comes from the probed geometry rather than a fixed count: three
    // operands of `len` doubles, one page group past the capacity.
    let capacity = leto_ops::cached_cache_geometry().l3_bytes();
    let len = capacity / (3 * core::mem::size_of::<f64>()) + (1 << 16);

    let lhs = Array::from_shape_vec([len], ramp(len, 0.5)).unwrap();
    let rhs = Array::from_shape_vec([len], ramp(len, -0.25)).unwrap();
    let mut out = Array::from_shape_vec([len], vec![0.0_f64; len]).unwrap();
    add(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();

    // Every element of the output is the sum of the two operands at the same
    // position: a dropped, doubled or shifted task run shows up here.
    for (index, ((&got, &x), &y)) in out
        .storage()
        .as_slice()
        .iter()
        .zip(lhs.storage().as_slice())
        .zip(rhs.storage().as_slice())
        .enumerate()
    {
        assert_eq!(got.to_bits(), (x + y).to_bits(), "add[{index}]");
    }
}

/// Rows and columns whose product puts three `f64` operands past the probed
/// last-level cache, which is the gate the strided elementwise paths take.
fn strided_parallel_shape(operands: usize) -> (usize, usize) {
    let capacity = leto_ops::cached_cache_geometry().l3_bytes();
    let elements = capacity / (operands * core::mem::size_of::<f64>()) + (1 << 16);
    let columns = 64;
    (elements.div_ceil(columns), columns)
}

#[test]
fn parallel_strided_binary_map_matches_the_serial_result() {
    // A transposed view walks its last axis by a row stride, which takes the
    // tiled block arm; the output is contiguous in the walked order.
    let (rows, columns) = strided_parallel_shape(3);
    let len = rows * columns;
    let lhs = Array::from_shape_vec([rows, columns], ramp(len, 0.5)).unwrap();
    let rhs = Array::from_shape_vec([rows, columns], ramp(len, -0.25)).unwrap();
    let mut out = Array::from_shape_vec([columns, rows], vec![0.0_f64; len]).unwrap();

    add(
        &lhs.transpose([1, 0]).unwrap(),
        &rhs.transpose([1, 0]).unwrap(),
        &mut out.view_mut(),
    )
    .unwrap();

    // Output element (column, row) is the sum at (row, column) of both inputs:
    // a dropped, doubled or shifted task run shows up as a mismatch here.
    let (lhs_values, rhs_values) = (lhs.storage().as_slice(), rhs.storage().as_slice());
    for (index, &got) in out.storage().as_slice().iter().enumerate() {
        let (column, row) = (index / rows, index % rows);
        let source = row * columns + column;
        assert_eq!(
            got.to_bits(),
            (lhs_values[source] + rhs_values[source]).to_bits(),
            "add[{column}, {row}]"
        );
    }
}

#[test]
fn parallel_strided_unary_map_matches_the_serial_result() {
    let (rows, columns) = strided_parallel_shape(2);
    let len = rows * columns;
    let source = Array::from_shape_vec([rows, columns], ramp(len, 0.5)).unwrap();
    let mut out = Array::from_shape_vec([columns, rows], vec![0.0_f64; len]).unwrap();

    map_into(
        &source.transpose([1, 0]).unwrap(),
        &mut out.view_mut(),
        |x| x.mul_add(3.0, -1.0),
    )
    .unwrap();

    let values = source.storage().as_slice();
    for (index, &got) in out.storage().as_slice().iter().enumerate() {
        let (column, row) = (index / rows, index % rows);
        let x = values[row * columns + column];
        assert_eq!(
            got.to_bits(),
            x.mul_add(3.0, -1.0).to_bits(),
            "map_into[{column}, {row}]"
        );
    }
}

#[test]
fn parallel_row_walk_arms_match_the_serial_result() {
    // A padded source sliced back to the output width keeps its last-axis
    // stride at one element while its row stride differs from the output's,
    // which is the row-walk arm rather than the tiled block arm.
    const PAD: usize = 3;
    let (rows, columns) = strided_parallel_shape(3);
    let padded_columns = columns + PAD;
    let padded_len = rows * padded_columns;
    let len = rows * columns;

    let lhs = Array::from_shape_vec([rows, padded_columns], ramp(padded_len, 0.5)).unwrap();
    let rhs = Array::from_shape_vec([rows, padded_columns], ramp(padded_len, -0.25)).unwrap();
    let lhs_view = lhs.slice(&[(0, rows, 1), (0, columns, 1)]).unwrap();
    let rhs_view = rhs.slice(&[(0, rows, 1), (0, columns, 1)]).unwrap();

    let mut out = Array::from_shape_vec([rows, columns], vec![0.0_f64; len]).unwrap();
    add(&lhs_view, &rhs_view, &mut out.view_mut()).unwrap();

    let (lhs_values, rhs_values) = (lhs.storage().as_slice(), rhs.storage().as_slice());
    for (index, &got) in out.storage().as_slice().iter().enumerate() {
        let (row, column) = (index / columns, index % columns);
        let source = row * padded_columns + column;
        assert_eq!(
            got.to_bits(),
            (lhs_values[source] + rhs_values[source]).to_bits(),
            "add[{row}, {column}]"
        );
    }

    // The unary row walk, on the same shape.
    let mut mapped = Array::from_shape_vec([rows, columns], vec![0.0_f64; len]).unwrap();
    map_into(&lhs_view, &mut mapped.view_mut(), |x| x.mul_add(3.0, -1.0)).unwrap();
    for (index, &got) in mapped.storage().as_slice().iter().enumerate() {
        let (row, column) = (index / columns, index % columns);
        let x = lhs_values[row * padded_columns + column];
        assert_eq!(
            got.to_bits(),
            x.mul_add(3.0, -1.0).to_bits(),
            "map_into[{row}, {column}]"
        );
    }
}
