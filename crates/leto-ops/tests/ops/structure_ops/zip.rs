//! Multi-input zip_mut_with structural semantics.

use super::*;

#[test]
fn test_zip_mut_with_fused_multiply_add() {
    // out = out + a * b, three-operand fused update.
    let mut out = arr([2, 2], vec![1.0, 1.0, 1.0, 1.0]);
    let a = arr([2, 2], vec![2.0, 3.0, 4.0, 5.0]);
    let b = arr([2, 2], vec![10.0, 10.0, 10.0, 10.0]);
    zip_mut_with(out.view_mut(), (&a.view(), &b.view()), |o, (&x, &y)| {
        *o += x * y;
    })
    .unwrap();
    assert_eq!(out.storage().as_slice(), &[21.0, 31.0, 41.0, 51.0]);
}

#[test]
fn test_zip_mut_with_strided_input() {
    // a is a transposed (strided) view; traversal must follow logical order.
    let mut out = arr([3, 2], vec![0.0; 6]);
    let a_src = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let a = a_src.transpose([1, 0]).unwrap(); // logical [[1,4],[2,5],[3,6]]
    let b = arr([3, 2], vec![100.0, 100.0, 100.0, 100.0, 100.0, 100.0]);
    zip_mut_with(out.view_mut(), (&a, &b.view()), |o, (&x, &y)| {
        *o = x + y;
    })
    .unwrap();
    assert_eq!(
        out.storage().as_slice(),
        &[101.0, 104.0, 102.0, 105.0, 103.0, 106.0]
    );
}

#[test]
fn test_zip_mut_with_three_inputs() {
    let mut out = arr([2, 2], vec![0.0; 4]);
    let prev = arr([2, 2], vec![1.0, 4.0, 9.0, 16.0]);
    let curr = arr([2, 2], vec![2.0, 5.0, 10.0, 17.0]);
    let next = arr([2, 2], vec![4.0, 8.0, 14.0, 22.0]);

    zip_mut_with(
        out.view_mut(),
        (&prev.view(), &curr.view(), &next.view()),
        |d, (&p0, &p1, &p2)| {
            *d = 2.0f64.mul_add(-p1, p0) + p2;
        },
    )
    .unwrap();

    assert_eq!(out.storage().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_zip_mut_with_three_strided_inputs_follow_logical_order() {
    let mut out = arr([3, 2], vec![0.0; 6]);
    let prev_src = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let curr_src = arr([2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let next_src = arr([2, 3], vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0]);
    let prev = prev_src.transpose([1, 0]).unwrap();
    let curr = curr_src.transpose([1, 0]).unwrap();
    let next = next_src.transpose([1, 0]).unwrap();

    zip_mut_with(out.view_mut(), (&prev, &curr, &next), |d, (&a, &b, &c)| {
        *d = a + b + c;
    })
    .unwrap();

    assert_eq!(
        out.storage().as_slice(),
        &[111.0, 444.0, 222.0, 555.0, 333.0, 666.0]
    );
}

#[test]
fn test_zip_mut_with_multiple_outputs_and_no_sources() {
    let mut first = arr([2, 2], vec![0.0; 4]);
    let mut second = arr([2, 2], vec![0.0; 4]);

    zip_mut_with(
        (first.view_mut(), second.view_mut()),
        (),
        |(first, second), ()| {
            *first = 2.0;
            *second = 3.0;
        },
    )
    .unwrap();

    assert_eq!(first.storage().as_slice(), &[2.0; 4]);
    assert_eq!(second.storage().as_slice(), &[3.0; 4]);
}

#[test]
fn test_indexed_zip_mut_with_multiple_outputs_and_no_sources() {
    let mut first = arr([2, 3], vec![0.0; 6]);
    let mut second = arr([2, 3], vec![0.0; 6]);
    let mut third = arr([2, 3], vec![0.0; 6]);

    indexed_zip_mut_with(
        (first.view_mut(), second.view_mut(), third.view_mut()),
        (),
        |[i, j], (first, second, third), ()| {
            let value = (10 * i + j) as f64;
            *first = value;
            *second = value + 1.0;
            *third = value + 2.0;
        },
    )
    .unwrap();

    assert_eq!(
        first.storage().as_slice(),
        &[0.0, 1.0, 2.0, 10.0, 11.0, 12.0]
    );
    assert_eq!(
        second.storage().as_slice(),
        &[1.0, 2.0, 3.0, 11.0, 12.0, 13.0]
    );
    assert_eq!(
        third.storage().as_slice(),
        &[2.0, 3.0, 4.0, 12.0, 13.0, 14.0]
    );
}

#[test]
fn test_zip_mut_with_strided_outputs_and_tuple_sources() {
    let mut first = arr([2, 3], vec![0.0; 6]);
    let mut second = arr([2, 3], vec![0.0; 6]);
    let first_source = arr([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let second_source = arr([3, 2], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

    {
        let first_view = first.transpose_mut([1, 0]).unwrap();
        let second_view = second.transpose_mut([1, 0]).unwrap();
        zip_mut_with(
            (first_view, second_view),
            (&first_source.view(), &second_source.view()),
            |(first, second), (&first_source, &second_source)| {
                *first = first_source;
                *second = second_source;
            },
        )
        .unwrap();
    }

    assert_eq!(first.storage().as_slice(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    assert_eq!(
        second.storage().as_slice(),
        &[10.0, 30.0, 50.0, 20.0, 40.0, 60.0]
    );
}

#[test]
fn test_zip_mut_with_five_inputs() {
    let mut out = arr([2, 2], vec![0.0; 4]);
    let a = arr([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = arr([2, 2], vec![10.0, 20.0, 30.0, 40.0]);
    let c = arr([2, 2], vec![100.0, 200.0, 300.0, 400.0]);
    let d = arr([2, 2], vec![1.0, 1.0, 1.0, 1.0]);
    let e = arr([2, 2], vec![2.0, 2.0, 2.0, 2.0]);

    zip_mut_with(
        out.view_mut(),
        (&a.view(), &b.view(), &c.view(), &d.view(), &e.view()),
        |o, (&av, &bv, &cv, &dv, &ev)| *o = av + bv - cv + dv * ev,
    )
    .unwrap();

    assert_eq!(out.storage().as_slice(), &[-87.0, -176.0, -265.0, -354.0]);
}

#[test]
fn test_zip_mut_with_five_strided_inputs_follow_logical_order() {
    let mut out = arr([3, 2], vec![0.0; 6]);
    let a_src = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b_src = arr([2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let c_src = arr([2, 3], vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0]);
    let d_src = arr([2, 3], vec![1.0; 6]);
    let e_src = arr([2, 3], vec![2.0; 6]);
    let a = a_src.transpose([1, 0]).unwrap();
    let b = b_src.transpose([1, 0]).unwrap();
    let c = c_src.transpose([1, 0]).unwrap();
    let d = d_src.transpose([1, 0]).unwrap();
    let e = e_src.transpose([1, 0]).unwrap();

    zip_mut_with(
        out.view_mut(),
        (&a, &b, &c, &d, &e),
        |o, (&av, &bv, &cv, &dv, &ev)| {
            *o = av + bv - cv + dv * ev;
        },
    )
    .unwrap();

    assert_eq!(
        out.storage().as_slice(),
        &[-87.0, -354.0, -176.0, -443.0, -265.0, -532.0]
    );
}

#[test]
fn test_zip_mut_with_preserves_heterogeneous_source_types() {
    let mut out = arr([2, 2], vec![0.0; 4]);
    let integer = Array::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();
    let scale = arr([2, 2], vec![0.5, 1.5, 2.5, 3.5]);

    zip_mut_with(
        out.view_mut(),
        (&integer.view(), &scale.view()),
        |value, (&integer, &scale)| *value = f64::from(integer) + scale,
    )
    .unwrap();

    assert_eq!(out.storage().as_slice(), &[1.5, 3.5, 5.5, 7.5]);
}
