use leto::Array3;

#[test]
fn array_index_fill_map_and_zip_are_value_semantic() {
    let mut values = Array3::from_shape_vec([2, 2, 2], (0..8).collect::<Vec<i32>>())
        .expect("shape matches storage length");

    assert_eq!(values[[0, 0, 0]], 0);
    assert_eq!(values[[1, 1, 1]], 7);

    values[[1, 0, 1]] = 42;
    assert_eq!(values[[1, 0, 1]], 42);

    let doubled = values.mapv(|value| value * 2);
    assert_eq!(doubled[[1, 0, 1]], 84);

    let summed = values.zip_map(&doubled, |left, right| left + right);
    assert_eq!(summed[[1, 0, 1]], 126);

    values.fill(-3);
    assert!(values.iter().all(|&value| value == -3));
}
