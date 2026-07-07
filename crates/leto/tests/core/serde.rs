use leto::{Array2, Layout, Storage};

#[test]
fn owned_array_round_trips_shape_and_values_through_serde() {
    let original = Array2::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let encoded = serde_json::to_string(&original).unwrap();
    let decoded: Array2<f64> = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded.shape(), [2, 3]);
    assert_eq!(decoded.storage().as_slice(), original.storage().as_slice());
    assert_eq!(decoded[[1, 2]], 6.0);
}

#[test]
fn layout_round_trips_const_rank_above_serde_array_limit() {
    const RANK: usize = 33;
    let shape = [2usize; RANK];
    let strides = [1isize; RANK];
    let original = Layout::<RANK>::new(shape, strides, 7);

    let encoded = serde_json::to_string(&original).unwrap();
    let decoded: Layout<RANK> = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, original);
}

#[test]
fn layout_deserialize_rejects_rank_mismatch() {
    let encoded = r#"{"shape":[2,3],"strides":[3,1],"offset":0}"#;

    let error = serde_json::from_str::<Layout<3>>(encoded).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("layout shape rank 2 does not match expected rank 3"),
        "rank mismatch must surface the violated layout rank invariant: {error}"
    );
}
