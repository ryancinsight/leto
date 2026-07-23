use leto::geometry::{Point1, Point2, Point3, Vector3};

#[test]
fn point1_constructs_and_round_trips_through_serde() {
    let point = Point1::new(3.5_f64);

    let encoded = serde_json::to_string(&point).unwrap();
    let decoded: Point1<f64> = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, point);
    assert_eq!(decoded.x, 3.5);
}

#[test]
fn fixed_points_expose_coordinates_by_value() {
    let point2 = Point2::new(1_i32, 2);
    let point3 = Point3::new(3_i32, 4, 5);

    assert_eq!((point2.x, point2.y), (1, 2));
    assert_eq!(point3.coords, Vector3::new(3, 4, 5));
    assert_eq!((point3.x, point3.y, point3.z), (3, 4, 5));
}
