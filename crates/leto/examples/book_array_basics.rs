//! Basic array creation, indexing, and element-wise arithmetic.
//!
//! [`Array1`] and [`Array2`] are the rank-1 and rank-2 aliases for
//! [`leto::Array`]. Values are stored in row-major order in a heap-backed
//! [`VecStorage`]. Zero-copy views borrow from the same storage.

extern crate leto;

use leto::{sum_all, Array1, Array2};

fn main() {
    let zeros: Array1<f64> = Array1::zeros(8);
    assert_eq!(sum_all(&zeros).expect("valid sum"), 0.0);
    println!("zeros(8) sum = {}", sum_all(&zeros).expect("valid sum"));

    let ones: Array1<f64> = Array1::ones(8);
    assert!((sum_all(&ones).expect("valid sum") - 8.0).abs() < 1e-10);
    println!("ones(8) sum  = {}", sum_all(&ones).expect("valid sum"));

    let a: Array1<f64> = Array1::from_vec(8, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("shape matches vector length");
    println!("a = {:?}", a.as_slice());
    assert_eq!(a[[0]], 1.0);
    assert_eq!(a[[7]], 8.0);

    let doubled = a.mapv(|x| x * 2.0);
    assert_eq!(doubled[[3]], 8.0);
    println!("a[3]*2 = {}", doubled[[3]]);

    let matrix: Array2<f64> = Array2::zeros((3, 3));
    assert_eq!(sum_all(&matrix).expect("valid sum"), 0.0);
    println!("3×3 zeros sum = {}", sum_all(&matrix).expect("valid sum"));

    let eye: Array2<f64> = Array2::eye(4);
    assert_eq!(sum_all(&eye).expect("valid sum"), 4.0);
    println!("eye(4) sum = {}", sum_all(&eye).expect("valid sum"));
    assert_eq!(eye[[0, 0]], 1.0);
    assert_eq!(eye[[0, 1]], 0.0);
    assert_eq!(eye[[1, 1]], 1.0);

    println!("all array-basics assertions passed");
}
