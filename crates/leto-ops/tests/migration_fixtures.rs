use eunomia::{Complex32, Complex64, F16};
use leto::{Array1, Array2, Array3, SliceArg, Storage};
use leto_ops::{add, mapv, matmul, mul, sum_axis_into};

fn assert_close(lhs: f32, rhs: f32) {
    assert!(
        (lhs - rhs).abs() <= 1.0e-5,
        "left {lhs} differs from right {rhs}"
    );
}

#[test]
fn apollo_fft_1d_complex_arrays_support_generation_mapping_and_output_storage() {
    let signal = Array1::from_shape_fn([8], |[index]| {
        let x = index as f64 * 0.25;
        Complex64::new(x.cos(), x.sin())
    });
    let converted = mapv(&signal.view(), |value| {
        Complex32::new(value.re as f32, value.im as f32)
    })
    .unwrap();
    let mut spectrum = Array1::<Complex32>::zeros([8]);

    for index in 0..8 {
        *spectrum.get_mut([index]).unwrap() = *converted.get([index]).unwrap();
    }

    assert_eq!(spectrum.shape(), [8]);
    assert_close(spectrum.get([0]).unwrap().re, 1.0);
    assert_close(spectrum.get([0]).unwrap().im, 0.0);
    assert_close(spectrum.get([4]).unwrap().re, 1.0f32.cos());
    assert_close(spectrum.get([4]).unwrap().im, 1.0f32.sin());
}

#[test]
fn apollo_fft_2d_and_3d_real_fields_support_generated_and_zeroed_outputs() {
    let field_2d = Array2::from_shape_fn([4, 5], |[row, col]| ((row + col) as f64 * 0.3).sin());
    let mut spectrum_2d = Array2::<Complex64>::zeros([4, 5]);
    for row in 0..4 {
        for col in 0..5 {
            let value = *field_2d.get([row, col]).unwrap();
            *spectrum_2d.get_mut([row, col]).unwrap() = Complex64::new(value, 0.0);
        }
    }

    let field_3d = Array3::from_shape_fn([3, 4, 5], |[x, y, z]| ((x + y + z) as f64 * 0.2).cos());
    let recovered_3d = mapv(&field_3d.view(), |value| value).unwrap();

    assert_eq!(spectrum_2d.shape(), [4, 5]);
    assert_eq!(recovered_3d.shape(), [3, 4, 5]);
    assert_eq!(
        recovered_3d.storage().as_slice(),
        field_3d.storage().as_slice()
    );
}

#[test]
fn apollo_fft_three_axis_lane_mutation_uses_leto_views_without_ndarray() {
    let base = || Array3::from_shape_fn([2, 3, 4], |[x, y, z]| (x * 12 + y * 4 + z) as i32);

    let mut axis2 = base();
    for x in 0..2 {
        for y in 0..3 {
            let mut lane = axis2
                .view_mut()
                .slice_with_mut::<1>(&[
                    SliceArg::Index(x as isize),
                    SliceArg::Index(y as isize),
                    SliceArg::All,
                ])
                .unwrap();
            for z in 0..4 {
                *lane.get_mut([z]).unwrap() += 100 + x * 10 + y;
            }
        }
    }
    for x in 0..2 {
        for y in 0..3 {
            for z in 0..4 {
                let expected = (x * 12 + y * 4 + z) as i32 + 100 + (x * 10 + y) as i32;
                assert_eq!(*axis2.get([x, y, z]).unwrap(), expected);
            }
        }
    }

    let mut axis1 = base();
    for x in 0..2 {
        for z in 0..4 {
            let mut lane = axis1
                .view_mut()
                .slice_with_mut::<1>(&[
                    SliceArg::Index(x as isize),
                    SliceArg::All,
                    SliceArg::Index(z as isize),
                ])
                .unwrap();
            for y in 0..3 {
                *lane.get_mut([y]).unwrap() += 200 + x * 10 + z;
            }
        }
    }
    for x in 0..2 {
        for y in 0..3 {
            for z in 0..4 {
                let expected = (x * 12 + y * 4 + z) as i32 + 200 + (x * 10 + z) as i32;
                assert_eq!(*axis1.get([x, y, z]).unwrap(), expected);
            }
        }
    }

    let mut axis0 = base();
    for y in 0..3 {
        for z in 0..4 {
            let mut lane = axis0
                .view_mut()
                .slice_with_mut::<1>(&[
                    SliceArg::All,
                    SliceArg::Index(y as isize),
                    SliceArg::Index(z as isize),
                ])
                .unwrap();
            for x in 0..2 {
                *lane.get_mut([x]).unwrap() += 300 + y * 10 + z;
            }
        }
    }
    for x in 0..2 {
        for y in 0..3 {
            for z in 0..4 {
                let expected = (x * 12 + y * 4 + z) as i32 + 300 + (y * 10 + z) as i32;
                assert_eq!(*axis0.get([x, y, z]).unwrap(), expected);
            }
        }
    }
}

#[test]
fn apollo_precision_conversion_fixture_supports_reduced_pair_storage() {
    let input = Array1::from_shape_fn([4], |[index]| {
        let value = index as f64 + 0.5;
        Complex64::new(value, -value)
    });

    let reduced_pairs = mapv(&input.view(), |value| {
        [
            F16::from_f32(value.re as f32),
            F16::from_f32(value.im as f32),
        ]
    })
    .unwrap();

    assert_eq!(reduced_pairs.shape(), [4]);
    assert_eq!(reduced_pairs.get([2]).unwrap()[0], F16::from_f32(2.5));
    assert_eq!(reduced_pairs.get([2]).unwrap()[1], F16::from_f32(-2.5));
}

#[test]
fn coeus_keepdim_reduction_broadcast_and_elementwise_fixture() {
    let activations =
        Array2::from_shape_vec([2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mut row_sums = Array2::<f32>::zeros([2, 1]);
    sum_axis_into(&activations.view(), 1, &mut row_sums.view_mut()).unwrap();

    let row_sums_broadcast = row_sums.view().broadcast([2, 3]).unwrap();
    let mut normalized = Array2::<f32>::zeros([2, 3]);
    let mut scaled = Array2::<f32>::zeros([2, 3]);

    add(
        &activations.view(),
        &row_sums_broadcast,
        &mut normalized.view_mut(),
    )
    .unwrap();
    mul(
        &activations.view(),
        &row_sums_broadcast,
        &mut scaled.view_mut(),
    )
    .unwrap();

    assert_eq!(row_sums.storage().as_slice(), &[6.0, 15.0]);
    assert_eq!(
        normalized.storage().as_slice(),
        &[7.0, 8.0, 9.0, 19.0, 20.0, 21.0]
    );
    assert_eq!(
        scaled.storage().as_slice(),
        &[6.0, 12.0, 18.0, 60.0, 75.0, 90.0]
    );
}

#[test]
fn coeus_tensor_matmul_fixture_matches_dense_layer_shape() {
    let batch = Array2::from_shape_vec([2, 3], vec![1.0f32, -2.0, 3.0, 4.0, 0.5, -6.0]).unwrap();
    let weights = Array2::from_shape_vec(
        [3, 4],
        vec![
            0.5, 1.0, -1.0, 2.0, 1.5, -0.5, 0.25, 3.0, -2.0, 4.0, 0.75, -1.5,
        ],
    )
    .unwrap();
    let mut output = Array2::<f32>::zeros([2, 4]);

    matmul(&batch.view(), &weights.view(), &mut output.view_mut()).unwrap();

    assert_eq!(output.shape(), [2, 4]);
    assert_eq!(
        output.storage().as_slice(),
        &[-8.5, 14.0, 0.75, -8.5, 14.75, -20.25, -8.375, 18.5]
    );
}

#[test]
fn apollo_dht_real_transform_simulation() {
    let n = 8;
    let signal = Array1::from_shape_fn([n], |[i]| (i as f64 * 0.5).sin());
    let mut dht = Array1::<f64>::zeros([n]);

    for k in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            let angle = 2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
            let cas = angle.cos() + angle.sin();
            sum += signal.get([j]).unwrap() * cas;
        }
        *dht.get_mut([k]).unwrap() = sum;
    }

    assert_eq!(dht.shape(), [n]);
    let sig_energy: f64 = signal.storage().as_slice().iter().map(|&x| x * x).sum();
    let dht_energy: f64 = dht.storage().as_slice().iter().map(|&x| x * x).sum();
    assert!((sig_energy - dht_energy / n as f64).abs() < 1.0e-5);
}

#[test]
fn apollo_ntt_modular_transform_simulation() {
    let n = 8;
    let q = 17;
    let root = 9;
    let signal = Array1::from_shape_fn([n], |[i]| (i as u64) % q);
    let mut ntt = Array1::<u64>::zeros([n]);

    for k in 0..n {
        let mut sum = 0;
        for j in 0..n {
            let exponent = (j * k) % n;
            let mut w = 1;
            for _ in 0..exponent {
                w = (w * root) % q;
            }
            sum = (sum + signal.get([j]).unwrap() * w) % q;
        }
        *ntt.get_mut([k]).unwrap() = sum;
    }

    assert_eq!(ntt.shape(), [n]);
    let expected_sum: u64 = signal.storage().as_slice().iter().sum::<u64>() % q;
    assert_eq!(*ntt.get([0]).unwrap(), expected_sum);
}

#[test]
fn apollo_nufft_non_uniform_grid_interpolation() {
    let m = 5;
    let n = 16;

    let coords = Array1::<f64>::from_shape_vec([m], vec![1.2, 4.7, 8.0, 11.5, 14.1]).unwrap();
    let strengths =
        Array1::from_shape_fn([m], |[i]| Complex64::new((i + 1) as f64, -((i + 1) as f64)));

    let mut grid = Array1::<Complex64>::zeros([n]);
    let width = 2.0;

    for i in 0..m {
        let x = *coords.get([i]).unwrap();
        let c = *strengths.get([i]).unwrap();

        let x_center = x.round() as isize;
        for offset in -2..=2 {
            let g_idx = ((x_center + offset).rem_euclid(n as isize)) as usize;
            let dist = x - (x_center + offset) as f64;
            let weight = (-dist * dist / (2.0 * width * width)).exp();

            let current = grid.get([g_idx]).unwrap();
            *grid.get_mut([g_idx]).unwrap() =
                Complex64::new(current.re + c.re * weight, current.im + c.im * weight);
        }
    }

    assert_eq!(grid.shape(), [n]);
    assert!(grid.get([1]).unwrap().re > 0.0);
    assert!(grid.get([8]).unwrap().re > 0.0);
}

#[test]
fn apollo_sht_spherical_harmonic_grid_eval() {
    let n_theta = 4;
    let n_phi = 8;

    let theta = Array1::from_shape_fn([n_theta], |[i]| {
        std::f64::consts::PI * (i as f64 + 0.5) / n_theta as f64
    });
    let _phi = Array1::from_shape_fn([n_phi], |[j]| {
        2.0 * std::f64::consts::PI * j as f64 / n_phi as f64
    });

    let factor = 0.5 * (3.0 / std::f64::consts::PI).sqrt();
    let mut grid = Array2::<f64>::zeros([n_theta, n_phi]);

    for i in 0..n_theta {
        let t = *theta.get([i]).unwrap();
        let val = factor * t.cos();
        for j in 0..n_phi {
            *grid.get_mut([i, j]).unwrap() = val;
        }
    }

    assert_eq!(grid.shape(), [n_theta, n_phi]);
    for i in 0..n_theta {
        let reference = *grid.get([i, 0]).unwrap();
        for j in 1..n_phi {
            assert_close(reference as f32, *grid.get([i, j]).unwrap() as f32);
        }
    }
}

#[test]
fn apollo_wgpu_host_mapped_buffer_verification() {
    let arr = Array3::from_shape_fn([2, 3, 4], |[x, y, z]| (x * 12 + y * 4 + z) as f32);

    assert_eq!(arr.strides(), [12, 4, 1]);

    let slice = arr.storage().as_slice();
    assert_eq!(slice.len(), 24);
    assert_eq!(slice[0], 0.0);
    assert_eq!(slice[23], 23.0);

    let byte_len = std::mem::size_of_val(slice);
    assert_eq!(byte_len, 96);
}

#[test]
fn apollo_python_bindings_strides_compatibility() {
    let arr = Array2::from_shape_vec([3, 4], (0..12).collect()).unwrap();

    let elem_strides = arr.strides();
    assert_eq!(elem_strides, [4, 1]);

    let item_size = std::mem::size_of::<i32>() as isize;
    let byte_strides = [elem_strides[0] * item_size, elem_strides[1] * item_size];
    assert_eq!(byte_strides, [16, 4]);

    let view = arr.view();
    let transposed = view.transpose([1, 0]).unwrap();
    assert_eq!(transposed.shape(), [4, 3]);
    assert_eq!(transposed.strides(), [1, 4]);

    let transposed_byte_strides = [
        transposed.strides()[0] * item_size,
        transposed.strides()[1] * item_size,
    ];
    assert_eq!(transposed_byte_strides, [4, 16]);
}

#[test]
fn coeus_tensor_layout_and_broadcasting() {
    let a = Array2::from_shape_vec([2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(a.shape(), [2, 3]);
    assert_eq!(a.strides(), [3, 1]);

    let a_t = a.view().transpose([1, 0]).unwrap();
    assert_eq!(a_t.shape(), [3, 2]);
    assert_eq!(a_t.strides(), [1, 3]);

    let row = Array2::from_shape_vec([1, 3], vec![10.0f32, 20.0, 30.0]).unwrap();
    let broadcasted = row.view().broadcast([2, 3]).unwrap();
    assert_eq!(broadcasted.shape(), [2, 3]);
    assert_eq!(broadcasted.strides(), [0, 1]);

    assert_eq!(*broadcasted.get([0, 0]).unwrap(), 10.0);
    assert_eq!(*broadcasted.get([0, 2]).unwrap(), 30.0);
    assert_eq!(*broadcasted.get([1, 0]).unwrap(), 10.0);
    assert_eq!(*broadcasted.get([1, 2]).unwrap(), 30.0);
}

#[test]
fn coeus_non_differentiable_storage_boundaries() {
    use leto::{Array, CowStorage, Layout};

    let backing = vec![1.0f32, 2.0, 3.0, 4.0];
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let mut array = Array::new(layout, CowStorage::borrowed(&backing)).unwrap();

    assert!(array.storage().is_borrowed());
    assert_eq!(*array.get([1, 0]).unwrap(), 3.0);

    *array.get_mut([0, 1]).unwrap() = 20.0;

    assert!(array.storage().is_owned());
    assert_eq!(array.storage().as_owned().unwrap(), &[1.0, 20.0, 3.0, 4.0]);
}
