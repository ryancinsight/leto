use super::*;

fn forward_contract<T: RealScalar + RealField>(tolerance: f64) {
    let query = array(
        [1, 2, 2],
        [1.0, 0.0, 0.0, 1.0].map(FloatElement::from_f64).to_vec(),
    );
    let key = array(
        [1, 2, 2],
        [1.0, 0.0, 0.0, 1.0].map(FloatElement::from_f64).to_vec(),
    );
    let value = array(
        [1, 2, 2],
        [2.0, 0.0, 0.0, 4.0].map(FloatElement::from_f64).to_vec(),
    );
    let mut output = array([1, 2, 2], vec![<T as NumericElement>::ZERO; 4]);
    let mut weights = array([1, 2, 2], vec![<T as NumericElement>::ZERO; 4]);

    scaled_dot_product_attention_into(
        &query.view(),
        &key.view(),
        &value.view(),
        AttentionMask::Unmasked,
        <T as NumericElement>::ONE,
        &mut output.view_mut(),
        &mut weights.view_mut(),
    )
    .expect("valid attention contract");

    let dominant = core::f64::consts::E / (core::f64::consts::E + 1.0);
    let residual = 1.0 - dominant;
    assert_close(
        weights.storage().as_slice(),
        &[dominant, residual, residual, dominant],
        tolerance,
    );
    assert_close(
        output.storage().as_slice(),
        &[
            2.0 * dominant,
            4.0 * residual,
            2.0 * residual,
            4.0 * dominant,
        ],
        tolerance,
    );
}

#[test]
fn forward_matches_closed_form_for_all_native_precisions() {
    forward_contract::<f32>(64.0 * f64::from(f32::EPSILON));
    forward_contract::<f64>(64.0 * f64::EPSILON);
}

#[test]
fn broadcast_mask_and_causal_policy_do_not_materialize() {
    let query = array([1, 2, 1], vec![1.0_f64, 1.0]);
    let key = array([1, 2, 1], vec![1.0_f64, 2.0]);
    let value = array([1, 2, 1], vec![3.0_f64, 7.0]);
    let mask = array([1, 1, 2], vec![1.0_f64, 0.0]);
    let mut output = array([1, 2, 1], vec![9.0_f64; 2]);
    let mut weights = array([1, 2, 2], vec![9.0_f64; 4]);

    scaled_dot_product_attention_into(
        &query.view(),
        &key.view(),
        &value.view(),
        AttentionMask::CausalKeep(mask.view()),
        1.0,
        &mut output.view_mut(),
        &mut weights.view_mut(),
    )
    .expect("broadcast mask is valid");

    assert_eq!(output.storage().as_slice(), &[3.0, 3.0]);
    assert_eq!(weights.storage().as_slice(), &[1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn grouped_mask_repeats_borrowed_groups_and_composes_with_causal_policy() {
    let query = array([4, 2, 1], vec![1.0_f64; 8]);
    let key = array([4, 2, 1], vec![1.0_f64; 8]);
    let value = array([4, 2, 1], vec![3.0_f64, 7.0, 3.0, 7.0, 3.0, 7.0, 3.0, 7.0]);
    let keep = array2([2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]);
    let grouped = GroupedKeepMask::new(
        keep.view(),
        NonZeroUsize::new(2).expect("test group width is nonzero"),
    );
    let mut output = array([4, 2, 1], vec![9.0_f64; 8]);
    let mut weights = array([4, 2, 2], vec![9.0_f64; 16]);

    scaled_dot_product_attention_into(
        &query.view(),
        &key.view(),
        &value.view(),
        AttentionMask::GroupedKeep(grouped),
        1.0,
        &mut output.view_mut(),
        &mut weights.view_mut(),
    )
    .expect("grouped keep mask is valid");
    assert_eq!(
        output.storage().as_slice(),
        &[3.0, 3.0, 3.0, 3.0, 7.0, 7.0, 7.0, 7.0]
    );

    scaled_dot_product_attention_into(
        &query.view(),
        &key.view(),
        &value.view(),
        AttentionMask::CausalGroupedKeep(grouped),
        1.0,
        &mut output.view_mut(),
        &mut weights.view_mut(),
    )
    .expect("causal grouped keep mask is valid");
    assert_eq!(
        output.storage().as_slice(),
        &[3.0, 3.0, 3.0, 3.0, 0.0, 7.0, 0.0, 7.0]
    );
}

#[test]
fn fully_masked_rows_produce_zero_output_and_weights() {
    let query = array([1, 1, 1], vec![2.0_f64]);
    let key = array([1, 2, 1], vec![1.0_f64, 3.0]);
    let value = array([1, 2, 1], vec![5.0_f64, 7.0]);
    let mask = array([1, 1, 2], vec![0.0_f64, 0.0]);
    let mut output = array([1, 1, 1], vec![9.0_f64]);
    let mut weights = array([1, 1, 2], vec![9.0_f64; 2]);

    scaled_dot_product_attention_into(
        &query.view(),
        &key.view(),
        &value.view(),
        AttentionMask::Keep(mask.view()),
        1.0,
        &mut output.view_mut(),
        &mut weights.view_mut(),
    )
    .expect("fully masked rows have defined zero semantics");

    assert_eq!(output.storage().as_slice(), &[0.0]);
    assert_eq!(weights.storage().as_slice(), &[0.0, 0.0]);
}

#[test]
fn strided_inputs_and_outputs_match_contiguous_semantics() {
    let query_data = [0.0_f64, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let key_data = query_data;
    let value_data = [0.0_f64, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0];
    let layout = Layout::try_new([1, 2, 2], [8, 4, 2], 1).expect("valid test layout");
    let query = ArrayView::try_new(layout, &query_data).expect("valid strided query");
    let key = ArrayView::try_new(layout, &key_data).expect("valid strided key");
    let value = ArrayView::try_new(layout, &value_data).expect("valid strided value");
    let mut output_data = [99.0_f64; 8];
    let mut weight_data = [99.0_f64; 8];
    let mut output = ArrayViewMut::try_new(layout, &mut output_data).expect("valid strided output");
    let mut weights =
        ArrayViewMut::try_new(layout, &mut weight_data).expect("valid strided weights");

    scaled_dot_product_attention_into(
        &query,
        &key,
        &value,
        AttentionMask::Unmasked,
        1.0,
        &mut output,
        &mut weights,
    )
    .expect("strided attention contract");

    let dominant = core::f64::consts::E / (core::f64::consts::E + 1.0);
    let residual = 1.0 - dominant;
    assert_close(
        &[
            weight_data[1],
            weight_data[3],
            weight_data[5],
            weight_data[7],
        ],
        &[dominant, residual, residual, dominant],
        64.0 * f64::EPSILON,
    );
    assert_close(
        &[
            output_data[1],
            output_data[3],
            output_data[5],
            output_data[7],
        ],
        &[
            2.0 * dominant,
            4.0 * residual,
            2.0 * residual,
            4.0 * dominant,
        ],
        64.0 * f64::EPSILON,
    );
}

#[test]
fn injective_interleaved_outputs_are_accepted_without_materialization() {
    let query = array([1, 3, 1], vec![0.0_f64; 3]);
    let key = array([1, 2, 1], vec![0.0_f64; 2]);
    let value = array([1, 2, 2], vec![2.0_f64, 4.0, 6.0, 8.0]);
    let interleaved = Layout::try_new([1, 3, 2], [6, 2, 3], 0).expect("valid test layout");
    let mut output_data = [99.0_f64; 8];
    let mut weight_data = [99.0_f64; 8];
    let mut output = ArrayViewMut::try_new(interleaved, &mut output_data)
        .expect("interleaved output stays within storage");
    let mut weights = ArrayViewMut::try_new(interleaved, &mut weight_data)
        .expect("interleaved weights stay within storage");

    scaled_dot_product_attention_into(
        &query.view(),
        &key.view(),
        &value.view(),
        AttentionMask::Unmasked,
        1.0,
        &mut output,
        &mut weights,
    )
    .expect("injective interleaved destinations are valid");

    assert_close(
        &output.as_view().iter().copied().collect::<Vec<_>>(),
        &[4.0, 6.0, 4.0, 6.0, 4.0, 6.0],
        8.0 * f64::EPSILON,
    );
    assert_close(
        &weights.as_view().iter().copied().collect::<Vec<_>>(),
        &[0.5; 6],
        8.0 * f64::EPSILON,
    );
}

#[test]
fn aliased_mutable_outputs_are_rejected_before_writing() {
    let query = array([1, 2, 1], vec![0.0_f64; 2]);
    let key = array([1, 1, 1], vec![0.0_f64]);
    let value = array([1, 1, 1], vec![3.0_f64]);
    let aliased = Layout::try_new([1, 2, 1], [2, 0, 1], 0).expect("valid test layout");
    let mut output_data = [7.0_f64];
    let mut weights = array([1, 2, 1], vec![8.0_f64; 2]);

    let error = {
        let mut output =
            ArrayViewMut::try_new(aliased, &mut output_data).expect("aliased layout is in storage");
        scaled_dot_product_attention_into(
            &query.view(),
            &key.view(),
            &value.view(),
            AttentionMask::Unmasked,
            1.0,
            &mut output,
            &mut weights.view_mut(),
        )
        .expect_err("aliased mutable destinations cannot satisfy exclusive access")
    };

    assert!(matches!(
        error,
        AttentionError::Layout {
            operand: AttentionOperand::Output,
            ..
        }
    ));
    assert_eq!(output_data, [7.0]);
    assert_eq!(weights.storage().as_slice(), &[8.0, 8.0]);
}

#[test]
fn nonzero_stride_collisions_are_rejected_before_writing() {
    let query = array([1, 3, 1], vec![0.0_f64; 3]);
    let key = array([1, 2, 1], vec![0.0_f64; 2]);
    let value = array([1, 2, 2], vec![2.0_f64, 4.0, 6.0, 8.0]);
    let aliased = Layout::try_new([1, 3, 2], [6, 2, 4], 0).expect("valid test layout");
    let mut output_data = [7.0_f64; 9];
    let mut weights = array([1, 3, 2], vec![8.0_f64; 6]);

    let error = {
        let mut output = ArrayViewMut::try_new(aliased, &mut output_data)
            .expect("nonzero-stride collision stays within storage");
        scaled_dot_product_attention_into(
            &query.view(),
            &key.view(),
            &value.view(),
            AttentionMask::Unmasked,
            1.0,
            &mut output,
            &mut weights.view_mut(),
        )
        .expect_err("distinct logical output indices share physical offset four")
    };

    assert!(matches!(
        error,
        AttentionError::Layout {
            operand: AttentionOperand::Output,
            ..
        }
    ));
    assert_eq!(output_data, [7.0; 9]);
    assert_eq!(weights.storage().as_slice(), &[8.0; 6]);
}

#[test]
fn negative_strides_preserve_logical_attention_order() {
    let query_data = [1.0_f64, 0.0, 0.0, 1.0];
    let key_data = query_data;
    let value_data = [2.0_f64, 0.0, 0.0, 4.0];
    let reversed_rows = Layout::try_new([1, 2, 2], [4, -2, 1], 2).expect("valid test layout");
    let query = ArrayView::try_new(reversed_rows, &query_data).expect("valid reversed query");
    let key = ArrayView::try_new(reversed_rows, &key_data).expect("valid reversed key");
    let value = ArrayView::try_new(reversed_rows, &value_data).expect("valid reversed value");
    let mut output_data = [99.0_f64; 4];
    let mut weight_data = [99.0_f64; 4];
    let mut output =
        ArrayViewMut::try_new(reversed_rows, &mut output_data).expect("valid reversed output");
    let mut weights =
        ArrayViewMut::try_new(reversed_rows, &mut weight_data).expect("valid reversed weights");

    scaled_dot_product_attention_into(
        &query,
        &key,
        &value,
        AttentionMask::Unmasked,
        1.0,
        &mut output,
        &mut weights,
    )
    .expect("negative-stride attention contract");

    let dominant = core::f64::consts::E / (core::f64::consts::E + 1.0);
    let residual = 1.0 - dominant;
    let logical_weights: Vec<_> = weights.as_view().iter().copied().collect();
    let logical_output: Vec<_> = output.as_view().iter().copied().collect();
    assert_close(
        &logical_weights,
        &[dominant, residual, residual, dominant],
        64.0 * f64::EPSILON,
    );
    assert_close(
        &logical_output,
        &[
            2.0 * residual,
            4.0 * dominant,
            2.0 * dominant,
            4.0 * residual,
        ],
        64.0 * f64::EPSILON,
    );
}
