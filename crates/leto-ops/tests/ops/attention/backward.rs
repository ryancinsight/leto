use super::*;

fn loss(query: &[f64], key: &[f64], value: &[f64], output_gradient: &[f64]) -> f64 {
    let query = array([1, 2, 2], query.to_vec());
    let key = array([1, 2, 2], key.to_vec());
    let value = array([1, 2, 2], value.to_vec());
    let mut output = array([1, 2, 2], vec![0.0; 4]);
    let mut weights = array([1, 2, 2], vec![0.0; 4]);
    scaled_dot_product_attention_into(
        &query.view(),
        &key.view(),
        &value.view(),
        AttentionMask::Unmasked,
        0.75,
        &mut output.view_mut(),
        &mut weights.view_mut(),
    )
    .expect("finite-difference forward");
    output
        .storage()
        .as_slice()
        .iter()
        .zip(output_gradient)
        .map(|(output, gradient)| output * gradient)
        .sum()
}

fn finite_difference(values: &[f64], index: usize, evaluate: impl Fn(&[f64]) -> f64) -> f64 {
    let step = f64::EPSILON.cbrt();
    let mut positive = values.to_vec();
    let mut negative = values.to_vec();
    positive[index] += step;
    negative[index] -= step;
    (evaluate(&positive) - evaluate(&negative)) / (2.0 * step)
}

#[test]
fn backward_accumulates_prefilled_targets_and_matches_finite_differences() {
    let query_values = [0.2, -0.3, 0.7, 0.5];
    let key_values = [0.4, -0.1, -0.2, 0.8];
    let value_values = [1.2, -0.7, 0.3, 0.9];
    let output_gradient_values = [0.5, -0.4, 0.2, 0.7];
    let query = array([1, 2, 2], query_values.to_vec());
    let key = array([1, 2, 2], key_values.to_vec());
    let value = array([1, 2, 2], value_values.to_vec());
    let output_gradient = array([1, 2, 2], output_gradient_values.to_vec());
    let mut output = array([1, 2, 2], vec![0.0; 4]);
    let mut weights = array([1, 2, 2], vec![0.0; 4]);
    scaled_dot_product_attention_into(
        &query.view(),
        &key.view(),
        &value.view(),
        AttentionMask::Unmasked,
        0.75,
        &mut output.view_mut(),
        &mut weights.view_mut(),
    )
    .expect("backward fixture forward");

    let mut query_gradient = array([1, 2, 2], vec![1.0; 4]);
    let mut key_gradient = array([1, 2, 2], vec![1.0; 4]);
    let mut value_gradient = array([1, 2, 2], vec![1.0; 4]);
    scaled_dot_product_attention_backward_accumulate(
        &output_gradient.view(),
        &query.view(),
        &key.view(),
        &value.view(),
        &weights.view(),
        0.75,
        AttentionGradients::new(
            Some(query_gradient.view_mut()),
            Some(key_gradient.view_mut()),
            Some(value_gradient.view_mut()),
        ),
    )
    .expect("valid backward contract");

    let tolerance = 64.0 * f64::EPSILON.cbrt();
    for (index, actual) in query_gradient.storage().as_slice().iter().enumerate() {
        let expected = finite_difference(&query_values, index, |query| {
            loss(query, &key_values, &value_values, &output_gradient_values)
        });
        assert!(((actual - 1.0) - expected).abs() <= tolerance);
    }
    for (index, actual) in key_gradient.storage().as_slice().iter().enumerate() {
        let expected = finite_difference(&key_values, index, |key| {
            loss(&query_values, key, &value_values, &output_gradient_values)
        });
        assert!(((actual - 1.0) - expected).abs() <= tolerance);
    }
    for (index, actual) in value_gradient.storage().as_slice().iter().enumerate() {
        let expected = finite_difference(&value_values, index, |value| {
            loss(&query_values, &key_values, value, &output_gradient_values)
        });
        assert!(((actual - 1.0) - expected).abs() <= tolerance);
    }

    let expected_query_increment: Vec<_> = query_gradient
        .storage()
        .as_slice()
        .iter()
        .map(|gradient| gradient - 1.0)
        .collect();
    let mut query_only_gradient = array([1, 2, 2], vec![3.0; 4]);
    scaled_dot_product_attention_backward_accumulate(
        &output_gradient.view(),
        &query.view(),
        &key.view(),
        &value.view(),
        &weights.view(),
        0.75,
        AttentionGradients::new(Some(query_only_gradient.view_mut()), None, None),
    )
    .expect("a single requested gradient is valid");
    for (actual, expected_increment) in query_only_gradient
        .storage()
        .as_slice()
        .iter()
        .zip(expected_query_increment)
    {
        assert!(((actual - 3.0) - expected_increment).abs() <= tolerance);
    }
}
