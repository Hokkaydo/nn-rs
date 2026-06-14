use nn_la::linalg::tensor::Tensor;
use nn_la::cpu::CPUBackend;

#[cfg(test)]
#[test]
fn test_sigmoid() {

    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let tensor = Tensor::<CPUBackend, 1>::new(data, [5]);
    let result = tensor.sigmoid();
    let expected_data = vec![
        1.0 / (1.0 + 2.0f32.exp()),
        1.0 / (1.0 + 1.0f32.exp()),
        0.5,
        1.0 / (1.0 + (-1.0f32).exp()),
        1.0 / (1.0 + (-2.0f32).exp()),
    ];
    for i in 0..5 {
        assert!((result.get([i]) - expected_data[i]).abs() < 1e-6);
    }
}

#[cfg(test)]
#[test]
fn test_log_softmax() {
    let data = vec![1.0, 2.0, 3.0];
    let tensor = Tensor::<CPUBackend, 1>::new(data.clone(), [3]);
    let result = tensor.log_softmax();
    let sum_exp: f32 = data.iter().map(|&x| x.exp()).sum();
    let expected_data: Vec<f32> = data.iter().map(|&x| (x.exp() / sum_exp).ln()).collect();
    for i in 0..3 {
        assert!((result.get([i]) - expected_data[i]).abs() < 1e-6);
    }
}

#[cfg(test)]
#[test]
fn test_softmax() {
    let data = vec![1.0, 2.0, 3.0];
    let tensor = Tensor::<CPUBackend, 1>::new(data.clone(), [3]);
    let result = tensor.softmax();
    let sum_exp: f32 = data.iter().map(|&x| x.exp()).sum();
    let expected_data: Vec<f32> = data.iter().map(|&x| x.exp() / sum_exp).collect();
    assert_eq!(
        result.shape().iter().product::<usize>(),
        expected_data.len()
    );
    for i in 0..3 {
        assert!((result.get([i]) - expected_data[i]).abs() < 1e-6);
    }
}

#[cfg(test)]
#[test]
fn test_relu() {
    let data = vec![-1.0, 0.0, 2.0, -3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 1>::new(data, [5]);
    let result = tensor.relu();
    let expected_data = vec![0.0, 0.0, 2.0, 0.0, 4.0];
    for i in 0..5 {
        assert_eq!(result.get([i]), expected_data[i]);
    }
}

#[cfg(test)]
#[test]
fn test_gelu_1d() {
    let data = vec![-1.0, 0.0, 1.0];
    let tensor = Tensor::<CPUBackend, 1>::new(data.clone(), [3]);
    let result = tensor.gelu();
    assert_eq!(result.shape(), [3]);
    for (i, &x) in data.iter().enumerate() {
        let expected = 0.5 * x * (1.0 + (x / 2.0f32.sqrt()).tanh());
        assert!((result.get([i]) - expected).abs() < 1e-6, "gelu({x}) mismatch");
    }
}

#[cfg(test)]
#[test]
fn test_gelu_2d() {
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data.clone(), [2, 3]);
    let result = tensor.gelu();
    assert_eq!(result.shape(), [2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            let x = data[i * 3 + j];
            let expected = 0.5 * x * (1.0 + (x / 2.0f32.sqrt()).tanh());
            assert!((result.get([i, j]) - expected).abs() < 1e-6);
        }
    }
}

#[cfg(test)]
#[test]
fn test_gelu_zero() {
    // gelu(0) == 0
    let tensor = Tensor::<CPUBackend, 1>::new(vec![0.0], [1]);
    assert!((tensor.gelu().as_scalar()).abs() < 1e-7);
}

#[cfg(test)]
#[test]
fn test_gelu_positive_approx_identity() {
    // For large positive x, gelu(x) ≈ x
    let x = 10.0f32;
    let tensor = Tensor::<CPUBackend, 1>::new(vec![x], [1]);
    let result = tensor.gelu().as_scalar();
    assert!((result - x).abs() < 1e-4, "gelu({x}) should be ~{x}, got {result}");
}
