use nn_rs::backend::cpu::CPUBackend;
use nn_rs::linalg::tensor::Tensor;

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
