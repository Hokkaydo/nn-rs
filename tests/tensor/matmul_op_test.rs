use nn_rs::backend::cpu::CPUBackend;
use nn_rs::linalg::tensor::Tensor;

#[cfg(test)]
#[test]
fn test_matmul11() {
    let data_a = vec![4.0];
    let data_b = vec![8.0];
    let tensor_a = Tensor::<CPUBackend, 1>::new(data_a, [1]);
    let tensor_b = Tensor::new(data_b, [1]);
    let result = tensor_a.matmul(&tensor_b);
    let expected_data = vec![32.0];
    assert_eq!(result.shape(), [1]);
    assert_eq!(result.get([0]), expected_data[0]);
}

#[cfg(test)]
#[test]
fn test_matmul12() {
    let data_a = vec![1.0, 2.0];
    let data_b = vec![5.0, 6.0, 7.0, 8.0];
    let tensor_a = Tensor::<CPUBackend, 1>::new(data_a, [2]);
    let tensor_b = Tensor::new(data_b, [2, 2]);
    let result = tensor_a.matmul(&tensor_b);
    let expected_data = vec![19.0, 22.0];
    assert_eq!(result.shape(), [2]);
    for i in 0..2 {
        assert_eq!(result.get([i]), expected_data[i]);
    }
}

#[cfg(test)]
#[test]
fn test_matmul21() {
    let data_a = vec![1.0, 2.0, 3.0, 4.0];
    let data_b = vec![5.0, 6.0];
    let tensor_a = Tensor::<CPUBackend, 2>::new(data_a, [2, 2]);
    let tensor_b = Tensor::new(data_b, [2]);
    let result = tensor_a.matmul(&tensor_b);
    let expected_data = vec![17.0, 39.0];
    assert_eq!(result.shape(), [2]);
    for i in 0..2 {
        assert_eq!(result.get([i]), expected_data[i]);
    }
}

#[cfg(test)]
#[test]
fn test_matmul22() {
    let data_a = vec![1.0, 2.0, 3.0, 4.0];
    let data_b = vec![5.0, 6.0, 7.0, 8.0];
    let tensor_a = Tensor::<CPUBackend, 2>::new(data_a, [2, 2]);
    let tensor_b = Tensor::new(data_b, [2, 2]);
    let result = tensor_a.matmul(&tensor_b);
    let expected_data = vec![19.0, 22.0, 43.0, 50.0];
    assert_eq!(result.shape(), [2, 2]);
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_mul() {
    let data_a = vec![1.0, 2.0, 3.0, 4.0];
    let data_b = vec![5.0, 6.0, 7.0, 8.0];
    let tensor_a = Tensor::<CPUBackend, 2>::new(data_a, [2, 2]);
    let tensor_b = Tensor::new(data_b, [2, 2]);
    let result = tensor_a * tensor_b;
    let expected_data = vec![5.0, 12.0, 21.0, 32.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}
