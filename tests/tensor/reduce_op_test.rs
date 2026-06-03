use nn_rs::backend::cpu::CPUBackend;
use nn_rs::linalg::tensor::Tensor;

#[cfg(test)]
#[test]
fn test_mean() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let mean_tensor = tensor.mean(&[1]);
    let expected_data = vec![1.5, 3.5];
    assert_eq!(mean_tensor.shape(), [2, 1]);
    for i in 0..2 {
        assert_eq!(mean_tensor.get([i, 0]), expected_data[i]);
    }
}

#[cfg(test)]
#[test]
fn test_mean_scalar() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let mean = tensor.mean_scalar();
    let expected_value = 2.5;
    assert_eq!(mean.shape(), [1, 1]);
    assert_eq!(mean.get([0, 0]), expected_value);
}

#[cfg(test)]
#[test]
fn test_slice() {
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [4, 3]);
    let sliced_tensor = tensor.slice(0, 1, 2);
    let expected_data = vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    assert_eq!(sliced_tensor.shape(), [2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(sliced_tensor.get([i, j]), expected_data[i * 3 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_gather() {
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [4, 3]);
    let gathered_tensor = tensor.gather(0, &[0, 2]);
    let expected_data = vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0];
    assert_eq!(gathered_tensor.shape(), [2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(gathered_tensor.get([i, j]), expected_data[i * 3 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_max() {
    let data = vec![1.0, 3.0, 2.0, 4.0, 6.0, 5.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 3]);
    let max_tensor = tensor.max(&[0]);
    let expected_data = vec![4.0, 6.0, 5.0];
    assert_eq!(max_tensor.shape(), [1, 3]);
    for i in 0..3 {
        assert_eq!(max_tensor.get([0, i]), expected_data[i]);
    }
}

#[cfg(test)]
#[test]
fn test_min() {
    let data = vec![4.0, 2.0, 3.0, 1.0, 6.0, 5.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 3]);
    let min_tensor = tensor.min(&[1]);
    let expected_data = vec![2.0, 1.0];
    assert_eq!(min_tensor.shape(), [2, 1]);
    for i in 0..2 {
        assert_eq!(min_tensor.get([i, 0]), expected_data[i]);
    }
}

#[cfg(test)]
#[test]
fn test_max_scalar() {
    let data = vec![1.0, 3.0, 2.0, 4.0, 6.0, 5.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 3]);
    let max_tensor = tensor.max_scalar();
    let expected_data = 6.0;
    assert_eq!(max_tensor.shape(), [1, 1]);
    assert_eq!(max_tensor.get([0, 0]), expected_data);
}

#[cfg(test)]
#[test]
fn test_min_scalar() {
    let data = vec![4.0, 2.0, 3.0, 1.0, 6.0, 5.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 3]);
    let min_tensor = tensor.min_scalar();
    let expected_data = 1.0;
    assert_eq!(min_tensor.shape(), [1, 1]);
    assert_eq!(min_tensor.get([0, 0]), expected_data);
}

#[cfg(test)]
#[test]
fn test_sum() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let sum_tensor = tensor.sum();
    let expected_value = 10.0;
    assert_eq!(sum_tensor.get([0, 0]), expected_value);
}

#[cfg(test)]
#[test]
fn test_sum_axis() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let sum_tensor = tensor.sum_axis(&[0]);
    let expected_data = vec![4.0, 6.0];
    assert_eq!(sum_tensor.shape(), [1, 2]);
    for i in 0..2 {
        assert_eq!(sum_tensor.get([0, i]), expected_data[i]);
    }
}
