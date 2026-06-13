use nn_core::linalg::tensor::Tensor;
use nn_core::cpu::CPUBackend;

#[cfg(test)]
#[test]
fn new() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data.clone(), [2, 2]);

    assert_eq!(tensor.shape(), [2, 2]);
    assert_eq!(tensor.as_slice(), data);
}

#[cfg(test)]
#[test]
#[should_panic]
fn test_new_invalid_shape() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let _tensor = Tensor::<CPUBackend, 2>::new(data, [3, 2]);
}

#[cfg(test)]
#[test]
fn test_reshape() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data.clone(), [2, 2]);
    let reshaped = tensor.reshape([4, 1]);

    assert_eq!(reshaped.shape(), [4, 1]);
    assert_eq!(reshaped.as_slice(), data);
}

#[cfg(test)]
#[test]
#[should_panic]
fn test_reshape_invalid() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let _reshaped = tensor.reshape([3, 2]);
}

#[cfg(test)]
#[test]
fn test_new_set_get() {
    let data = vec![1.0, 2.0, 3.0, 4.0];

    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);

    assert_eq!(tensor.get([0, 0]), 1.0);
    tensor.with_mut_data(|data| data[0] = 5.0);
    assert_eq!(tensor.get([0, 0]), 5.0);

    let tensor_data = tensor.as_slice();
    assert_eq!(tensor_data[1], 2.0);
    tensor.set([0, 1], 3.0);
    let tensor_data = tensor.as_slice();
    assert_eq!(tensor_data[1], 3.0);
}

#[cfg(test)]
#[test]
fn test_clone() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data.clone(), [2, 2]);
    let tensor_clone = tensor.clone();

    assert_eq!(tensor.shape(), tensor_clone.shape());
    assert_eq!(tensor.as_slice(), tensor_clone.as_slice());
}

#[cfg(test)]
#[test]
fn test_increment_indices() {
    let mut indices = [0, 0];
    let shape = vec![2, 2];
    Tensor::<CPUBackend, 2>::increment_indices(&mut indices, &shape);
    assert_eq!(indices, [0, 1]);

    let mut indices = [0, 1];
    let shape = vec![2, 2];
    Tensor::<CPUBackend, 2>::increment_indices(&mut indices, &shape);
    assert_eq!(indices, [1, 0]);
}

#[cfg(test)]
#[test]
fn test_reduce_shape() {
    let reduced_tensor = Tensor::<CPUBackend, 2>::reduce_shape(&[1, 2, 1, 3]);
    assert_eq!(reduced_tensor, vec![2, 3]);
}

#[cfg(test)]
#[test]
fn test_is_scalar() {
    let data = vec![42.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [1, 1]);
    assert!(tensor.is_scalar());

    let data = vec![1.0, 2.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 1]);
    assert!(!tensor.is_scalar());
}
