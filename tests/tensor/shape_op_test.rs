use nn_rs::backend::cpu::CPUBackend;
use nn_rs::linalg::tensor::Tensor;

#[cfg(test)]
#[test]
fn test_transpose() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 3]);
    let transposed = tensor.transpose();
    assert_eq!(transposed.shape(), [3, 2]);
    let expected_data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    for i in 0..3 {
        for j in 0..2 {
            assert_eq!(transposed.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_zeros() {
    let tensor = Tensor::<CPUBackend, 2>::zeros([3, 4]);
    assert_eq!(tensor.shape(), [3, 4]);
    for i in 0..3 {
        for j in 0..4 {
            assert_eq!(tensor.get([i, j]), 0.0);
        }
    }
}

#[cfg(test)]
#[test]
fn test_ones() {
    let tensor = Tensor::<CPUBackend, 2>::ones([2, 5]);
    assert_eq!(tensor.shape(), [2, 5]);
    for i in 0..2 {
        for j in 0..5 {
            assert_eq!(tensor.get([i, j]), 1.0);
        }
    }
}

#[cfg(test)]
#[test]
fn test_as_slice() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data.clone(), [2, 2]);
    assert_eq!(tensor.as_slice(), data);
}

#[cfg(test)]
#[test]
fn test_as_mut_slice() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    tensor.with_mut_data(|data| data[0] = 5.0);
    assert_eq!(tensor.get([0, 0]), 5.0);
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
fn test_unsqueeze() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data.clone(), [2, 2]);
    let unsqueezed = tensor.unsqueeze(0);

    assert_eq!(unsqueezed.shape(), [1, 2, 2]);
    assert_eq!(unsqueezed.as_slice(), data);
}

#[cfg(test)]
#[test]
fn test_squeeze() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 3>::new(data.clone(), [1, 2, 2]);
    let squeezed = tensor.squeeze(0);
    assert_eq!(squeezed.shape(), [2, 2]);
    assert_eq!(squeezed.as_slice(), data);
}
