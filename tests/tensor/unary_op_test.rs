use nn_rs::backend::cpu::CPUBackend;
use nn_rs::linalg::tensor::Tensor;

#[cfg(test)]
#[test]
fn test_neg() {
    let data = vec![1.0, -2.0, 3.0, -4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let result = -tensor;
    let expected_data = vec![-1.0, 2.0, -3.0, 4.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_pow() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let exponent = 3.0;
    let result = tensor.pow(exponent);
    let expected_data = vec![1.0, 8.0, 27.0, 64.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_square() {
    let data = vec![1.0, -2.0, 3.0, -4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let result = tensor.square();
    let expected_data = vec![1.0, 4.0, 9.0, 16.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_abs() {
    let data = vec![-1.0, -2.0, 3.0, -4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let result = tensor.abs();
    let expected_data = vec![1.0, 2.0, 3.0, 4.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_clamp() {
    let data = vec![-1.0, 0.5, 2.0, 3.5];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let min = 0.0;
    let max = 2.0;
    let result = tensor.clamp(min, max);
    let expected_data = vec![0.0, 0.5, 2.0, 2.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_log() {
    let data = vec![
        1.0,
        std::f32::consts::E,
        std::f32::consts::E.powf(2.0),
        std::f32::consts::E.powf(3.0),
    ];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let result = tensor.log();
    let expected_data = vec![0.0, 1.0, 2.0, 3.0];
    for i in 0..2 {
        for j in 0..2 {
            assert!((result.get([i, j]) - expected_data[i * 2 + j]).abs() < 1e-6);
        }
    }
}

#[cfg(test)]
#[test]
fn test_argmax2d() {
    let data = vec![1.0, 3.0, 2.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 3]);
    let argmax_tensor = tensor.argmax(1);
    let expected_data = vec![1.0, 2.0];
    assert_eq!(argmax_tensor.shape(), [2]);
    assert_eq!(argmax_tensor.as_slice(), expected_data);
}

#[cfg(test)]
#[test]
fn test_argmax() {
    let data = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
    let tensor = Tensor::<CPUBackend, 1>::new(data, [6]);
    let result = tensor.argmax(0);
    assert!(result.is_scalar());
    assert_eq!(result.as_scalar(), 5.0);
}

#[cfg(test)]
#[test]
#[should_panic]
fn test_argmax_invalid_axis() {
    let data = vec![1.0, 2.0, 3.0];
    let tensor = Tensor::<CPUBackend, 1>::new(data, [3]);
    let _result = tensor.argmax(2); // Invalid axis for 1D tensor
}

#[cfg(test)]
#[test]
fn test_sqrt() {
    let data = vec![1.0, 4.0, 9.0, 16.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let result = tensor.sqrt();
    let expected_data = vec![1.0, 2.0, 3.0, 4.0];
    for i in 0..2 {
        for j in 0..2 {
            assert!((result.get([i, j]) - expected_data[i * 2 + j]).abs() < 1e-6);
        }
    }
}

#[cfg(test)]
#[test]
fn test_exp() {
    let data = vec![0.0, 1.0, 2.0];
    let tensor = Tensor::<CPUBackend, 1>::new(data, [3]);
    let result = tensor.exp();
    let expected_data = vec![1.0, std::f32::consts::E, std::f32::consts::E.powf(2.0)];
    for i in 0..3 {
        assert!((result.get([i]) - expected_data[i]).abs() < 1e-6);
    }
}

#[cfg(test)]
#[test]
fn test_sign() {
    let data = vec![-3.0, 0.0, 4.0];
    let tensor = Tensor::<CPUBackend, 1>::new(data, [3]);
    let result = tensor.sign();
    let expected_data = vec![-1.0, 0.0, 1.0];
    assert_eq!(result.shape(), [3]);
    for i in 0..3 {
        assert_eq!(result.get([i]), expected_data[i]);
    }
}
