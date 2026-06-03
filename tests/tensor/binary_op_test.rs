use nn_rs::backend::cpu::CPUBackend;
use nn_rs::linalg::tensor::Tensor;

#[cfg(test)]
#[test]
fn test_add() {
    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let data2 = vec![5.0, 6.0, 7.0, 8.0];
    let tensor1 = Tensor::<CPUBackend, 2>::new(data1, [2, 2]);
    let tensor2 = Tensor::new(data2, [2, 2]);
    let result = tensor1 + tensor2;
    let expected_data = vec![6.0, 8.0, 10.0, 12.0];
    println!("Result: {:?}", result.as_slice());
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_sub() {
    let data1 = vec![5.0, 6.0, 7.0, 8.0];
    let data2 = vec![1.0, 2.0, 3.0, 4.0];
    let tensor1 = Tensor::<CPUBackend, 2>::new(data1, [2, 2]);
    let tensor2 = Tensor::new(data2, [2, 2]);
    let result = tensor1 - tensor2;
    let expected_data = vec![4.0, 4.0, 4.0, 4.0];
    println!("Result: {:?}", result.as_slice());
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_scalar_mul() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let scalar = 3.0;
    let result = tensor * scalar;
    let expected_data = vec![3.0, 6.0, 9.0, 12.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_mul_ews() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let result = tensor.clone() * tensor;
    let expected_data = vec![1.0, 4.0, 9.0, 16.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_scalar_sub() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let scalar = 10.0;
    let result = scalar - tensor;
    let expected_data = vec![9.0, 8.0, 7.0, 6.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_sub_scalar() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let scalar = 10.0;
    let result = tensor - scalar;
    let expected_data = vec![-9.0, -8.0, -7.0, -6.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_scalar_div() {
    let data = vec![2.0, 4.0, 8.0, 16.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let scalar = 32.0;
    let result = scalar / tensor;
    let expected_data = vec![16.0, 8.0, 4.0, 2.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_scalar_add() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let scalar = 10.0;
    let result = scalar + tensor;
    let expected_data = vec![11.0, 12.0, 13.0, 14.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_add_scalar() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let scalar = 10.0;
    let result = tensor + scalar;
    let expected_data = vec![11.0, 12.0, 13.0, 14.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}
#[cfg(test)]
#[test]
fn test_div() {
    let data1 = vec![8.0, 16.0, 32.0, 64.0];
    let data2 = vec![2.0, 4.0, 8.0, 16.0];
    let tensor1 = Tensor::<CPUBackend, 2>::new(data1, [2, 2]);
    let tensor2 = Tensor::new(data2, [2, 2]);
    let result = tensor1 / tensor2;
    let expected_data = vec![4.0, 4.0, 4.0, 4.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_div_scalar() {
    let data = vec![2.0, 4.0, 8.0, 16.0];
    let tensor = Tensor::<CPUBackend, 2>::new(data, [2, 2]);
    let scalar = 2.0;
    let result = tensor / scalar;
    let expected_data = vec![1.0, 2.0, 4.0, 8.0];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(result.get([i, j]), expected_data[i * 2 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_broadcast_add() {
    let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data2 = vec![10.0, 20.0, 30.0];
    let tensor1 = Tensor::<CPUBackend, 2>::new(data1, [2, 3]);
    let tensor2 = Tensor::new(data2, [3]);
    let result = tensor1.broadcast_add(&tensor2);
    let expected_data = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
    println!("Result: {:?}", result.as_slice());
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(result.get([i, j]), expected_data[i * 3 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_broadcast_add_full() {
    let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data2 = vec![10.0, 20.0, 30.0];
    let tensor1 = Tensor::<CPUBackend, 2>::new(data1, [2, 3]);
    let tensor2 = Tensor::new(data2, [1, 3]);
    let result = tensor1.broadcast_add(&tensor2);
    let expected_data = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(result.get([i, j]), expected_data[i * 3 + j]);
        }
    }
}

#[cfg(test)]
#[test]
fn test_broadcast_add_3d() {
    let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let data2 = vec![10.0, 20.0];
    let tensor1 = Tensor::<CPUBackend, 3>::new(data1, [2, 2, 2]);
    let tensor2 = Tensor::new(data2, [1, 2, 1]);
    let result = tensor1.broadcast_add(&tensor2);
    let expected_data = vec![11., 12., 23., 24., 15., 16., 27., 28.];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                assert_eq!(result.get([i, j, k]), expected_data[i * 4 + j * 2 + k]);
            }
        }
    }
}
