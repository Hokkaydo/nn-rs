use crate::backend::backend::ActivationOps;
use crate::backend::cpu::CPUBackend;
use crate::linalg::tensor::Tensor;

impl ActivationOps<Self> for CPUBackend {
    fn sigmoid<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let result_data: Vec<f32> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Tensor::new(result_data, tensor.shape())
    }

    fn softmax<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let shape = tensor.shape();
        let cols = shape[NDIM - 1];
        let rows = data.len() / cols;

        let mut result_data = vec![0.0f32; data.len()];
        for r in 0..rows {
            let row = &data[r * cols..(r + 1) * cols];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_row: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_row.iter().sum();
            for c in 0..cols {
                result_data[r * cols + c] = exp_row[c] / sum_exp;
            }
        }

        Tensor::new(result_data, shape)
    }

    fn log_softmax<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let shape = tensor.shape();
        let cols = shape[NDIM - 1];
        let rows = data.len() / cols;

        let mut result_data = vec![0.0f32; data.len()];
        for r in 0..rows {
            let row = &data[r * cols..(r + 1) * cols];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_row: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_row.iter().sum();
            for c in 0..cols {
                result_data[r * cols + c] = (exp_row[c] / sum_exp).ln();
            }
        }

        Tensor::new(result_data, shape)
    }

    fn relu<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let result_data: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();
        Tensor::new(result_data, tensor.shape())
    }
}
