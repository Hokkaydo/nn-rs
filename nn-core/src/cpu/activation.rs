use crate::backend::ActivationOps;
use crate::cpu::CPUBackend;
use crate::linalg::tensor::Tensor;

impl ActivationOps<Self> for CPUBackend {
    fn sigmoid<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let shape = tensor.shape();
        let mut result = Vec::with_capacity(shape.iter().product());
        tensor.with_data(|d| result.extend(d.iter().map(|&x| 1.0 / (1.0 + (-x).exp()))));
        Tensor::new(result, shape)
    }

    fn softmax<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let shape = tensor.shape();
        let cols = shape[NDIM - 1];
        let mut result_data = Vec::with_capacity(shape.iter().product());
        tensor.with_data(|data| {
            let rows = data.len() / cols;
            result_data.resize(data.len(), 0.0f32);
            for r in 0..rows {
                let row = &data[r * cols..(r + 1) * cols];
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_row: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
                let sum_exp: f32 = exp_row.iter().sum();
                for c in 0..cols {
                    result_data[r * cols + c] = exp_row[c] / sum_exp;
                }
            }
        });
        Tensor::new(result_data, shape)
    }

    fn log_softmax<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let shape = tensor.shape();
        let cols = shape[NDIM - 1];
        let mut result_data = Vec::with_capacity(shape.iter().product());
        tensor.with_data(|data| {
            let rows = data.len() / cols;
            result_data.resize(data.len(), 0.0f32);
            for r in 0..rows {
                let row = &data[r * cols..(r + 1) * cols];
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_row: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
                let sum_exp: f32 = exp_row.iter().sum();
                for c in 0..cols {
                    result_data[r * cols + c] = (exp_row[c] / sum_exp).ln();
                }
            }
        });
        Tensor::new(result_data, shape)
    }

    fn relu<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let shape = tensor.shape();
        let mut result = Vec::with_capacity(shape.iter().product());
        tensor.with_data(|d| result.extend(d.iter().map(|&x| x.max(0.0))));
        Tensor::new(result, shape)
    }

    fn gelu<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let shape = tensor.shape();
        let mut result = Vec::with_capacity(shape.iter().product());
        tensor.with_data(|d| {
            result.extend(d.iter().map(|&x| 0.5 * x * (1.0 + (x / 2.0f32.sqrt()).tanh())))
        });
        Tensor::new(result, shape)
    }
}
