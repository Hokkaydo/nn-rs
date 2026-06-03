use crate::backend::backend::UnaryOps;
use crate::backend::cpu::CPUBackend;
use crate::linalg::tensor::{Scalar, Tensor};

impl UnaryOps<Self> for CPUBackend {
    fn neg<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let result_data: Vec<Scalar> = data.iter().map(|&x| -x).collect();
        Tensor::new(result_data, tensor.shape())
    }

    fn sqrt<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        Self::pow(tensor, 0.5)
    }

    fn exp<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let result_data: Vec<Scalar> = data.iter().map(|&x| x.exp()).collect();
        Tensor::new(result_data, tensor.shape())
    }

    fn log<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let result_data: Vec<Scalar> = data.iter().map(|&x| x.ln()).collect();
        Tensor::new(result_data, tensor.shape())
    }

    fn abs<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let result_data: Vec<Scalar> = data.iter().map(|&x| x.abs()).collect();
        Tensor::new(result_data, tensor.shape())
    }

    fn sign<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let result_data: Vec<Scalar> = data
            .iter()
            .map(|&x| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect();
        Tensor::new(result_data, tensor.shape())
    }

    fn clamp<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        min: Scalar,
        max: Scalar,
    ) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let result_data: Vec<Scalar> = data.iter().map(|&x| x.max(min).min(max)).collect();
        Tensor::new(result_data, tensor.shape())
    }

    fn pow<const NDIM: usize>(tensor: &Tensor<Self, NDIM>, exponent: Scalar) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let result_data: Vec<Scalar> = data.iter().map(|&x| x.powf(exponent)).collect();
        Tensor::new(result_data, tensor.shape())
    }
}
