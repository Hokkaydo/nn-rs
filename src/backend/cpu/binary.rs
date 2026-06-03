use crate::backend::backend::{BinaryOps, ReverseScalarOps, ScalarOps, UnaryOps};
use crate::backend::cpu::CPUBackend;
use crate::helpers::{broadcast_dimensions, compute_strides};
use crate::linalg::tensor::{Scalar, Tensor};

impl BinaryOps<Self> for CPUBackend {
    fn add<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        other: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let data_a = tensor.as_slice();
        let data_b = other.as_slice();
        let shape_a = tensor.shape();
        let shape_b = other.shape();
        let stride_a = tensor.strides();
        let stride_b = other.strides();

        let result_shape = broadcast_dimensions(&shape_a, &shape_b);
        let result_strides = compute_strides(&result_shape);

        let mut result_data = vec![0.0; result_shape.iter().product()];
        for idx in 0..result_data.len() {
            let mut idx_a = 0;
            let mut idx_b = 0;
            let mut tmp = idx;
            for i in 0..NDIM {
                let dim_idx = tmp / result_strides[i];
                tmp %= result_strides[i];
                let a_dim = if shape_a[i] == 1 { 0 } else { dim_idx };
                let b_dim = if shape_b[i] == 1 { 0 } else { dim_idx };
                idx_a += a_dim * stride_a[i];
                idx_b += b_dim * stride_b[i];
            }
            result_data[idx] = data_a[idx_a] + data_b[idx_b];
        }
        Tensor::new(result_data, result_shape)
    }

    fn sub<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        other: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        Self::add(tensor, &Self::neg(other))
    }

    fn mul<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        other: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let data_a = tensor.as_slice();
        let data_b = other.as_slice();
        let shape_a = tensor.shape();
        let shape_b = other.shape();
        let stride_a = tensor.strides();
        let stride_b = other.strides();

        let result_shape = broadcast_dimensions(&shape_a, &shape_b);
        let result_strides = compute_strides(&result_shape);

        let mut result_data = vec![0.0; result_shape.iter().product()];

        for idx in 0..result_data.len() {
            let mut idx_a = 0;
            let mut idx_b = 0;
            let mut tmp = idx;
            for i in 0..NDIM {
                let dim_idx = tmp / result_strides[i];
                tmp %= result_strides[i];
                let a_dim = if shape_a[i] == 1 { 0 } else { dim_idx };
                let b_dim = if shape_b[i] == 1 { 0 } else { dim_idx };
                idx_a += a_dim * stride_a[i];
                idx_b += b_dim * stride_b[i];
            }
            result_data[idx] = data_a[idx_a] * data_b[idx_b];
        }
        Tensor::new(result_data, result_shape)
    }

    fn div<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        other: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let data_a = tensor.as_slice();
        let data_b = other.as_slice();
        let shape_a = tensor.shape();
        let shape_b = other.shape();
        let stride_a = tensor.strides();
        let stride_b = other.strides();

        let result_shape = broadcast_dimensions(&shape_a, &shape_b);
        let result_strides = compute_strides(&result_shape);

        let mut result_data = vec![0.0; result_shape.iter().product()];
        for idx in 0..result_data.len() {
            let mut idx_a = 0;
            let mut idx_b = 0;
            let mut tmp = idx;
            for i in 0..NDIM {
                let dim_idx = tmp / result_strides[i];
                tmp %= result_strides[i];
                let a_dim = if shape_a[i] == 1 { 0 } else { dim_idx };
                let b_dim = if shape_b[i] == 1 { 0 } else { dim_idx };
                idx_a += a_dim * stride_a[i];
                idx_b += b_dim * stride_b[i];
            }
            result_data[idx] = data_a[idx_a] / data_b[idx_b];
        }
        Tensor::new(result_data, result_shape)
    }
}

impl ScalarOps<Self> for CPUBackend {
    fn add_scalar<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        scalar: Scalar,
    ) -> Tensor<Self, NDIM> {
        let result_data = tensor.as_slice().iter().map(|&x| x + scalar).collect();
        Tensor::new(result_data, tensor.shape())
    }

    fn sub_scalar<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        scalar: Scalar,
    ) -> Tensor<Self, NDIM> {
        let result_data = tensor.as_slice().iter().map(|&x| x - scalar).collect();
        Tensor::new(result_data, tensor.shape())
    }

    fn mul_scalar<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        scalar: Scalar,
    ) -> Tensor<Self, NDIM> {
        let result_data = tensor.as_slice().iter().map(|&x| x * scalar).collect();
        Tensor::new(result_data, tensor.shape())
    }

    fn div_scalar<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        scalar: Scalar,
    ) -> Tensor<Self, NDIM> {
        let result_data = tensor.as_slice().iter().map(|&x| x / scalar).collect();
        Tensor::new(result_data, tensor.shape())
    }
}

impl ReverseScalarOps<Self> for CPUBackend {
    fn scalar_add<const NDIM: usize>(
        scalar: Scalar,
        tensor: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        Self::add_scalar(tensor, scalar)
    }

    fn scalar_sub<const NDIM: usize>(
        scalar: Scalar,
        tensor: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let result_data = tensor.as_slice().iter().map(|&x| scalar - x).collect();
        Tensor::new(result_data, tensor.shape())
    }

    fn scalar_mul<const NDIM: usize>(
        scalar: Scalar,
        tensor: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        Self::mul_scalar(tensor, scalar)
    }

    fn scalar_div<const NDIM: usize>(
        scalar: Scalar,
        tensor: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let result_data = tensor.as_slice().iter().map(|&x| scalar / x).collect();
        Tensor::new(result_data, tensor.shape())
    }
}
