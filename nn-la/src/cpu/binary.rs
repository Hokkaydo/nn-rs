use crate::backend::{BinaryOps, ReverseScalarOps, ScalarOps};
use crate::cpu::CPUBackend;
use crate::helpers::{broadcast_dimensions, compute_strides};
use crate::linalg::tensor::{Scalar, Tensor};

/// Returns true when the tensor occupies a contiguous row-major buffer with no
/// offset (strides match the default row-major strides and offset == 0).
#[inline(always)]
fn is_contiguous<const NDIM: usize>(shape: &[usize; NDIM], strides: &[usize; NDIM], offset: usize) -> bool {
    if offset != 0 {
        return false;
    }
    let mut expected = 1usize;
    for i in (0..NDIM).rev() {
        if strides[i] != expected {
            return false;
        }
        expected *= shape[i];
    }
    true
}

/// Apply a pointwise binary op element-wise over two tensors.
/// Fast path: when both tensors are contiguous with the same shape, iterate
/// directly over the raw buffer slices with no index arithmetic.
/// Slow path: full broadcast with per-element stride calculations.
#[inline(always)]
fn binary_op<const NDIM: usize, F: Fn(Scalar, Scalar) -> Scalar>(
    a: &Tensor<CPUBackend, NDIM>,
    b: &Tensor<CPUBackend, NDIM>,
    f: F,
) -> Tensor<CPUBackend, NDIM> {
    let shape_a = a.shape();
    let shape_b = b.shape();
    let stride_a = a.strides();
    let stride_b = b.strides();
    let off_a = a.offset();
    let off_b = b.offset();

    if shape_a == shape_b
        && is_contiguous(&shape_a, &stride_a, off_a)
        && is_contiguous(&shape_b, &stride_b, off_b)
    {
        // Fast path: direct zip over raw buffers, no index arithmetic.
        let mut result = Vec::with_capacity(shape_a.iter().product());
        a.with_data(|da| {
            b.with_data(|db| {
                result.extend(da.iter().zip(db.iter()).map(|(&x, &y)| f(x, y)));
            });
        });
        return Tensor::new(result, shape_a);
    }

    // Slow path: generic broadcast.
    let result_shape = broadcast_dimensions(&shape_a, &shape_b);
    let result_strides = compute_strides(&result_shape);
    let n: usize = result_shape.iter().product();
    let mut result_data = vec![0.0; n];
    let data_a = a.as_slice();
    let data_b = b.as_slice();
    for idx in 0..n {
        let mut idx_a = off_a;
        let mut idx_b = off_b;
        let mut tmp = idx;
        for i in 0..NDIM {
            let dim_idx = tmp / result_strides[i];
            tmp %= result_strides[i];
            let a_dim = if shape_a[i] == 1 { 0 } else { dim_idx };
            let b_dim = if shape_b[i] == 1 { 0 } else { dim_idx };
            idx_a += a_dim * stride_a[i];
            idx_b += b_dim * stride_b[i];
        }
        result_data[idx] = f(data_a[idx_a], data_b[idx_b]);
    }
    Tensor::new(result_data, result_shape)
}

impl BinaryOps<Self> for CPUBackend {
    fn add<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        other: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        binary_op(tensor, other, |a, b| a + b)
    }

    fn sub<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        other: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        binary_op(tensor, other, |a, b| a - b)
    }

    fn mul<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        other: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        binary_op(tensor, other, |a, b| a * b)
    }

    fn div<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        other: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        binary_op(tensor, other, |a, b| a / b)
    }
}

impl ScalarOps<Self> for CPUBackend {
    fn add_scalar<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        scalar: Scalar,
    ) -> Tensor<Self, NDIM> {
        let mut result = Vec::with_capacity(tensor.shape().iter().product());
        tensor.with_data(|d| result.extend(d.iter().map(|&x| x + scalar)));
        Tensor::new(result, tensor.shape())
    }

    fn sub_scalar<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        scalar: Scalar,
    ) -> Tensor<Self, NDIM> {
        let mut result = Vec::with_capacity(tensor.shape().iter().product());
        tensor.with_data(|d| result.extend(d.iter().map(|&x| x - scalar)));
        Tensor::new(result, tensor.shape())
    }

    fn mul_scalar<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        scalar: Scalar,
    ) -> Tensor<Self, NDIM> {
        let mut result = Vec::with_capacity(tensor.shape().iter().product());
        tensor.with_data(|d| result.extend(d.iter().map(|&x| x * scalar)));
        Tensor::new(result, tensor.shape())
    }

    fn div_scalar<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        scalar: Scalar,
    ) -> Tensor<Self, NDIM> {
        let mut result = Vec::with_capacity(tensor.shape().iter().product());
        tensor.with_data(|d| result.extend(d.iter().map(|&x| x / scalar)));
        Tensor::new(result, tensor.shape())
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
        let mut result = Vec::with_capacity(tensor.shape().iter().product());
        tensor.with_data(|d| result.extend(d.iter().map(|&x| scalar - x)));
        Tensor::new(result, tensor.shape())
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
        let mut result = Vec::with_capacity(tensor.shape().iter().product());
        tensor.with_data(|d| result.extend(d.iter().map(|&x| scalar / x)));
        Tensor::new(result, tensor.shape())
    }
}
