use crate::backend::UnaryOps;
use crate::cpu::CPUBackend;
use crate::linalg::tensor::{Scalar, Tensor};

#[inline(always)]
fn unary_op<const NDIM: usize, F: Fn(Scalar) -> Scalar>(
    tensor: &Tensor<CPUBackend, NDIM>,
    f: F,
) -> Tensor<CPUBackend, NDIM> {
    let shape = tensor.shape();
    let n: usize = shape.iter().product();
    let mut result = Vec::with_capacity(n);
    tensor.with_data(|d| result.extend(d.iter().map(|&x| f(x))));
    Tensor::new(result, shape)
}

impl UnaryOps<Self> for CPUBackend {
    fn neg<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        unary_op(tensor, |x| -x)
    }

    fn sqrt<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        Self::pow(tensor, 0.5)
    }

    fn exp<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        unary_op(tensor, |x| x.exp())
    }

    fn log<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        unary_op(tensor, |x| x.ln())
    }

    fn abs<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        unary_op(tensor, |x| x.abs())
    }

    fn sign<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        unary_op(tensor, |x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 })
    }

    fn clamp<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        min: Scalar,
        max: Scalar,
    ) -> Tensor<Self, NDIM> {
        unary_op(tensor, |x| x.max(min).min(max))
    }

    fn pow<const NDIM: usize>(tensor: &Tensor<Self, NDIM>, exponent: Scalar) -> Tensor<Self, NDIM> {
        unary_op(tensor, |x| x.powf(exponent))
    }

    fn tanh<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        unary_op(tensor, |x| x.tanh())
    }
}
