use crate::autograd::{as_inner, wrap, Autograd, GradNode, GradOp};
use nn_core::backend::{Backend, UnaryOps};
use nn_core::linalg::tensor::{Scalar, Tensor};

impl<B: Backend> UnaryOps<Self> for Autograd<B> {
    fn neg<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::neg(&as_inner(tensor));
        Self::record_op(GradNode {
            grad_op: GradOp::Neg,  
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn sqrt<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::sqrt(&as_inner(tensor));
        Self::record_op(GradNode {
            grad_op: GradOp::Pow { exponent: 0.5 },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn exp<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::exp(&as_inner(tensor));
        Self::record_op(GradNode {
            grad_op: GradOp::Exp,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn log<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::log(&as_inner(tensor));
        Self::record_op(GradNode {
            grad_op: GradOp::Log,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn abs<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::abs(&as_inner(tensor));
        Self::record_op(GradNode {
            grad_op: GradOp::Abs,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn sign<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::sign(&as_inner(tensor));
        Self::record_op(GradNode {
            grad_op: GradOp::Sign,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn clamp<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        min: Scalar,
        max: Scalar,
    ) -> Tensor<Self, NDIM> {
        let result = B::clamp(&as_inner(tensor), min, max);
        Self::record_op(GradNode {
            grad_op: GradOp::Clamp { min, max },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn pow<const NDIM: usize>(tensor: &Tensor<Self, NDIM>, exponent: Scalar) -> Tensor<Self, NDIM> {
        let result = B::pow(&as_inner(tensor), exponent);
        Self::record_op(GradNode {
            grad_op: GradOp::Pow { exponent },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn tanh<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::tanh(&as_inner(tensor));
        Self::record_op(GradNode {
            grad_op: GradOp::Tanh,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }
}
