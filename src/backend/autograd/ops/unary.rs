use crate::backend::autograd::{Autograd, GradNode, GradOp};
use crate::backend::backend::{Backend, UnaryOps};
use crate::linalg::tensor::{Scalar, Tensor};

impl<B: Backend> UnaryOps<Self> for Autograd<B> {
    fn neg<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::neg(&tensor.into());
        Self::record_op(GradNode {
            grad_op: GradOp::Neg,  
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn sqrt<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::sqrt(&tensor.into());
        Self::record_op(GradNode {
            grad_op: GradOp::Pow { exponent: 0.5 },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn exp<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::exp(&tensor.into());
        Self::record_op(GradNode {
            grad_op: GradOp::Exp,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn log<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::log(&tensor.into());
        Self::record_op(GradNode {
            grad_op: GradOp::Log,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn abs<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::abs(&tensor.into());
        Self::record_op(GradNode {
            grad_op: GradOp::Abs,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn sign<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::sign(&tensor.into());
        Self::record_op(GradNode {
            grad_op: GradOp::Sign,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn clamp<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        min: Scalar,
        max: Scalar,
    ) -> Tensor<Self, NDIM> {
        let result = B::clamp(&tensor.into(), min, max);
        Self::record_op(GradNode {
            grad_op: GradOp::Clamp { min, max },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn pow<const NDIM: usize>(tensor: &Tensor<Self, NDIM>, exponent: Scalar) -> Tensor<Self, NDIM> {
        let result = B::pow(&tensor.into(), exponent);
        Self::record_op(GradNode {
            grad_op: GradOp::Pow { exponent },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }
}
