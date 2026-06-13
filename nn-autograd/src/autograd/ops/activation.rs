use crate::autograd::{as_inner, wrap, Autograd, GradNode, GradOp};
use nn_core::backend::{ActivationOps, Backend};
use nn_core::linalg::tensor::Tensor;

impl<B: Backend> ActivationOps<Self> for Autograd<B> {
    fn sigmoid<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::sigmoid(&as_inner(tensor));
        Self::record_op(GradNode {
            grad_op: GradOp::Sigmoid,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn softmax<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::softmax(&as_inner(tensor));
        Self::record_op(GradNode {
            grad_op: GradOp::Softmax,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn log_softmax<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::log_softmax(&as_inner(tensor));
        Self::record_op(GradNode {
            grad_op: GradOp::LogSoftmax,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn relu<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::relu(&as_inner(tensor));
        Self::record_op(GradNode {
            grad_op: GradOp::ReLU,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn gelu<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::gelu(&as_inner(tensor));
        Self::record_op(GradNode {
            grad_op: GradOp::GELU,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }
}
