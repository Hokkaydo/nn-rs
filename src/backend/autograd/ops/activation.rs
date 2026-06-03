use crate::backend::autograd::{Autograd, GradNode, GradOp};
use crate::backend::backend::{ActivationOps, Backend};
use crate::linalg::tensor::Tensor;

impl<B: Backend> ActivationOps<Self> for Autograd<B> {
    fn sigmoid<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::sigmoid(&tensor.into());
        Self::record_op(GradNode {
            grad_op: GradOp::Sigmoid,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn softmax<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::softmax(&tensor.into());
        Self::record_op(GradNode {
            grad_op: GradOp::Softmax,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn log_softmax<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::log_softmax(&tensor.into());
        Self::record_op(GradNode {
            grad_op: GradOp::LogSoftmax,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn relu<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::relu(&tensor.into());
        Self::record_op(GradNode {
            grad_op: GradOp::ReLU,
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }
}
