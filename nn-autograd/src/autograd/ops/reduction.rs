use crate::autograd::{as_inner, wrap, Autograd, GradNode, GradOp};
use nn_core::backend::{Backend, ReductionOps};
use nn_core::linalg::tensor::Tensor;

impl<B: Backend> ReductionOps<Self> for Autograd<B> {
    fn sum<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<&[usize]>,
    ) -> Tensor<Self, NDIM> {
        let result = B::sum(&as_inner(tensor), axes);
        Self::record_op(GradNode {
            grad_op: GradOp::Sum { axes: axes.map(|a| a.to_vec()) },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn mean<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<&[usize]>,
    ) -> Tensor<Self, NDIM> {
        let result = B::mean(&as_inner(tensor), axes);
        Self::record_op(GradNode {
            grad_op: GradOp::Mean { axes: axes.map(|a| a.to_vec()) },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn max<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<&[usize]>,
    ) -> Tensor<Self, NDIM> {
        let result = B::max(&as_inner(tensor), axes);
        Self::record_op(GradNode {
            grad_op: GradOp::Max { axes: axes.map(|a| a.to_vec()) },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn min<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<&[usize]>,
    ) -> Tensor<Self, NDIM> {
        let result = B::min(&as_inner(tensor), axes);
        Self::record_op(GradNode {
            grad_op: GradOp::Min { axes: axes.map(|a| a.to_vec()) },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn argmax<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
    ) -> Tensor<Self, { NDIM - 1 }> {
        wrap(B::argmax(&as_inner(tensor), axis))
    }
}
