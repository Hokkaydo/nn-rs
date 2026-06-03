use crate::backend::autograd::{Autograd, GradNode, GradOp};
use crate::backend::backend::{Backend, ReductionOps};
use crate::linalg::tensor::Tensor;

impl<B: Backend> ReductionOps<Self> for Autograd<B> {
    fn sum<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<&[usize]>,
    ) -> Tensor<Self, NDIM> {
        let result = B::sum(&tensor.into(), axes);
        Self::record_op(GradNode {
            grad_op: GradOp::Sum { axes: axes.map(|a| a.to_vec()) },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn mean<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<&[usize]>,
    ) -> Tensor<Self, NDIM> {
        let result = B::mean(&tensor.into(), axes);
        Self::record_op(GradNode {
            grad_op: GradOp::Mean { axes: axes.map(|a| a.to_vec()) },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn max<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<&[usize]>,
    ) -> Tensor<Self, NDIM> {
        let result = B::max(&tensor.into(), axes);
        Self::record_op(GradNode {
            grad_op: GradOp::Max { axes: axes.map(|a| a.to_vec()) },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn min<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<&[usize]>,
    ) -> Tensor<Self, NDIM> {
        let result = B::min(&tensor.into(), axes);
        Self::record_op(GradNode {
            grad_op: GradOp::Min { axes: axes.map(|a| a.to_vec()) },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn argmax<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
    ) -> Tensor<Self, { NDIM - 1 }> {
        B::argmax(&tensor.into(), axis).into()
    }
}
