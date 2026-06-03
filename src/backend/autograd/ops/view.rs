use crate::backend::autograd::{Autograd, GradNode, GradOp};
use crate::backend::backend::{Backend, ViewOps};
use crate::linalg::tensor::Tensor;

impl<B: Backend> ViewOps<Self> for Autograd<B> {
    fn slice<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
        start: usize,
        len: usize,
    ) -> Tensor<Self, NDIM> {
        let result = B::slice(&tensor.into(), axis, start, len);
        Self::record_op(GradNode {
            grad_op: GradOp::Slice {
                axis,
                start,
                len,
            },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn gather<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
        indices: &[usize],
    ) -> Tensor<Self, NDIM> {
        let result = B::gather(&tensor.into(), axis, indices);
        Self::record_op(GradNode { 
            grad_op: GradOp::Gather { axis, indices: indices.to_vec() },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn scatter_add<const NDMIM: usize>(
            target: &Tensor<Self, NDMIM>,
            src: &Tensor<Self, NDMIM>,
            axis: usize,
            indices: &[usize],
        ) -> Tensor<Self, NDMIM> {
        let result = B::scatter_add(&target.into(), &src.into(), axis, indices);
        Self::record_op(GradNode {
            grad_op: GradOp::ScatterAdd { axis, indices: indices.to_vec() },
            input_ids: vec![target.id, src.id],
            inputs_ndims: vec![NDMIM, NDMIM],
            output_id: result.id,
            output_ndim: NDMIM,
        });
        result.into()
    }
}
