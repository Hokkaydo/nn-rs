use crate::autograd::{as_inner, wrap, Autograd, GradNode, GradOp};
use nn_la::backend::{Backend, ViewOps};
use nn_la::linalg::tensor::Tensor;

impl<B: Backend> ViewOps<Self> for Autograd<B> {
    fn slice<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
        start: usize,
        len: usize,
    ) -> Tensor<Self, NDIM> {
        let result = B::slice(&as_inner(tensor), axis, start, len);
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
        wrap(result)
    }

    fn gather<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
        indices: &[usize],
    ) -> Tensor<Self, NDIM> {
        let result = B::gather(&as_inner(tensor), axis, indices);
        Self::record_op(GradNode { 
            grad_op: GradOp::Gather { axis, indices: indices.to_vec() },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn scatter_add<const NDMIM: usize>(
            target: &Tensor<Self, NDMIM>,
            src: &Tensor<Self, NDMIM>,
            axis: usize,
            indices: &[usize],
        ) -> Tensor<Self, NDMIM> {
        let result = B::scatter_add(&as_inner(target), &as_inner(src), axis, indices);
        Self::record_op(GradNode {
            grad_op: GradOp::ScatterAdd { axis, indices: indices.to_vec() },
            input_ids: vec![target.id, src.id],
            inputs_ndims: vec![NDMIM, NDMIM],
            output_id: result.id,
            output_ndim: NDMIM,
        });
        wrap(result)
    }
}
