use crate::autograd::{as_inner, wrap, Autograd, GradNode, GradOp};
use nn_core::backend::{Backend, ShapeOps};
use nn_core::linalg::tensor::Tensor;

impl<B: Backend> ShapeOps<Self> for Autograd<B> {
    fn reshape<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        new_shape: [usize; NDIM],
    ) -> Tensor<Self, NDIM> {
        let result = B::reshape(&as_inner(tensor), new_shape);
        Self::record_op(GradNode {
            grad_op: GradOp::Reshape { new_shape: new_shape.to_vec() },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn transpose<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<[usize; NDIM]>,
    ) -> Tensor<Self, NDIM> {
        let result = B::transpose(&as_inner(tensor), axes);
        Self::record_op(GradNode {
            grad_op: GradOp::Transpose { axes: axes.map(|a| a.to_vec()) },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn squeeze<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
    ) -> Tensor<Self, { NDIM - 1 }> {
        let result = B::squeeze(&as_inner(tensor), axis);
        Self::record_op(GradNode {
            grad_op: GradOp::Squeeze { new_shape: B::shape(&result).to_vec(), axis},
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM - 1,
        });
        wrap(result)
    }

    fn unsqueeze<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
    ) -> Tensor<Self, { NDIM + 1 }> {
        let result = B::unsqueeze(&as_inner(tensor), axis);
        Self::record_op(GradNode {
            grad_op: GradOp::Unsqueeze { axis },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM + 1,
        });
        wrap(result)
    }

    fn broadcast<const OLD_NDIM: usize, const NEW_NDIM: usize>(
        tensor: &Tensor<Self, OLD_NDIM>,
        new_shape: [usize; NEW_NDIM],
    ) -> Tensor<Self, NEW_NDIM> {
        let result = B::broadcast(&as_inner(tensor), new_shape);
        Self::record_op(GradNode {
            grad_op: GradOp::Broadcast { new_shape: new_shape.to_vec() },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![OLD_NDIM],
            output_id: result.id,
            output_ndim: NEW_NDIM,
        });
        wrap(result)
    }
}
