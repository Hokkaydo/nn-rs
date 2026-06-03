use crate::backend::autograd::{Autograd, GradNode, GradOp};
use crate::backend::backend::{Backend, ShapeOps};
use crate::linalg::tensor::Tensor;

impl<B: Backend> ShapeOps<Self> for Autograd<B> {
    fn reshape<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        new_shape: [usize; NDIM],
    ) -> Tensor<Self, NDIM> {
        let result = B::reshape(&tensor.into(), new_shape);
        Self::record_op(GradNode {
            grad_op: GradOp::Reshape { new_shape: new_shape.to_vec() },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn transpose<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<[usize; NDIM]>,
    ) -> Tensor<Self, NDIM> {
        let result = B::transpose(&tensor.into(), axes);
        Self::record_op(GradNode {
            grad_op: GradOp::Transpose { axes: axes.map(|a| a.to_vec()) },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        result.into()
    }

    fn squeeze<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
    ) -> Tensor<Self, { NDIM - 1 }> {
        let result = B::squeeze(&tensor.into(), axis);
        Self::record_op(GradNode {
            grad_op: GradOp::Squeeze { new_shape: B::shape(&result).to_vec(), axis},
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM - 1,
        });
        result.into()
    }

    fn unsqueeze<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
    ) -> Tensor<Self, { NDIM + 1 }> {
        let result = B::unsqueeze(&tensor.into(), axis);
        Self::record_op(GradNode {
            grad_op: GradOp::Unsqueeze { axis },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM + 1,
        });
        result.into()
    }

    fn broadcast<const OLD_NDIM: usize, const NEW_NDIM: usize>(
        tensor: &Tensor<Self, OLD_NDIM>,
        new_shape: [usize; NEW_NDIM],
    ) -> Tensor<Self, NEW_NDIM> {
        let result = B::broadcast(&tensor.into(), new_shape);
        Self::record_op(GradNode {
            grad_op: GradOp::Broadcast { new_shape: new_shape.to_vec() },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![OLD_NDIM],
            output_id: result.id,
            output_ndim: NEW_NDIM,
        });
        result.into()
    }
}
