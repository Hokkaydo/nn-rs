use crate::autograd::{as_inner, wrap, Autograd, GradNode, GradOp};
use nn_core::backend::{Backend, BinaryOps, ReverseScalarOps, ScalarOps};
use nn_core::linalg::tensor::{Scalar, Tensor};

impl<B: Backend> BinaryOps<Self> for Autograd<B> {
    fn add<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        other: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let result = B::add(&as_inner(tensor), &as_inner(other));
        Self::record_op(GradNode {
            grad_op: GradOp::Add,
            input_ids: vec![tensor.id, other.id],
            inputs_ndims: vec![NDIM, NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn sub<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        other: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let result = B::sub(&as_inner(tensor), &as_inner(other));
        Self::record_op(GradNode {
            grad_op: GradOp::Sub,
            input_ids: vec![tensor.id, other.id],
            inputs_ndims: vec![NDIM, NDIM], 
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn mul<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        other: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let result = B::mul(&as_inner(tensor), &as_inner(other));
        Self::record_op(GradNode {
            grad_op: GradOp::Mul,
            input_ids: vec![tensor.id, other.id],
            inputs_ndims: vec![NDIM, NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn div<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        other: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let result = B::div(&as_inner(tensor), &as_inner(other));
        Self::record_op(GradNode {
            grad_op: GradOp::Div,
            input_ids: vec![tensor.id, other.id],
            inputs_ndims: vec![NDIM, NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }
}

impl<B: Backend> ScalarOps<Self> for Autograd<B> {
    fn add_scalar<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        scalar: Scalar,
    ) -> Tensor<Self, NDIM> {
        let result = B::add_scalar(&as_inner(tensor), scalar);
        Self::record_op(GradNode {
            grad_op: GradOp::AddScalar { scalar },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn sub_scalar<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        scalar: Scalar,
    ) -> Tensor<Self, NDIM> {
        let result = B::sub_scalar(&as_inner(tensor), scalar);
        Self::record_op(GradNode {
            grad_op: GradOp::SubScalar { scalar },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn mul_scalar<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        scalar: Scalar,
    ) -> Tensor<Self, NDIM> {
        let result = B::mul_scalar(&as_inner(tensor), scalar);
        Self::record_op(GradNode {
            grad_op: GradOp::MulScalar { scalar },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn div_scalar<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        scalar: Scalar,
    ) -> Tensor<Self, NDIM> {
        let result = B::div_scalar(&as_inner(tensor), scalar);
        Self::record_op(GradNode {
            grad_op: GradOp::DivScalar { scalar },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }
}

impl<B: Backend> ReverseScalarOps<Self> for Autograd<B> {
    fn scalar_add<const NDIM: usize>(
        scalar: Scalar,
        tensor: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        Self::add_scalar(tensor, scalar)
    }

    fn scalar_sub<const NDIM: usize>(
        scalar: Scalar,
        tensor: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let result = B::scalar_sub(scalar, &as_inner(tensor));
        Self::record_op(GradNode {
            grad_op: GradOp::ScalarSub { scalar },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }

    fn scalar_mul<const NDIM: usize>(
        scalar: Scalar,
        tensor: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        Self::mul_scalar(tensor, scalar)
    }

    fn scalar_div<const NDIM: usize>(
        scalar: Scalar,
        tensor: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let result = B::scalar_div(scalar, &as_inner(tensor));
        Self::record_op(GradNode {
            grad_op: GradOp::ScalarDiv { scalar },
            input_ids: vec![tensor.id],
            inputs_ndims: vec![NDIM],
            output_id: result.id,
            output_ndim: NDIM,
        });
        wrap(result)
    }
}
