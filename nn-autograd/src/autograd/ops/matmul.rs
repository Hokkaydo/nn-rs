use crate::autograd::{as_inner, wrap, Autograd, GradNode, GradOp};
use nn_core::backend::{Backend, MatMulOps};
use nn_core::linalg::tensor::Tensor;

impl<B: Backend> MatMulOps<Self> for Autograd<B> {
    fn matmul_11(a: &Tensor<Self, 1>, b: &Tensor<Self, 1>) -> Tensor<Self, 1> {
        let result = B::matmul_11(&as_inner(a), &as_inner(b));
        Self::record_op(GradNode {
            grad_op: GradOp::MatMul,
            input_ids: vec![a.id, b.id],
            inputs_ndims: vec![1, 1],   
            output_id: result.id,
            output_ndim: 1,
        });
        wrap(result)
    }

    fn matmul_12(a: &Tensor<Self, 1>, b: &Tensor<Self, 2>) -> Tensor<Self, 1> {
        let result = B::matmul_12(&as_inner(a), &as_inner(b));
        Self::record_op(GradNode {
            grad_op: GradOp::MatMul,
            input_ids: vec![a.id, b.id],
            inputs_ndims: vec![1, 2],
            output_id: result.id,
            output_ndim: 1,
        });
        wrap(result)
    }

    fn matmul_21(a: &Tensor<Self, 2>, b: &Tensor<Self, 1>) -> Tensor<Self, 1> {
        let result = B::matmul_21(&as_inner(a), &as_inner(b));
        Self::record_op(GradNode {
            grad_op: GradOp::MatMul,
            input_ids: vec![a.id, b.id],
            inputs_ndims: vec![2, 1],
            output_id: result.id,
            output_ndim: 1,
        });
        wrap(result)
    }

    fn matmul_22(a: &Tensor<Self, 2>, b: &Tensor<Self, 2>) -> Tensor<Self, 2> {
        let result = B::matmul_22(&as_inner(a), &as_inner(b));
        Self::record_op(GradNode {
            grad_op: GradOp::MatMul,
            input_ids: vec![a.id, b.id],
            inputs_ndims: vec![2, 2],
            output_id: result.id,
            output_ndim: 2,
        });
        wrap(result)
    }

    fn matmul_33(a: &Tensor<Self, 3>, b: &Tensor<Self, 3>) -> Tensor<Self, 3> {
        let result = B::matmul_33(&as_inner(a), &as_inner(b));
        Self::record_op(GradNode {
            grad_op: GradOp::BatchedMatMul,
            input_ids: vec![a.id, b.id],
            inputs_ndims: vec![3, 3],
            output_id: result.id,
            output_ndim: 3,
        });
        wrap(result)
    }
}
