use crate::backend::autograd::{Autograd, GradNode, GradOp};
use crate::backend::backend::{Backend, MatMulOps};
use crate::linalg::tensor::Tensor;

impl<B: Backend> MatMulOps<Self> for Autograd<B> {
    fn matmul_11(a: &Tensor<Self, 1>, b: &Tensor<Self, 1>) -> Tensor<Self, 1> {
        let result = B::matmul_11(&a.into(), &b.into());
        Self::record_op(GradNode {
            grad_op: GradOp::MatMul,
            input_ids: vec![a.id, b.id],
            inputs_ndims: vec![1, 1],   
            output_id: result.id,
            output_ndim: 1,
        });
        result.into()
    }

    fn matmul_12(a: &Tensor<Self, 1>, b: &Tensor<Self, 2>) -> Tensor<Self, 1> {
        let result = B::matmul_12(&a.into(), &b.into());
        Self::record_op(GradNode {
            grad_op: GradOp::MatMul,
            input_ids: vec![a.id, b.id],
            inputs_ndims: vec![1, 2],
            output_id: result.id,
            output_ndim: 1,
        });
        result.into()
    }

    fn matmul_21(a: &Tensor<Self, 2>, b: &Tensor<Self, 1>) -> Tensor<Self, 1> {
        let result = B::matmul_21(&a.into(), &b.into());
        Self::record_op(GradNode {
            grad_op: GradOp::MatMul,
            input_ids: vec![a.id, b.id],
            inputs_ndims: vec![2, 1],
            output_id: result.id,
            output_ndim: 1,
        });
        result.into()
    }

    fn matmul_22(a: &Tensor<Self, 2>, b: &Tensor<Self, 2>) -> Tensor<Self, 2> {
        let result = B::matmul_22(&a.into(), &b.into());
        Self::record_op(GradNode {
            grad_op: GradOp::MatMul,
            input_ids: vec![a.id, b.id],
            inputs_ndims: vec![2, 2],
            output_id: result.id,
            output_ndim: 2,
        });
        result.into()
    }
}
