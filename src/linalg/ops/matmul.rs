use crate::backend::backend::Backend;
use crate::linalg::tensor::Tensor;

pub trait MatMul<Rhs> {
    type Output;
    fn matmul(&self, rhs: &Rhs) -> Self::Output;
}

impl<B: Backend> MatMul<Tensor<B, 1>> for Tensor<B, 1> {
    type Output = Tensor<B, 1>;
    fn matmul(&self, rhs: &Tensor<B, 1>) -> Self::Output {
        B::matmul_11(self, rhs)
    }
}

impl<B: Backend> MatMul<Tensor<B, 2>> for Tensor<B, 1> {
    type Output = Tensor<B, 1>;
    fn matmul(&self, rhs: &Tensor<B, 2>) -> Self::Output {
        B::matmul_12(self, rhs)
    }
}

impl<B: Backend> MatMul<Tensor<B, 1>> for Tensor<B, 2> {
    type Output = Tensor<B, 1>;
    fn matmul(&self, rhs: &Tensor<B, 1>) -> Self::Output {
        B::matmul_21(self, rhs)
    }
}

impl<B: Backend> MatMul<Tensor<B, 2>> for Tensor<B, 2> {
    type Output = Tensor<B, 2>;
    fn matmul(&self, rhs: &Tensor<B, 2>) -> Self::Output {
        B::matmul_22(self, rhs)
    }
}

impl<B: Backend, const NDIM: usize> Tensor<B, NDIM> {
    pub fn matmul<Rhs>(&self, rhs: &Rhs) -> <Self as MatMul<Rhs>>::Output
    where
        Self: MatMul<Rhs>,
    {
        <Self as MatMul<Rhs>>::matmul(self, rhs)
    }
}
