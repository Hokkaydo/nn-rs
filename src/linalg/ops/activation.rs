use crate::backend::backend::Backend;
use crate::linalg::tensor::Tensor;

impl<B: Backend, const NDIM: usize> Tensor<B, NDIM> {
    /// Computes the sigmoid of the tensor
    /// # Returns
    /// A tensor containing the sigmoid values
    pub fn sigmoid(&self) -> Self {
        B::sigmoid(self)
    }

    /// Computes the softmax of the tensor
    /// # Returns
    /// A tensor containing the softmax values
    pub fn softmax(&self) -> Self {
        B::softmax(self)
    }

    /// Computes the log-softmax of the tensor
    /// # Returns
    /// A tensor containing the log-softmax values
    pub fn log_softmax(&self) -> Self {
        B::log_softmax(self)
    }

    pub fn relu(&self) -> Self {
        B::relu(self)
    }
}
