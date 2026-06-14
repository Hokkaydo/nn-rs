use crate::backend::Backend;
use crate::linalg::tensor::{Scalar, Tensor};
use std::ops::Neg;

impl<B: Backend, const NDIM: usize> Neg for &Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn neg(self) -> Self::Output {
        B::neg(self)
    }
}

impl<B: Backend, const NDIM: usize> Neg for Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<B: Backend, const NDIM: usize> Tensor<B, NDIM> {
    /// Raises each element of the tensor to the given exponent
    /// # Arguments
    /// * `exponent` - The exponent to raise each element to
    /// # Returns
    /// A tensor containing the results
    pub fn pow(&self, exponent: Scalar) -> Self {
        B::pow(self, exponent)
    }

    /// Computes the square of the tensor
    /// # Returns
    /// A tensor containing the squared values
    pub fn square(&self) -> Self {
        self.pow(2.0)
    }

    /// Computes the square root of the tensor
    /// # Returns
    /// A tensor containing the square root values
    pub fn sqrt(&self) -> Self {
        B::sqrt(self)
    }

    /// Computes the absolute value of the tensor
    /// # Returns
    /// A tensor containing the absolute values
    pub fn abs(&self) -> Self {
        B::abs(self)
    }

    /// Clamps the tensor values between min and max
    /// # Arguments
    /// * `min` - Minimum value
    /// * `max` - Maximum value
    /// # Returns
    /// A tensor with values clamped between min and max
    pub fn clamp(&self, min: Scalar, max: Scalar) -> Self {
        B::clamp(self, min, max)
    }

    /// Computes the natural logarithm of the tensor
    /// # Returns
    /// A tensor containing the logarithm values
    pub fn log(&self) -> Self {
        B::log(self)
    }

    /// Computes the exponential of the tensor
    /// # Returns
    /// A tensor containing the exponential values
    pub fn exp(&self) -> Self {
        B::exp(self)
    }

    /// Computes the sign of the tensor
    /// # Returns
    /// A tensor containing the sign values
    pub fn sign(&self) -> Self {
        B::sign(self)
    }

    pub fn tanh(&self) -> Self {
        B::tanh(self)
    }
}
