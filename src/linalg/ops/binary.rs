use crate::backend::backend::Backend;
use crate::linalg::tensor::{Scalar, Tensor};
use std::ops::{Add, Div, Mul, Sub};

/// Addition implementations
/// --------------------------------------

impl<B: Backend, const NDIM: usize> Add for &Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn add(self, other: &Tensor<B, NDIM>) -> Tensor<B, NDIM> {
        B::add(self, other)
    }
}

impl<B: Backend, const NDIM: usize> Add for Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn add(self, other: Tensor<B, NDIM>) -> Self::Output {
        &self + &other
    }
}

impl<B: Backend, const NDIM: usize> Add<Scalar> for &Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn add(self, other: Scalar) -> Self::Output {
        B::add_scalar(self, other)
    }
}

impl<B: Backend, const NDIM: usize> Add<&Tensor<B, NDIM>> for Scalar {
    type Output = Tensor<B, NDIM>;
    fn add(self, other: &Tensor<B, NDIM>) -> Self::Output {
        B::scalar_add(self, other)
    }
}
impl<B: Backend, const NDIM: usize> Add<Scalar> for Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn add(self, other: Scalar) -> Self::Output {
        &self + other
    }
}

impl<B: Backend, const NDIM: usize> Add<Tensor<B, NDIM>> for Scalar {
    type Output = Tensor<B, NDIM>;
    fn add(self, other: Tensor<B, NDIM>) -> Self::Output {
        self + &other
    }
}

/// Subtraction implementations
// --------------------------------------
impl<B: Backend, const NDIM: usize> Sub for &Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn sub(self, other: &Tensor<B, NDIM>) -> Self::Output {
        B::sub(self, other)
    }
}

impl<B: Backend, const NDIM: usize> Sub for Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn sub(self, other: Tensor<B, NDIM>) -> Self::Output {
        &self - &other
    }
}

impl<B: Backend, const NDIM: usize> Sub<Scalar> for &Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn sub(self, other: Scalar) -> Self::Output {
        B::sub_scalar(self, other)
    }
}

impl<B: Backend, const NDIM: usize> Sub<&Tensor<B, NDIM>> for Scalar {
    type Output = Tensor<B, NDIM>;
    fn sub(self, other: &Tensor<B, NDIM>) -> Tensor<B, NDIM> {
        B::scalar_sub(self, other)
    }
}

impl<B: Backend, const NDIM: usize> Sub<Scalar> for Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn sub(self, other: Scalar) -> Tensor<B, NDIM> {
        &self - other
    }
}

impl<B: Backend, const NDIM: usize> Sub<Tensor<B, NDIM>> for Scalar {
    type Output = Tensor<B, NDIM>;
    fn sub(self, other: Tensor<B, NDIM>) -> Tensor<B, NDIM> {
        self - &other
    }
}

/// Scalar & Element-wise multiplication implementations
/// --------------------------------------
impl<B: Backend, const NDIM: usize> Mul for &Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn mul(self, other: &Tensor<B, NDIM>) -> Self::Output {
        B::mul(other, self)
    }
}

impl<B: Backend, const NDIM: usize> Mul for Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn mul(self, other: Tensor<B, NDIM>) -> Self::Output {
        &self * &other
    }
}

impl<B: Backend, const NDIM: usize> Mul<Scalar> for &Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn mul(self, other: Scalar) -> Self::Output {
        B::mul_scalar(self, other)
    }
}

impl<B: Backend, const NDIM: usize> Mul<&Tensor<B, NDIM>> for Scalar {
    type Output = Tensor<B, NDIM>;
    fn mul(self, other: &Tensor<B, NDIM>) -> Self::Output {
        B::scalar_mul(self, other)
    }
}

impl<B: Backend, const NDIM: usize> Mul<Scalar> for Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn mul(self, other: Scalar) -> Self::Output {
        &self * other
    }
}

impl<B: Backend, const NDIM: usize> Mul<Tensor<B, NDIM>> for Scalar {
    type Output = Tensor<B, NDIM>;
    fn mul(self, other: Tensor<B, NDIM>) -> Self::Output {
        self * &other
    }
}

/// Scalar & Element-wise division implementations
/// --------------------------------------
impl<B: Backend, const NDIM: usize> Div for &Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn div(self, other: &Tensor<B, NDIM>) -> Self::Output {
        B::div(self, other)
    }
}

impl<B: Backend, const NDIM: usize> Div for Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn div(self, other: Tensor<B, NDIM>) -> Self::Output {
        &self / &other
    }
}

impl<B: Backend, const NDIM: usize> Div<Scalar> for &Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn div(self, other: Scalar) -> Self::Output {
        B::div_scalar(self, other)
    }
}

impl<B: Backend, const NDIM: usize> Div<&Tensor<B, NDIM>> for Scalar {
    type Output = Tensor<B, NDIM>;
    fn div(self, other: &Tensor<B, NDIM>) -> Self::Output {
        B::scalar_div(self, other)
    }
}

impl<B: Backend, const NDIM: usize> Div<Scalar> for Tensor<B, NDIM> {
    type Output = Tensor<B, NDIM>;
    fn div(self, other: Scalar) -> Self::Output {
        &self / other
    }
}

impl<B: Backend, const NDIM: usize> Div<Tensor<B, NDIM>> for Scalar {
    type Output = Tensor<B, NDIM>;
    fn div(self, other: Tensor<B, NDIM>) -> Self::Output {
        self / &other
    }
}

impl<B: Backend, const NDIM: usize> Tensor<B, NDIM> {
    /// Broadcast addition of a tensor along last dimensions
    /// # Arguments
    /// * `other` - The tensor to add, must be broadcastable to self
    /// # Returns
    /// A new tensor containing the result of the broadcast addition
    pub fn broadcast_add<const BC_DIM: usize>(&self, other: &Tensor<B, BC_DIM>) -> Tensor<B, NDIM> {
        let broadcasted_other = B::broadcast(other, self.shape());
        self + &broadcasted_other
    }
}
