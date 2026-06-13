use crate::autograd::{Autograd, GradNode};
use nn_core::backend::Backend;
use nn_core::linalg::tensor::{Scalar, Tensor, TensorId};
use std::collections::HashMap;
use std::marker::PhantomData;

mod reverse;

pub trait AutogradStrategy {
    fn backward<B: Backend, const NDIM: usize>(
        tensor: &Tensor<Autograd<B>, NDIM>,
        tape: Vec<GradNode>,
        requires_grad: HashMap<TensorId, bool>,
    );
}

pub enum GradValue<B: Backend> {
    Rank1(Tensor<B, 1>),
    Rank2(Tensor<B, 2>),
    Rank3(Tensor<B, 3>),
    Rank4(Tensor<B, 4>),
}

impl<B: Backend> Clone for GradValue<B> {
    fn clone(&self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(t.clone()),
            Self::Rank2(t) => Self::Rank2(t.clone()),
            Self::Rank3(t) => Self::Rank3(t.clone()),
            Self::Rank4(t) => Self::Rank4(t.clone()),
        }
    }
}

impl<B: Backend> GradValue<B> {

    pub fn from_id(id: TensorId, ndim: usize) -> Self {
        match ndim {
            1 => Self::Rank1(Tensor {
                id,
                _backend: PhantomData,
            }),
            2 => Self::Rank2(Tensor {
                id,
                _backend: PhantomData,
            }),
            3 => Self::Rank3(Tensor {
                id,
                _backend: PhantomData,
            }),
            4 => Self::Rank4(Tensor {
                id,
                _backend: PhantomData,
            }),
            n => panic!("GradValue: unsupported rank {n}"),
        }
    }

    /// Ones tensor with the same shape as the tensor stored at `id`.
    pub fn ones_like_id(id: TensorId, ndim: usize) -> Self {
        let shape = B::shape_dyn(id);
        let n: usize = shape.iter().product();
        match ndim {
            1 => {
                let s: [usize; 1] = shape.try_into().unwrap();
                Self::Rank1(Tensor::<B, 1>::new(vec![1.0; n], s))
            }
            2 => {
                let s: [usize; 2] = shape.try_into().unwrap();
                Self::Rank2(Tensor::<B, 2>::new(vec![1.0; n], s))
            }
            3 => {
                let s: [usize; 3] = shape.try_into().unwrap();
                Self::Rank3(Tensor::<B, 3>::new(vec![1.0; n], s))
            }
            4 => {
                let s: [usize; 4] = shape.try_into().unwrap();
                Self::Rank4(Tensor::<B, 4>::new(vec![1.0; n], s))
            }
            n => panic!("GradValue::ones_like_id: unsupported rank {n}"),
        }
    }

    /// Zeros tensor with the same shape as the tensor stored at `id`.
    pub fn zeros_like_id(id: TensorId, ndim: usize) -> Self {
        let shape = B::shape_dyn(id);
        let n: usize = shape.iter().product();
        match ndim {
            1 => {
                let s: [usize; 1] = shape.try_into().unwrap();
                Self::Rank1(Tensor::<B, 1>::new(vec![0.0; n], s))
            }
            2 => {
                let s: [usize; 2] = shape.try_into().unwrap();
                Self::Rank2(Tensor::<B, 2>::new(vec![0.0; n], s))
            }
            3 => {
                let s: [usize; 3] = shape.try_into().unwrap();
                Self::Rank3(Tensor::<B, 3>::new(vec![0.0; n], s))
            }
            4 => {
                let s: [usize; 4] = shape.try_into().unwrap();
                Self::Rank4(Tensor::<B, 4>::new(vec![0.0; n], s))
            }
            n => panic!("GradValue::zeros_like_id: unsupported rank {n}"),
        }
    }

    pub fn id(&self) -> TensorId {
        match self {
            Self::Rank1(t) => t.id,
            Self::Rank2(t) => t.id,
            Self::Rank3(t) => t.id,
            Self::Rank4(t) => t.id,
        }
    }

    pub fn ndim(&self) -> usize {
        match self {
            Self::Rank1(_) => 1,
            Self::Rank2(_) => 2,
            Self::Rank3(_) => 3,
            Self::Rank4(_) => 4,
        }
    }

    pub fn shape_vec(&self) -> Vec<usize> {
        match self {
            Self::Rank1(t) => t.shape().to_vec(),
            Self::Rank2(t) => t.shape().to_vec(),
            Self::Rank3(t) => t.shape().to_vec(),
            Self::Rank4(t) => t.shape().to_vec(),
        }
    }

    pub fn as_scalar(&self) -> Scalar {
        match self {
            Self::Rank1(t) => t.as_scalar(),
            Self::Rank2(t) => t.as_scalar(),
            Self::Rank3(t) => t.as_scalar(),
            Self::Rank4(t) => t.as_scalar(),
        }
    }

    pub fn add_into(self, other: Self) -> Self {
        match (self, other) {
            (Self::Rank1(a), Self::Rank1(b)) => Self::Rank1(B::add(&a, &b)),
            (Self::Rank2(a), Self::Rank2(b)) => Self::Rank2(B::add(&a, &b)),
            (Self::Rank3(a), Self::Rank3(b)) => Self::Rank3(B::add(&a, &b)),
            (Self::Rank4(a), Self::Rank4(b)) => Self::Rank4(B::add(&a, &b)),
            _ => panic!("GradValue::add_into: rank mismatch"),
        }
    }

    pub fn sub(self, other: &Self) -> Self {
        match (self, other) {
            (Self::Rank1(a), Self::Rank1(b)) => Self::Rank1(B::sub(&a, b)),
            (Self::Rank2(a), Self::Rank2(b)) => Self::Rank2(B::sub(&a, b)),
            (Self::Rank3(a), Self::Rank3(b)) => Self::Rank3(B::sub(&a, b)),
            (Self::Rank4(a), Self::Rank4(b)) => Self::Rank4(B::sub(&a, b)),
            _ => panic!("GradValue::sub: rank mismatch"),
        }
    }

    pub fn mul(self, other: &Self) -> Self {
        match (self, other) {
            (Self::Rank1(a), Self::Rank1(b)) => Self::Rank1(B::mul(&a, b)),
            (Self::Rank2(a), Self::Rank2(b)) => Self::Rank2(B::mul(&a, b)),
            (Self::Rank3(a), Self::Rank3(b)) => Self::Rank3(B::mul(&a, b)),
            (Self::Rank4(a), Self::Rank4(b)) => Self::Rank4(B::mul(&a, b)),
            _ => panic!("GradValue::mul: rank mismatch"),
        }
    }

    pub fn div(self, other: &Self) -> Self {
        match (self, other) {
            (Self::Rank1(a), Self::Rank1(b)) => Self::Rank1(B::div(&a, b)),
            (Self::Rank2(a), Self::Rank2(b)) => Self::Rank2(B::div(&a, b)),
            (Self::Rank3(a), Self::Rank3(b)) => Self::Rank3(B::div(&a, b)),
            (Self::Rank4(a), Self::Rank4(b)) => Self::Rank4(B::div(&a, b)),
            _ => panic!("GradValue::div: rank mismatch"),
        }
    }

    pub fn neg(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::neg(&t)),
            Self::Rank2(t) => Self::Rank2(B::neg(&t)),
            Self::Rank3(t) => Self::Rank3(B::neg(&t)),
            Self::Rank4(t) => Self::Rank4(B::neg(&t)),
        }
    }

    pub fn exp(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::exp(&t)),
            Self::Rank2(t) => Self::Rank2(B::exp(&t)),
            Self::Rank3(t) => Self::Rank3(B::exp(&t)),
            Self::Rank4(t) => Self::Rank4(B::exp(&t)),
        }
    }

    pub fn log(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::log(&t)),
            Self::Rank2(t) => Self::Rank2(B::log(&t)),
            Self::Rank3(t) => Self::Rank3(B::log(&t)),
            Self::Rank4(t) => Self::Rank4(B::log(&t)),
        }
    }

    pub fn sign(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::sign(&t)),
            Self::Rank2(t) => Self::Rank2(B::sign(&t)),
            Self::Rank3(t) => Self::Rank3(B::sign(&t)),
            Self::Rank4(t) => Self::Rank4(B::sign(&t)),
        }
    }

    pub fn abs(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::abs(&t)),
            Self::Rank2(t) => Self::Rank2(B::abs(&t)),
            Self::Rank3(t) => Self::Rank3(B::abs(&t)),
            Self::Rank4(t) => Self::Rank4(B::abs(&t)),
        }
    }

    pub fn relu(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::relu(&t)),
            Self::Rank2(t) => Self::Rank2(B::relu(&t)),
            Self::Rank3(t) => Self::Rank3(B::relu(&t)),
            Self::Rank4(t) => Self::Rank4(B::relu(&t)),
        }
    }

    pub fn gelu(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::gelu(&t)),
            Self::Rank2(t) => Self::Rank2(B::gelu(&t)),
            Self::Rank3(t) => Self::Rank3(B::gelu(&t)),
            Self::Rank4(t) => Self::Rank4(B::gelu(&t)),
        }
    }

    pub fn pow(self, exp: Scalar) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::pow(&t, exp)),
            Self::Rank2(t) => Self::Rank2(B::pow(&t, exp)),
            Self::Rank3(t) => Self::Rank3(B::pow(&t, exp)),
            Self::Rank4(t) => Self::Rank4(B::pow(&t, exp)),
        }
    }

    pub fn tanh(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::tanh(&t)),
            Self::Rank2(t) => Self::Rank2(B::tanh(&t)),
            Self::Rank3(t) => Self::Rank3(B::tanh(&t)),
            Self::Rank4(t) => Self::Rank4(B::tanh(&t)),
        }
    }

    pub fn mul_scalar(self, s: Scalar) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::mul_scalar(&t, s)),
            Self::Rank2(t) => Self::Rank2(B::mul_scalar(&t, s)),
            Self::Rank3(t) => Self::Rank3(B::mul_scalar(&t, s)),
            Self::Rank4(t) => Self::Rank4(B::mul_scalar(&t, s)),
        }
    }

    pub fn div_scalar(self, s: Scalar) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::div_scalar(&t, s)),
            Self::Rank2(t) => Self::Rank2(B::div_scalar(&t, s)),
            Self::Rank3(t) => Self::Rank3(B::div_scalar(&t, s)),
            Self::Rank4(t) => Self::Rank4(B::div_scalar(&t, s)),
        }
    }

    pub fn add_scalar(self, s: Scalar) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::add_scalar(&t, s)),
            Self::Rank2(t) => Self::Rank2(B::add_scalar(&t, s)),
            Self::Rank3(t) => Self::Rank3(B::add_scalar(&t, s)),
            Self::Rank4(t) => Self::Rank4(B::add_scalar(&t, s)),
        }
    }

    /// Sum along specific axes, keepdim=true (same rank as input).
    pub fn sum_axes(self, axes: &[usize]) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::sum(&t, Some(axes))),
            Self::Rank2(t) => Self::Rank2(B::sum(&t, Some(axes))),
            Self::Rank3(t) => Self::Rank3(B::sum(&t, Some(axes))),
            Self::Rank4(t) => Self::Rank4(B::sum(&t, Some(axes))),
        }
    }

    /// Sum all elements to scalar GradValue (shape [1] or [1,1]).
    pub fn sum_all(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::sum(&t, None)),
            Self::Rank2(t) => Self::Rank2(B::sum(&t, None)),
            Self::Rank3(t) => Self::Rank3(B::sum(&t, None)),
            Self::Rank4(t) => Self::Rank4(B::sum(&t, None)),
        }
    }

    /// Broadcast this GradValue to match the shape of the tensor at `target_id`.
    pub fn broadcast_to_id(self, target_id: TensorId) -> Self {
        let target_shape = B::shape_dyn(target_id);
        match self {
            Self::Rank1(t) => {
                let s: [usize; 1] = target_shape
                    .try_into()
                    .expect("broadcast_to_id rank mismatch");
                let bcast = B::broadcast(&t, s);
                Self::Rank1(B::add(&bcast, &Tensor::<B, 1>::zeros(s)))
            }
            Self::Rank2(t) => {
                let s: [usize; 2] = target_shape
                    .try_into()
                    .expect("broadcast_to_id rank mismatch");
                let bcast = B::broadcast(&t, s);
                Self::Rank2(B::add(&bcast, &Tensor::<B, 2>::zeros(s)))
            }
            Self::Rank3(t) => {
                let s: [usize; 3] = target_shape
                    .try_into()
                    .expect("broadcast_to_id rank mismatch");
                let bcast = B::broadcast(&t, s);
                Self::Rank3(B::add(&bcast, &Tensor::<B, 3>::zeros(s)))
            }
            Self::Rank4(t) => {
                let s: [usize; 4] = target_shape.try_into()
                    .expect("broadcast_to_id rank mismatch");
                let bcast = B::broadcast(&t, s);
                Self::Rank4(B::add(&bcast, &Tensor::<B, 4>::zeros(s)))
            }
        }
    }

    /// Sum this GradValue's dimensions down to match the shape of the tensor at `target_id`.
    pub fn sum_to_shape_of(self, target_id: TensorId, target_ndim: usize) -> Self {
        let src_shape = self.shape_vec();
        let target_shape = B::shape_dyn(target_id);

        let offset = src_shape.len().saturating_sub(target_shape.len());

        // Identify which axes in src need to be summed
        let axes_to_sum: Vec<usize> = src_shape
            .iter()
            .enumerate()
            .filter_map(|(i, &dim)| {
                if i < offset {
                    Some(i) // extra leading dim added by broadcast: always sum
                } else {
                    let t_i = i - offset;
                    if target_shape[t_i] == 1 && dim > 1 {
                        Some(i)
                    } else {
                        None
                    }
                }
            })
            .collect();

        if axes_to_sum.is_empty() {
            return self;
        }

        let summed = self.sum_axes(&axes_to_sum);

        // If target has fewer dims, squeeze the extra leading dims
        if target_ndim < summed.ndim() {
            let diff = summed.ndim() - target_ndim;
            let mut result = summed;
            for _ in 0..diff {
                result = match result {
                    Self::Rank2(t) => Self::Rank1(B::squeeze(&t, 0)),
                    other => other,
                };
            }
            result
        } else {
            summed
        }
    }

    /// Reshape to match the shape of the tensor stored at `target_id`.
    pub fn reshape_to_id(self, target_id: TensorId) -> Self {
        let target_shape = B::shape_dyn(target_id);
        match self {
            Self::Rank1(t) => {
                let s: [usize; 1] = target_shape.try_into().unwrap();
                Self::Rank1(B::reshape(&t, s))
            }
            Self::Rank2(t) => {
                let s: [usize; 2] = target_shape.try_into().unwrap();
                Self::Rank2(B::reshape(&t, s))
            }
            Self::Rank3(t) => {
                let s: [usize; 3] = target_shape.try_into().unwrap();
                Self::Rank3(B::reshape(&t, s))
            }
            Self::Rank4(t) => {
                let s: [usize; 4] = target_shape.try_into().unwrap();
                Self::Rank4(B::reshape(&t, s))
            }
        }
    }

    /// Transpose (identity for 1D, reverse axes for 2D and higher).
    pub fn transpose_default(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(t),
            Self::Rank2(t) => Self::Rank2(B::transpose(&t, None)),
            Self::Rank3(t) => Self::Rank3(B::transpose(&t, None)),
            Self::Rank4(t) => Self::Rank4(B::transpose(&t, None)),
        }
    }

    /// Unsqueeze a Rank1 tensor at `axis`
    pub fn unsqueeze_r1_to_r2(t: Tensor<B, 1>, axis: usize) -> Self {
        Self::Rank2(B::unsqueeze(&t, axis))
    }

    /// Squeeze a Rank2 tensor at `axis`
    pub fn squeeze_r2_to_r1(t: Tensor<B, 2>, axis: usize) -> Self {
        Self::Rank1(B::squeeze(&t, axis))
    }

    /// Returns mask: 1.0 where `self == other`, 0.0 elsewhere.
    pub fn eq_mask(self, other: Self) -> Self {
        let diff = self.sub(&other);
        let s = diff.sign();
        let a = s.abs();
        // 1 - abs(sign(diff))
        a.neg().add_scalar(1.0)
    }

    /// Clamp backward mask: 1.0 where min < self < max, 0.0 elsewhere.
    pub fn clamp_mask(self, min: Scalar, max: Scalar) -> Self {
        // above_min: sign(relu(a - min)) = 1 where a > min
        let above_min = self.clone().add_scalar(-min).relu().sign();
        // below_max: sign(relu(max - a)) = 1 where a < max
        let below_max = self.neg().add_scalar(max).relu().sign();
        above_min.mul(&below_max)
    }

    pub fn scatter_add_at(self, src: Self, axis: usize, indices: &[usize]) -> Self {
        match (self, src) {
            (Self::Rank1(target), Self::Rank1(s)) => {
                Self::Rank1(B::scatter_add(&target, &s, axis, indices))
            }
            (Self::Rank2(target), Self::Rank2(s)) => {
                Self::Rank2(B::scatter_add(&target, &s, axis, indices))
            }
            _ => panic!("GradValue::scatter_add_at: rank mismatch"),
        }
    }

    /// Gather along `axis` at `indices` (inverse of scatter_add for the src gradient).
    pub fn gather_at(self, axis: usize, indices: &[usize]) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::gather(&t, axis, indices)),
            Self::Rank2(t) => Self::Rank2(B::gather(&t, axis, indices)),
            Self::Rank3(t) => Self::Rank3(B::gather(&t, axis, indices)),
            Self::Rank4(t) => Self::Rank4(B::gather(&t, axis, indices)),
        }
    }
}

pub use reverse::ReverseMode;
