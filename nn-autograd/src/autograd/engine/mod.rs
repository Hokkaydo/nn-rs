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

pub(crate) enum GradValue<B: Backend> {
    Rank1(Tensor<B, 1>),
    Rank2(Tensor<B, 2>),
}

impl<B: Backend> Clone for GradValue<B> {
    fn clone(&self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(t.clone()),
            Self::Rank2(t) => Self::Rank2(t.clone()),
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
            n => panic!("GradValue::zeros_like_id: unsupported rank {n}"),
        }
    }

    pub fn id(&self) -> TensorId {
        match self {
            Self::Rank1(t) => t.id,
            Self::Rank2(t) => t.id,
        }
    }

    pub fn ndim(&self) -> usize {
        match self {
            Self::Rank1(_) => 1,
            Self::Rank2(_) => 2,
        }
    }

    pub fn shape_vec(&self) -> Vec<usize> {
        match self {
            Self::Rank1(t) => t.shape().to_vec(),
            Self::Rank2(t) => t.shape().to_vec(),
        }
    }

    pub fn as_scalar(&self) -> Scalar {
        match self {
            Self::Rank1(t) => t.as_scalar(),
            Self::Rank2(t) => t.as_scalar(),
        }
    }

    pub fn add_into(self, other: Self) -> Self {
        match (self, other) {
            (Self::Rank1(a), Self::Rank1(b)) => Self::Rank1(B::add(&a, &b)),
            (Self::Rank2(a), Self::Rank2(b)) => Self::Rank2(B::add(&a, &b)),
            _ => panic!("GradValue::add_into: rank mismatch"),
        }
    }

    pub fn sub(self, other: &Self) -> Self {
        match (self, other) {
            (Self::Rank1(a), Self::Rank1(b)) => Self::Rank1(B::sub(&a, b)),
            (Self::Rank2(a), Self::Rank2(b)) => Self::Rank2(B::sub(&a, b)),
            _ => panic!("GradValue::sub: rank mismatch"),
        }
    }

    pub fn mul(self, other: &Self) -> Self {
        match (self, other) {
            (Self::Rank1(a), Self::Rank1(b)) => Self::Rank1(B::mul(&a, b)),
            (Self::Rank2(a), Self::Rank2(b)) => Self::Rank2(B::mul(&a, b)),
            _ => panic!("GradValue::mul: rank mismatch"),
        }
    }

    pub fn div(self, other: &Self) -> Self {
        match (self, other) {
            (Self::Rank1(a), Self::Rank1(b)) => Self::Rank1(B::div(&a, b)),
            (Self::Rank2(a), Self::Rank2(b)) => Self::Rank2(B::div(&a, b)),
            _ => panic!("GradValue::div: rank mismatch"),
        }
    }

    pub fn neg(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::neg(&t)),
            Self::Rank2(t) => Self::Rank2(B::neg(&t)),
        }
    }

    pub fn exp(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::exp(&t)),
            Self::Rank2(t) => Self::Rank2(B::exp(&t)),
        }
    }

    pub fn log(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::log(&t)),
            Self::Rank2(t) => Self::Rank2(B::log(&t)),
        }
    }

    pub fn sign(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::sign(&t)),
            Self::Rank2(t) => Self::Rank2(B::sign(&t)),
        }
    }

    pub fn abs(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::abs(&t)),
            Self::Rank2(t) => Self::Rank2(B::abs(&t)),
        }
    }

    pub fn relu(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::relu(&t)),
            Self::Rank2(t) => Self::Rank2(B::relu(&t)),
        }
    }

    pub fn gelu(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::gelu(&t)),
            Self::Rank2(t) => Self::Rank2(B::gelu(&t)),
        }
    }

    pub fn pow(self, exp: Scalar) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::pow(&t, exp)),
            Self::Rank2(t) => Self::Rank2(B::pow(&t, exp)),
        }
    }

    pub fn tanh(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::tanh(&t)),
            Self::Rank2(t) => Self::Rank2(B::tanh(&t)),
        }
    }

    pub fn mul_scalar(self, s: Scalar) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::mul_scalar(&t, s)),
            Self::Rank2(t) => Self::Rank2(B::mul_scalar(&t, s)),
        }
    }

    pub fn div_scalar(self, s: Scalar) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::div_scalar(&t, s)),
            Self::Rank2(t) => Self::Rank2(B::div_scalar(&t, s)),
        }
    }

    pub fn add_scalar(self, s: Scalar) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::add_scalar(&t, s)),
            Self::Rank2(t) => Self::Rank2(B::add_scalar(&t, s)),
        }
    }

    /// Sum along specific axes, keepdim=true (same rank as input).
    pub fn sum_axes(self, axes: &[usize]) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::sum(&t, Some(axes))),
            Self::Rank2(t) => Self::Rank2(B::sum(&t, Some(axes))),
        }
    }

    /// Sum all elements to scalar GradValue (shape [1] or [1,1]).
    pub fn sum_all(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(B::sum(&t, None)),
            Self::Rank2(t) => Self::Rank2(B::sum(&t, None)),
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
        }
    }

    /// Transpose (reverse axes for 2D; identity for 1D).
    pub fn transpose_default(self) -> Self {
        match self {
            Self::Rank1(t) => Self::Rank1(t),
            Self::Rank2(t) => Self::Rank2(B::transpose(&t, None)),
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
        }
    }
}

pub use reverse::ReverseMode;

#[cfg(test)]
mod tests {
    use super::GradValue;
    use nn_core::cpu::CPUBackend;
    use nn_core::linalg::tensor::Tensor;

    type GV = GradValue<CPUBackend>;

    fn r1(data: Vec<f32>, shape: [usize; 1]) -> GV {
        GV::Rank1(Tensor::<CPUBackend, 1>::new(data, shape))
    }

    fn r2(data: Vec<f32>, shape: [usize; 2]) -> GV {
        GV::Rank2(Tensor::<CPUBackend, 2>::new(data, shape))
    }

    fn slice1(v: &GV) -> Vec<f32> {
        match v {
            GV::Rank1(t) => t.as_slice(),
            GV::Rank2(t) => t.as_slice(),
        }
    }

    fn assert_close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() < 1e-5, "mismatch at {i}: {x} vs {y}");
        }
    }

    // --- constructors / metadata ---

    #[test]
    fn from_id_rank1() {
        let t = Tensor::<CPUBackend, 1>::new(vec![1.0, 2.0], [2]);
        let gv = GV::from_id(t.id, 1);
        assert_eq!(gv.ndim(), 1);
        assert_eq!(gv.shape_vec(), vec![2]);
    }

    #[test]
    fn from_id_rank2() {
        let t = Tensor::<CPUBackend, 2>::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let gv = GV::from_id(t.id, 2);
        assert_eq!(gv.ndim(), 2);
        assert_eq!(gv.shape_vec(), vec![2, 2]);
    }

    #[test]
    #[should_panic(expected = "unsupported rank 3")]
    fn from_id_invalid_rank() {
        let t = Tensor::<CPUBackend, 1>::new(vec![1.0], [1]);
        GV::from_id(t.id, 3);
    }

    #[test]
    fn ones_like_id_rank1() {
        let t = Tensor::<CPUBackend, 1>::new(vec![5.0, 6.0, 7.0], [3]);
        let ones = GV::ones_like_id(t.id, 1);
        assert_eq!(ones.ndim(), 1);
        assert_eq!(slice1(&ones), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn ones_like_id_rank2() {
        let t = Tensor::<CPUBackend, 2>::new(vec![9.0; 6], [2, 3]);
        let ones = GV::ones_like_id(t.id, 2);
        assert_eq!(ones.ndim(), 2);
        assert_eq!(slice1(&ones), vec![1.0; 6]);
    }

    #[test]
    fn zeros_like_id_rank1() {
        let t = Tensor::<CPUBackend, 1>::new(vec![3.0, 4.0], [2]);
        let zeros = GV::zeros_like_id(t.id, 1);
        assert_eq!(slice1(&zeros), vec![0.0, 0.0]);
    }

    #[test]
    fn zeros_like_id_rank2() {
        let t = Tensor::<CPUBackend, 2>::new(vec![1.0; 4], [2, 2]);
        let zeros = GV::zeros_like_id(t.id, 2);
        assert_eq!(slice1(&zeros), vec![0.0; 4]);
    }

    #[test]
    fn as_scalar_rank1() {
        let gv = r1(vec![42.0], [1]);
        assert_eq!(gv.as_scalar(), 42.0);
    }

    #[test]
    fn as_scalar_rank2() {
        let gv = r2(vec![7.0], [1, 1]);
        assert_eq!(gv.as_scalar(), 7.0);
    }

    // --- arithmetic ---

    #[test]
    fn add_into_r1() {
        let a = r1(vec![1.0, 2.0], [2]);
        let b = r1(vec![3.0, 4.0], [2]);
        let c = a.add_into(b);
        assert_eq!(slice1(&c), vec![4.0, 6.0]);
    }

    #[test]
    fn add_into_r2() {
        let a = r2(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let b = r2(vec![4.0, 3.0, 2.0, 1.0], [2, 2]);
        let c = a.add_into(b);
        assert_eq!(slice1(&c), vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "rank mismatch")]
    fn add_into_rank_mismatch() {
        let a = r1(vec![1.0], [1]);
        let b = r2(vec![1.0], [1, 1]);
        let _ = a.add_into(b);
    }

    #[test]
    fn sub_r1() {
        let a = r1(vec![5.0, 3.0], [2]);
        let b = r1(vec![1.0, 2.0], [2]);
        let c = a.sub(&b);
        assert_eq!(slice1(&c), vec![4.0, 1.0]);
    }

    #[test]
    fn mul_r1() {
        let a = r1(vec![2.0, 3.0], [2]);
        let b = r1(vec![4.0, 5.0], [2]);
        let c = a.mul(&b);
        assert_eq!(slice1(&c), vec![8.0, 15.0]);
    }

    #[test]
    fn div_r1() {
        let a = r1(vec![6.0, 9.0], [2]);
        let b = r1(vec![2.0, 3.0], [2]);
        let c = a.div(&b);
        assert_eq!(slice1(&c), vec![3.0, 3.0]);
    }

    #[test]
    fn neg_r1() {
        let a = r1(vec![1.0, -2.0, 3.0], [3]);
        let c = a.neg();
        assert_eq!(slice1(&c), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn mul_scalar_r2() {
        let a = r2(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let c = a.mul_scalar(3.0);
        assert_eq!(slice1(&c), vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn div_scalar_r1() {
        let a = r1(vec![4.0, 6.0], [2]);
        let c = a.div_scalar(2.0);
        assert_eq!(slice1(&c), vec![2.0, 3.0]);
    }

    #[test]
    fn add_scalar_r1() {
        let a = r1(vec![1.0, 2.0], [2]);
        let c = a.add_scalar(10.0);
        assert_eq!(slice1(&c), vec![11.0, 12.0]);
    }

    // --- unary ops ---

    #[test]
    fn exp_r1() {
        let a = r1(vec![0.0, 1.0], [2]);
        let c = a.exp();
        assert_close(&slice1(&c), &[1.0, std::f32::consts::E]);
    }

    #[test]
    fn log_r1() {
        let a = r1(vec![1.0, std::f32::consts::E], [2]);
        let c = a.log();
        assert_close(&slice1(&c), &[0.0, 1.0]);
    }

    #[test]
    fn abs_r1() {
        let a = r1(vec![-3.0, 4.0], [2]);
        let c = a.abs();
        assert_eq!(slice1(&c), vec![3.0, 4.0]);
    }

    #[test]
    fn sign_r1() {
        let a = r1(vec![-5.0, 0.0, 3.0], [3]);
        let c = a.sign();
        assert_eq!(slice1(&c), vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn relu_r1() {
        let a = r1(vec![-1.0, 0.0, 2.0], [3]);
        let c = a.relu();
        assert_eq!(slice1(&c), vec![0.0, 0.0, 2.0]);
    }

    #[test]
    fn gelu_r1() {
        let data = vec![-1.0, 0.0, 1.0];
        let a = r1(data.clone(), [3]);
        let c = a.gelu();
        let expected: Vec<f32> = data.iter().map(|&x| 0.5 * x * (1.0 + (x / 2.0f32.sqrt()).tanh())).collect();
        assert_close(&slice1(&c), &expected);
    }

    #[test]
    fn tanh_r1() {
        let data = vec![-1.0, 0.0, 1.0];
        let a = r1(data.clone(), [3]);
        let c = a.tanh();
        let expected: Vec<f32> = data.iter().map(|&x| x.tanh()).collect();
        assert_close(&slice1(&c), &expected);
    }

    #[test]
    fn pow_r1() {
        let a = r1(vec![2.0, 3.0], [2]);
        let c = a.pow(3.0);
        assert_close(&slice1(&c), &[8.0, 27.0]);
    }

    // --- reduction ---

    #[test]
    fn sum_axes_r2() {
        // [2,3] sum over axis 0 keepdim → [1,3]
        let a = r2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let c = a.sum_axes(&[0]);
        assert_eq!(c.ndim(), 2);
        assert_eq!(c.shape_vec(), vec![1, 3]);
        assert_close(&slice1(&c), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn sum_all_r1() {
        // sum(None) on rank-1 produces a scalar tensor of shape [1]
        let a = r1(vec![1.0, 2.0, 3.0], [3]);
        let c = a.sum_all();
        assert_eq!(c.ndim(), 1);
        assert_eq!(c.shape_vec(), vec![1]);
        assert_close(&slice1(&c), &[6.0]);
    }

    #[test]
    fn sum_all_r2() {
        // sum(None) on rank-2 produces a scalar of shape [1,1]
        let a = r2(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let c = a.sum_all();
        assert_eq!(c.ndim(), 2);
        assert_eq!(c.shape_vec(), vec![1, 1]);
        assert_close(&slice1(&c), &[10.0]);
    }

    // --- shape ---

    #[test]
    fn transpose_default_r1_is_identity() {
        let a = r1(vec![1.0, 2.0, 3.0], [3]);
        let original = slice1(&a);
        let c = a.transpose_default();
        assert_eq!(c.ndim(), 1);
        assert_eq!(slice1(&c), original);
    }

    #[test]
    fn transpose_default_r2() {
        // Input [2,3]: row0=[1,2,3], row1=[4,5,6]
        // After transpose to [3,2]: col0=[1,4], col1=[2,5], col2=[3,6]
        let a = r2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let c = a.transpose_default();
        assert_eq!(c.ndim(), 2);
        assert_eq!(c.shape_vec(), vec![3, 2]);
        // Use element access (respects strides) rather than as_slice (raw buffer order)
        let t = match &c { GV::Rank2(t) => t, _ => panic!() };
        assert_eq!(t.get([0, 0]), 1.0);
        assert_eq!(t.get([0, 1]), 4.0);
        assert_eq!(t.get([1, 0]), 2.0);
        assert_eq!(t.get([1, 1]), 5.0);
        assert_eq!(t.get([2, 0]), 3.0);
        assert_eq!(t.get([2, 1]), 6.0);
    }

    #[test]
    fn unsqueeze_r1_to_r2_axis0() {
        let t = Tensor::<CPUBackend, 1>::new(vec![1.0, 2.0, 3.0], [3]);
        let gv = GV::unsqueeze_r1_to_r2(t, 0);
        assert_eq!(gv.ndim(), 2);
        assert_eq!(gv.shape_vec(), vec![1, 3]);
    }

    #[test]
    fn squeeze_r2_to_r1() {
        let t = Tensor::<CPUBackend, 2>::new(vec![1.0, 2.0, 3.0], [1, 3]);
        let gv = GV::squeeze_r2_to_r1(t, 0);
        assert_eq!(gv.ndim(), 1);
        assert_eq!(gv.shape_vec(), vec![3]);
        assert_eq!(slice1(&gv), vec![1.0, 2.0, 3.0]);
    }

    // --- complex helpers ---

    #[test]
    fn eq_mask_matches() {
        let a = r1(vec![1.0, 2.0, 3.0], [3]);
        let b = r1(vec![1.0, 9.0, 3.0], [3]);
        let mask = a.eq_mask(b);
        assert_eq!(slice1(&mask), vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn clamp_mask_interior() {
        // Values inside (min, max) get mask 1, values at or outside get 0
        let a = r1(vec![-2.0, 0.0, 1.0, 3.0, 5.0], [5]);
        let mask = a.clamp_mask(0.0, 3.0);
        // -2 < 0 → 0; 0 == min → 0; 1 interior → 1; 3 == max → 0; 5 > max → 0
        assert_eq!(slice1(&mask), vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn scatter_add_at_r1() {
        let target = r1(vec![0.0, 0.0, 0.0], [3]);
        let src = r1(vec![1.0, 2.0], [2]);
        let result = target.scatter_add_at(src, 0, &[0, 2]);
        assert_eq!(slice1(&result), vec![1.0, 0.0, 2.0]);
    }

    #[test]
    fn gather_at_r2() {
        // [3,2] gather rows 0 and 2
        let a = r2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]);
        let result = a.gather_at(0, &[0, 2]);
        assert_eq!(result.ndim(), 2);
        assert_eq!(result.shape_vec(), vec![2, 2]);
        assert_eq!(slice1(&result), vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn broadcast_to_id_r2() {
        // broadcast [1,3] → [2,3]
        let src = r2(vec![1.0, 2.0, 3.0], [1, 3]);
        let target = Tensor::<CPUBackend, 2>::new(vec![0.0; 6], [2, 3]);
        let result = src.broadcast_to_id(target.id);
        assert_eq!(result.shape_vec(), vec![2, 3]);
        assert_close(&slice1(&result), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn sum_to_shape_of_reduces_axis() {
        // [2,3] sum to [1,3] target
        let src = r2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let target = Tensor::<CPUBackend, 2>::new(vec![0.0; 3], [1, 3]);
        let result = src.sum_to_shape_of(target.id, 2);
        assert_eq!(result.shape_vec(), vec![1, 3]);
        assert_close(&slice1(&result), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn reshape_to_id_r2() {
        // [2,3] → [3,2]
        let src = r2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let target = Tensor::<CPUBackend, 2>::new(vec![0.0; 6], [3, 2]);
        let result = src.reshape_to_id(target.id);
        assert_eq!(result.shape_vec(), vec![3, 2]);
        assert_eq!(slice1(&result), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
