mod ops;
pub mod engine;

use crate::backend::backend::{
    Backend, TensorOps,
};
use crate::linalg::tensor::{Scalar, Tensor, TensorId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use crate::backend::autograd::engine::AutogradStrategy;

#[derive(Clone)]
pub struct Autograd<B: Backend> {
    _backend: PhantomData<B>,
}

#[derive(Clone)]
pub struct GradNode {
    pub input_ids: Vec<TensorId>,
    pub inputs_ndims: Vec<usize>,
    pub output_id: TensorId,
    pub output_ndim: usize,
    pub grad_op: GradOp
}

#[derive(Clone)]
pub enum GradOp {
    // Unary operations
    Neg,
    Pow { exponent: Scalar },
    Exp,
    Log,
    Abs,
    Sign,
    Clamp { min: Scalar, max: Scalar },

    // Binary operations
    Add,
    AddScalar { scalar: Scalar },
    Sub,
    SubScalar { scalar: Scalar },
    ScalarSub { scalar: Scalar },
    Mul,
    MulScalar { scalar: Scalar },
    Div,
    DivScalar { scalar: Scalar },
    ScalarDiv { scalar: Scalar },
    MatMul,

    // Reduction operations
    Sum { axes: Option<Vec<usize>>, },
    Mean { axes: Option<Vec<usize>>, },
    Max { axes: Option<Vec<usize>>, },
    Min { axes: Option<Vec<usize>>, },

    // Shape operations
    Reshape { new_shape: Vec<usize>, },
    Transpose { axes: Option<Vec<usize>>, },
    Squeeze { new_shape: Vec<usize>, axis: usize, },
    Unsqueeze { axis: usize, },
    Broadcast { new_shape: Vec<usize>, },

    // View operations
    Slice { axis: usize, start: usize, len: usize, },
    Gather { axis: usize, indices: Vec<usize>, },
    ScatterAdd { axis: usize, indices: Vec<usize>, },

    // Activation functions
    Sigmoid,
    ReLU,
    Softmax,
    LogSoftmax,
}

thread_local! {
    static AUTOGRAD_TAPE: RefCell<Vec<GradNode>> = RefCell::new(Vec::new());
    static REQUIRES_GRAD: RefCell<HashMap<TensorId, bool>> = RefCell::new(HashMap::new());
    /// Maps parameter TensorId to gradient TensorId after backward().
    static GRAD_STORAGE: RefCell<HashMap<TensorId, TensorId>> = RefCell::new(HashMap::new());
}

/// Clears the current tape and gradient storage. Call between training steps.
pub fn clear_tape() {
    AUTOGRAD_TAPE.with(|t| t.borrow_mut().clear());
}

pub fn clear_grad_storage() {
    GRAD_STORAGE.with(|s| s.borrow_mut().clear());
}

impl<B: Backend> Autograd<B> {
    pub(crate) fn record_op(node: GradNode) {
        AUTOGRAD_TAPE.with(|tape| {
            tape.borrow_mut().push(node);
        });
    }

    fn tensor_with_grad<const NDIM: usize>(data: Vec<Scalar>, shape: [usize; NDIM]) -> Tensor<Self, NDIM> {
        let registered = B::tensor(data, shape);
        REQUIRES_GRAD.with(|list| {
            list.borrow_mut().insert(registered.id, true);
        });
        Tensor::<Self, NDIM> {
            id: registered.id,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Default for Autograd<B> {
    fn default() -> Self {
        Autograd {
            _backend: PhantomData,
        }
    }
}

impl<B: Backend, const NDIM: usize> Tensor<Autograd<B>, NDIM> {
    pub fn backward<S: AutogradStrategy>(&self) {
        REQUIRES_GRAD.with(|list| {
            let requires_grad = list.borrow();
            AUTOGRAD_TAPE.with(|tape| {
                S::backward(self, tape.borrow().clone(), requires_grad.clone());
            })
        })
    }

    /// Returns the gradient of this tensor after backward(), or None if not computed.
    pub fn grad(&self) -> Option<Tensor<Autograd<B>, NDIM>> {
        GRAD_STORAGE.with(|s| {
            s.borrow().get(&self.id).map(|&grad_id| Tensor::from_id(grad_id))
        })
    }

    /// Removes the stored gradient for this tensor.
    pub fn zero_grad(&self) {
        GRAD_STORAGE.with(|s| { s.borrow_mut().remove(&self.id); });
    }

    /// Creates a tensor that requires gradient tracking.
    pub fn with_grad(data: Vec<Scalar>, shape: [usize; NDIM]) -> Self {
        Autograd::<B>::tensor_with_grad(data, shape)
    }
}

impl<B: Backend, const NDIM: usize> Into<Tensor<Autograd<B>, NDIM>> for &Tensor<B, NDIM> {
    fn into(self) -> Tensor<Autograd<B>, NDIM> {
        Tensor::<Autograd<B>, NDIM> {
            id: self.id,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend, const NDIM: usize> Into<Tensor<Autograd<B>, NDIM>> for Tensor<B, NDIM> {
    fn into(self) -> Tensor<Autograd<B>, NDIM> {
        Tensor::<Autograd<B>, NDIM> {
            id: self.id,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend, const NDIM: usize> Into<Tensor<B, NDIM>> for &Tensor<Autograd<B>, NDIM> {
    fn into(self) -> Tensor<B, NDIM> {
        Tensor::<B, NDIM> {
            id: self.id,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> TensorOps<Self> for Autograd<B> {}

impl<B: Backend> Backend for Autograd<B> {
    fn name() -> &'static str {
        "Autograd"
    }

    fn tensor<const NDIM: usize>(data: Vec<Scalar>, shape: [usize; NDIM]) -> Tensor<Self, NDIM> {
        let registered = B::tensor(data, shape);
        REQUIRES_GRAD.with(|list| {
            list.borrow_mut().insert(registered.id, false);
        });
        Tensor::<Self, NDIM> {
            id: registered.id,
            _backend: PhantomData,
        }
    }

    fn tensor_from_raw_parts<const NDIM: usize>(
        data: Vec<Scalar>,
        shape: [usize; NDIM],
        strides: [usize; NDIM],
    ) -> Tensor<Self, NDIM> {
        let registered = B::tensor_from_raw_parts(data, shape, strides);
        Tensor::<Self, NDIM> {
            id: registered.id,
            _backend: PhantomData,
        }
    }

    fn shape<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> [usize; NDIM] {
        B::shape(&tensor.into())
    }

    fn data<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Vec<Scalar> {
        B::data(&tensor.into())
    }

    fn with_mut_data<F, const NDIM: usize>(tensor: &Tensor<Self, NDIM>, f: F)
    where
        F: FnOnce(&mut [Scalar]),
    {
        B::with_mut_data(&tensor.into(), f)
    }

    fn strides<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> [usize; NDIM] {
        B::strides(&tensor.into())
    }

    fn offset<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> usize {
        B::offset(&tensor.into())
    }

    fn internal_debug<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> String {
        B::internal_debug(&tensor.into())
    }

    fn shape_dyn(id: TensorId) -> Vec<usize> {
        B::shape_dyn(id)
    }
}

/// Writes computed gradients into GRAD_STORAGE. Called by the backward engine.
pub(crate) fn store_grad(param_id: TensorId, grad_id: TensorId) {
    GRAD_STORAGE.with(|s| { s.borrow_mut().insert(param_id, grad_id); });
}

pub(crate) fn get_requires_grad(id: TensorId) -> bool {
    REQUIRES_GRAD.with(|r| r.borrow().get(&id).copied().unwrap_or(false))
}

pub fn get_grad_id(param_id: TensorId) -> Option<TensorId> {
    GRAD_STORAGE.with(|s| s.borrow().get(&param_id).copied())
}
