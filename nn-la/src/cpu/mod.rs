mod activation;
mod binary;
mod matmul;
mod reduction;
mod shape;
mod unary;
mod view;

use crate::allocator::{Allocator, ArenaAllocator};
use crate::backend::{Backend, TensorOps};
use crate::linalg::tensor::{Scalar, Tensor, TensorId};
use std::cell::RefCell;

pub fn mark_generation_start() -> TensorId {
    ALLOCATOR.with(|a| a.borrow_mut().mark_generation_start())
}

pub fn truncate_to_generation() {
    ALLOCATOR.with(|a| a.borrow_mut().truncate_to_generation())
}

pub fn shape_dyn_raw(id: TensorId) -> Vec<usize> {
    ALLOCATOR.with(|a| {
        a.borrow()
            .shape_dyn(id)
            .expect("Tensor not found in allocator")
    })
}

pub fn data_by_id(id: TensorId) -> Vec<crate::linalg::tensor::Scalar> {
    ALLOCATOR.with(|a| a.borrow().data(id).expect("Tensor not found").to_vec())
}

pub fn with_mut_data_by_id<F: FnOnce(&mut [crate::linalg::tensor::Scalar])>(id: TensorId, f: F) {
    ALLOCATOR.with(|a| {
        if let Some(data) = a.borrow_mut().data_mut(id) {
            f(data);
        } else {
            panic!("Tensor not found: id={id}");
        }
    });
}

#[derive(Clone, Default)]
pub struct CPUBackend;

impl TensorOps<Self> for CPUBackend {}

thread_local! {
    static ALLOCATOR: RefCell<ArenaAllocator> = RefCell::new(ArenaAllocator::default());
}

impl Backend for CPUBackend {
    fn name() -> &'static str {
        "CPU"
    }

    fn tensor<const NDIM: usize>(data: Vec<Scalar>, shape: [usize; NDIM]) -> Tensor<Self, NDIM> {
        let id = ALLOCATOR.with(|allocator| allocator.borrow_mut().allocate(data, shape.to_vec()));

        Tensor::<Self, NDIM> {
            id,
            _backend: std::marker::PhantomData,
        }
    }

    fn tensor_from_raw_parts<const NDIM: usize>(
        data: Vec<Scalar>,
        shape: [usize; NDIM],
        strides: [usize; NDIM],
    ) -> Tensor<Self, NDIM> {
        let id = ALLOCATOR.with(|allocator| {
            allocator
                .borrow_mut()
                .allocate_with_strides(data, shape.to_vec(), strides.to_vec())
        });

        Tensor::<Self, NDIM> {
            id,
            _backend: std::marker::PhantomData,
        }
    }

    fn shape<const NDIM: usize>(tensor: &Tensor<CPUBackend, NDIM>) -> [usize; NDIM] {
        ALLOCATOR.with(|allocator| {
            allocator
                .borrow()
                .shape(tensor.id)
                .expect("Tensor not found")
                .try_into()
                .expect("Shape dimension mismatch")
        })
    }

    fn data<const NDIM: usize>(tensor: &Tensor<CPUBackend, NDIM>) -> Vec<Scalar> {
        ALLOCATOR.with(|allocator| {
            allocator
                .borrow()
                .data(tensor.id)
                .expect("Tensor not found")
                .to_vec()
        })
    }

    fn with_data<F, const NDIM: usize>(tensor: &Tensor<CPUBackend, NDIM>, f: F)
    where
        F: FnOnce(&[Scalar]),
    {
        ALLOCATOR.with(|allocator| {
            if let Some(data) = allocator.borrow().data(tensor.id) {
                f(data);
            } else {
                panic!("Tensor not found: id={}", tensor.id);
            }
        });
    }

    fn with_mut_data<F, const NDIM: usize>(tensor: &Tensor<CPUBackend, NDIM>, f: F)
    where
        F: FnOnce(&mut [Scalar]),
    {
        ALLOCATOR.with(|allocator| {
            if let Some(data) = allocator.borrow_mut().data_mut(tensor.id) {
                f(data);
            } else {
                panic!("Tensor not found");
            }
        });
    }

    fn strides<const NDIM: usize>(tensor: &Tensor<CPUBackend, NDIM>) -> [usize; NDIM] {
        ALLOCATOR.with(|allocator| {
            allocator
                .borrow()
                .strides(tensor.id)
                .expect("Tensor not found")
                .try_into()
                .expect("Strides dimension mismatch")
        })
    }

    fn offset<const NDIM: usize>(tensor: &Tensor<CPUBackend, NDIM>) -> usize {
        ALLOCATOR.with(|allocator| {
            allocator
                .borrow()
                .offset(tensor.id)
                .expect("Tensor not found")
        })
    }

    fn shape_dyn(id: TensorId) -> Vec<usize> {
        shape_dyn_raw(id)
    }

    fn internal_debug<const NDIM: usize>(tensor: &Tensor<CPUBackend, NDIM>) -> String {
        ALLOCATOR.with(|allocator| {
            let tensor_internal = allocator.borrow();
            let shape = tensor_internal.shape(tensor.id).expect("Tensor not found");
            let strides = tensor_internal
                .strides(tensor.id)
                .expect("Tensor not found");
            let offset = tensor_internal.offset(tensor.id).expect("Tensor not found");
            format!(
                "shape: {:?}, strides: {:?}, offset: {}",
                shape, strides, offset
            )
        })
    }
}
