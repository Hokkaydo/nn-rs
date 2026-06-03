use crate::linalg::tensor::{Scalar, Tensor, TensorId};

pub trait ActivationOps<B: Backend> {
    fn sigmoid<const NDIM: usize>(tensor: &Tensor<B, NDIM>) -> Tensor<B, NDIM>;
    fn softmax<const NDIM: usize>(tensor: &Tensor<B, NDIM>) -> Tensor<B, NDIM>;
    fn log_softmax<const NDIM: usize>(tensor: &Tensor<B, NDIM>) -> Tensor<B, NDIM>;
    fn relu<const NDIM: usize>(tensor: &Tensor<B, NDIM>) -> Tensor<B, NDIM>;
}

pub trait UnaryOps<B: Backend> {
    fn neg<const NDIM: usize>(tensor: &Tensor<B, NDIM>) -> Tensor<B, NDIM>;
    fn sqrt<const NDIM: usize>(tensor: &Tensor<B, NDIM>) -> Tensor<B, NDIM>;
    fn exp<const NDIM: usize>(tensor: &Tensor<B, NDIM>) -> Tensor<B, NDIM>;
    fn log<const NDIM: usize>(tensor: &Tensor<B, NDIM>) -> Tensor<B, NDIM>;
    fn abs<const NDIM: usize>(tensor: &Tensor<B, NDIM>) -> Tensor<B, NDIM>;
    fn sign<const NDIM: usize>(tensor: &Tensor<B, NDIM>) -> Tensor<B, NDIM>;
    fn clamp<const NDIM: usize>(
        tensor: &Tensor<B, NDIM>,
        min: Scalar,
        max: Scalar,
    ) -> Tensor<B, NDIM>;
    fn pow<const NDIM: usize>(tensor: &Tensor<B, NDIM>, exponent: Scalar) -> Tensor<B, NDIM>;
}

pub trait BinaryOps<B: Backend> {
    fn add<const NDIM: usize>(tensor: &Tensor<B, NDIM>, other: &Tensor<B, NDIM>)
    -> Tensor<B, NDIM>;
    fn sub<const NDIM: usize>(tensor: &Tensor<B, NDIM>, other: &Tensor<B, NDIM>)
    -> Tensor<B, NDIM>;
    fn mul<const NDIM: usize>(tensor: &Tensor<B, NDIM>, other: &Tensor<B, NDIM>)
    -> Tensor<B, NDIM>;
    fn div<const NDIM: usize>(tensor: &Tensor<B, NDIM>, other: &Tensor<B, NDIM>)
    -> Tensor<B, NDIM>;
}

pub trait MatMulOps<B: Backend> {
    fn matmul_11(a: &Tensor<B, 1>, b: &Tensor<B, 1>) -> Tensor<B, 1>;
    fn matmul_12(a: &Tensor<B, 1>, b: &Tensor<B, 2>) -> Tensor<B, 1>;
    fn matmul_21(a: &Tensor<B, 2>, b: &Tensor<B, 1>) -> Tensor<B, 1>;
    fn matmul_22(a: &Tensor<B, 2>, b: &Tensor<B, 2>) -> Tensor<B, 2>;
}

pub trait ScalarOps<B: Backend> {
    fn add_scalar<const NDIM: usize>(tensor: &Tensor<B, NDIM>, scalar: Scalar) -> Tensor<B, NDIM>;
    fn sub_scalar<const NDIM: usize>(tensor: &Tensor<B, NDIM>, scalar: Scalar) -> Tensor<B, NDIM>;
    fn mul_scalar<const NDIM: usize>(tensor: &Tensor<B, NDIM>, scalar: Scalar) -> Tensor<B, NDIM>;
    fn div_scalar<const NDIM: usize>(tensor: &Tensor<B, NDIM>, scalar: Scalar) -> Tensor<B, NDIM>;
}

pub trait ReverseScalarOps<B: Backend> {
    fn scalar_add<const NDIM: usize>(scalar: Scalar, tensor: &Tensor<B, NDIM>) -> Tensor<B, NDIM>;
    fn scalar_sub<const NDIM: usize>(scalar: Scalar, tensor: &Tensor<B, NDIM>) -> Tensor<B, NDIM>;
    fn scalar_mul<const NDIM: usize>(scalar: Scalar, tensor: &Tensor<B, NDIM>) -> Tensor<B, NDIM>;
    fn scalar_div<const NDIM: usize>(scalar: Scalar, tensor: &Tensor<B, NDIM>) -> Tensor<B, NDIM>;
}

pub trait ReductionOps<B: Backend> {
    fn sum<const NDIM: usize>(tensor: &Tensor<B, NDIM>, axes: Option<&[usize]>) -> Tensor<B, NDIM>;
    fn mean<const NDIM: usize>(tensor: &Tensor<B, NDIM>, axes: Option<&[usize]>)
    -> Tensor<B, NDIM>;
    fn max<const NDIM: usize>(tensor: &Tensor<B, NDIM>, axes: Option<&[usize]>) -> Tensor<B, NDIM>;
    fn min<const NDIM: usize>(tensor: &Tensor<B, NDIM>, axes: Option<&[usize]>) -> Tensor<B, NDIM>;
    fn argmax<const NDIM: usize>(tensor: &Tensor<B, NDIM>, axis: usize) -> Tensor<B, { NDIM - 1 }>;
}

pub trait ShapeOps<B: Backend> {
    fn reshape<const NDIM: usize>(
        tensor: &Tensor<B, NDIM>,
        new_shape: [usize; NDIM],
    ) -> Tensor<B, NDIM>;
    fn transpose<const NDIM: usize>(
        tensor: &Tensor<B, NDIM>,
        axes: Option<[usize; NDIM]>,
    ) -> Tensor<B, NDIM>;
    fn squeeze<const NDIM: usize>(tensor: &Tensor<B, NDIM>, axis: usize)
    -> Tensor<B, { NDIM - 1 }>;
    fn unsqueeze<const NDIM: usize>(
        tensor: &Tensor<B, NDIM>,
        axis: usize,
    ) -> Tensor<B, { NDIM + 1 }>;
    fn broadcast<const OLD_NDIM: usize, const NEW_NDIM: usize>(
        tensor: &Tensor<B, OLD_NDIM>,
        new_shape: [usize; NEW_NDIM],
    ) -> Tensor<B, NEW_NDIM>;
}

pub trait ViewOps<B: Backend> {
    fn slice<const NDIM: usize>(
        tensor: &Tensor<B, NDIM>,
        axis: usize,
        start: usize,
        len: usize,
    ) -> Tensor<B, NDIM>;
    fn gather<const NDIM: usize>(
        tensor: &Tensor<B, NDIM>,
        axis: usize,
        indices: &[usize],
    ) -> Tensor<B, NDIM>;
    fn scatter_add<const NDMIM: usize>(
        target: &Tensor<B, NDMIM>,
        src: &Tensor<B, NDMIM>,
        axis: usize,
        indices: &[usize],
    ) -> Tensor<B, NDMIM>;
}

pub trait TensorOps<B: Backend>:
    UnaryOps<B>
    + BinaryOps<B>
    + ScalarOps<B>
    + ReverseScalarOps<B>
    + ReductionOps<B>
    + ShapeOps<B>
    + ViewOps<B>
    + MatMulOps<B>
    + ActivationOps<B>
{

}

pub trait Backend: Default + Clone + TensorOps<Self> {
    fn name() -> &'static str;
    fn tensor<const NDIM: usize>(data: Vec<Scalar>, shape: [usize; NDIM]) -> Tensor<Self, NDIM>;
    fn tensor_from_raw_parts<const NDIM: usize>(
        data: Vec<Scalar>,
        shape: [usize; NDIM],
        strides: [usize; NDIM],
    ) -> Tensor<Self, NDIM>;

    fn shape<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> [usize; NDIM];
    fn data<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Vec<Scalar>;
    fn with_mut_data<F, const NDIM: usize>(tensor: &Tensor<Self, NDIM>, f: F)
    where
        F: FnOnce(&mut [Scalar]);
    fn strides<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> [usize; NDIM];
    fn offset<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> usize;
    fn internal_debug<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> String;

    /// Returns the shape of any tensor by ID without requiring a compile-time NDIM.
    fn shape_dyn(id: TensorId) -> Vec<usize>;
}
