use crate::backend::backend::ShapeOps;
use crate::backend::cpu::CPUBackend;
use crate::linalg::tensor::Tensor;

impl ShapeOps<Self> for CPUBackend {
    fn reshape<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        new_shape: [usize; NDIM],
    ) -> Tensor<Self, NDIM> {
        Tensor::new(tensor.as_slice(), new_shape)
    }

    fn transpose<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<[usize; NDIM]>,
    ) -> Tensor<Self, NDIM> {
        let shape = tensor.shape();
        let strides = tensor.strides();
        let axes = axes.unwrap_or_else(|| {
            let mut default_axes = [0; NDIM];
            for i in 0..NDIM {
                default_axes[i] = NDIM - 1 - i;
            }
            default_axes
        });
        let new_shape = axes.map(|ax| shape[ax]);
        let new_strides = axes.map(|ax| strides[ax]);
        Tensor::from_raw_parts(tensor.as_slice(), new_shape, new_strides)
    }

    fn squeeze<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
    ) -> Tensor<Self, { NDIM - 1 }> {
        let shape = tensor.shape();
        let mut new_shape = [0; NDIM - 1];
        for i in 0..axis {
            new_shape[i] = shape[i];
        }
        for i in axis + 1..NDIM {
            new_shape[i - 1] = shape[i];
        }
        Tensor::new(tensor.as_slice(), new_shape)
    }

    fn unsqueeze<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
    ) -> Tensor<Self, { NDIM + 1 }> {
        let shape = tensor.shape();
        let mut new_shape = [0; NDIM + 1];
        for i in 0..axis {
            new_shape[i] = shape[i];
        }
        new_shape[axis] = 1;
        for i in axis..NDIM {
            new_shape[i + 1] = shape[i];
        }
        Tensor::new(tensor.as_slice(), new_shape)
    }

    fn broadcast<const OLD_NDIM: usize, const NEW_NDIM: usize>(
        tensor: &Tensor<Self, OLD_NDIM>,
        new_shape: [usize; NEW_NDIM],
    ) -> Tensor<Self, NEW_NDIM> {
        let stride = tensor.strides();
        let mut new_stride = [0; NEW_NDIM];
        let old_shape = tensor.shape();
        for i in 0..NEW_NDIM {
            if i < NEW_NDIM - OLD_NDIM {
                new_stride[i] = 0;
            } else if old_shape[i - (NEW_NDIM - OLD_NDIM)] == 1 {
                new_stride[i] = 0;
            } else {
                new_stride[i] = stride[i - (NEW_NDIM - OLD_NDIM)];
            }
        }
        Tensor::from_raw_parts(tensor.as_slice(), new_shape, new_stride)
    }
}
