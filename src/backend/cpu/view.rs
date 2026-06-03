use crate::backend::backend::ViewOps;
use crate::backend::cpu::CPUBackend;
use crate::linalg::tensor::Tensor;

impl ViewOps<Self> for CPUBackend {
    fn slice<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
        start: usize,
        len: usize,
    ) -> Tensor<Self, NDIM> {
        let shape = tensor.shape();
        assert!(axis < NDIM, "Axis out of bounds");
        assert!(start + len <= shape[axis], "Invalid slice indices");

        let mut new_shape = shape;
        new_shape[axis] = len;
        let mut result_data = Vec::with_capacity(new_shape.iter().product());
        let mut indices = vec![0; NDIM];
        let mut total_elements = 1;
        for &dim in &new_shape {
            total_elements *= dim;
        }
        for i in 0..total_elements {
            let mut idx = i;
            for d in (0..NDIM).rev() {
                indices[d] = idx % new_shape[d];
                idx /= new_shape[d];
            }
            indices[axis] += start;
            let flat_index: usize = indices
                .iter()
                .zip(shape.iter())
                .fold(0, |acc, (&idx, &dim)| acc * dim + idx);
            result_data.push(tensor.as_slice()[flat_index]);
        }
        Tensor::new(result_data, new_shape)
    }

    fn gather<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
        indices: &[usize],
    ) -> Tensor<Self, NDIM> {
        let shape = tensor.shape();
        assert!(axis < NDIM, "Axis out of bounds");

        let mut new_shape = shape;
        new_shape[axis] = indices.len();
        let mut result_data = Vec::with_capacity(new_shape.iter().product());
        let mut multi_indices = vec![0; NDIM];
        let mut total_elements = 1;
        for &dim in &new_shape {
            total_elements *= dim;
        }
        for i in 0..total_elements {
            let mut idx = i;
            for d in (0..NDIM).rev() {
                multi_indices[d] = idx % new_shape[d];
                idx /= new_shape[d];
            }
            multi_indices[axis] = indices[multi_indices[axis]];
            let flat_index: usize = multi_indices
                .iter()
                .zip(shape.iter())
                .fold(0, |acc, (&idx, &dim)| acc * dim + idx);
            result_data.push(tensor.as_slice()[flat_index]);
        }
        Tensor::new(result_data, new_shape)
    }

    fn scatter_add<const NDIM: usize>(
        target: &Tensor<Self, NDIM>,
        src: &Tensor<Self, NDIM>,
        axis: usize,
        indices: &[usize],
    ) -> Tensor<Self, NDIM> {
        assert!(axis < NDIM, "Axis out of bounds");
        let target_shape = target.shape();
        let src_shape = src.shape();
        let src_data = src.as_slice();

        let mut result_data = target.as_slice();
        let src_total: usize = src_shape.iter().product();

        let mut multi_indices = vec![0usize; NDIM];
        for i in 0..src_total {
            let mut idx = i;
            for d in (0..NDIM).rev() {
                multi_indices[d] = idx % src_shape[d];
                idx /= src_shape[d];
            }
            let target_axis_idx = indices[multi_indices[axis]];
            assert!(
                target_axis_idx < target_shape[axis],
                "scatter_add index out of bounds"
            );
            let mut target_multi = multi_indices.clone();
            target_multi[axis] = target_axis_idx;
            let flat_target: usize = target_multi
                .iter()
                .zip(target_shape.iter())
                .fold(0, |acc, (&i, &dim)| acc * dim + i);
            result_data[flat_target] += src_data[i];
        }
        Tensor::new(result_data, target_shape)
    }
}
