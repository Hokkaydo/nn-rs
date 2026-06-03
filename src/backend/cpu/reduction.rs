use crate::backend::backend::ReductionOps;
use crate::backend::cpu::CPUBackend;
use crate::helpers::compute_strides;
use crate::linalg::tensor::{Scalar, Tensor};

impl ReductionOps<Self> for CPUBackend {
    fn sum<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<&[usize]>,
    ) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let shape = tensor.shape();
        let strides = tensor.strides();

        let mut result_shape = shape;
        if let Some(axes) = axes {
            for &a in axes {
                result_shape[a] = 1;
            }
        } else {
            for d in &mut result_shape {
                *d = 1;
            }
        }

        let result_strides = compute_strides(&result_shape);
        let mut result_data = vec![0.0; result_shape.iter().product()];

        for flat in 0..data.len() {
            let mut tmp = flat;
            let mut result_idx = 0;

            for i in 0..NDIM {
                let dim = if strides[i] == 0 {
                    0
                } else {
                    let d = tmp / strides[i];
                    tmp %= strides[i];
                    d
                };
                let out_dim = match axes {
                    None => 0,
                    Some(axes) if axes.contains(&i) => 0,
                    Some(_) => dim,
                };
                result_idx += out_dim * result_strides[i];
            }
            result_data[result_idx] += data[flat];
        }
        Tensor::new(result_data, result_shape)
    }

    fn mean<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<&[usize]>,
    ) -> Tensor<Self, NDIM> {
        let sum_tensor = Self::sum(tensor, axes);
        let mut count = 1;
        let new_shape = tensor.shape();
        if let Some(axes) = axes {
            for &axis in axes {
                count *= new_shape[axis];
            }
        } else {
            for dim in &new_shape {
                count *= *dim;
            }
        }
        let mean_data: Vec<Scalar> = sum_tensor
            .as_slice()
            .iter()
            .map(|&x| x / (count as Scalar))
            .collect();
        let new_shape = sum_tensor.shape();
        Tensor::new(mean_data, new_shape)
    }

    fn max<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<&[usize]>,
    ) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let shape = tensor.shape();
        let mut result_shape = shape;
        if let Some(axes) = axes {
            for &axis in axes {
                result_shape[axis] = 1;
            }
        } else {
            for dim in &mut result_shape {
                *dim = 1;
            }
        }

        let result_strides = compute_strides(&result_shape);
        let mut result_data = vec![Scalar::MIN; result_shape.iter().product()];

        for idx in 0..data.len() {
            let mut result_idx = 0;
            let mut tmp = idx;
            for i in 0..NDIM {
                let dim_idx = tmp / tensor.strides()[i];
                tmp %= tensor.strides()[i];
                let res_dim = if let Some(axes) = axes {
                    if axes.contains(&i) { 0 } else { dim_idx }
                } else {
                    0
                };
                result_idx += res_dim * result_strides[i];
            }
            if data[idx] > result_data[result_idx] {
                result_data[result_idx] = data[idx];
            }
        }
        Tensor::new(result_data, result_shape)
    }

    fn min<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<&[usize]>,
    ) -> Tensor<Self, NDIM> {
        let data = tensor.as_slice();
        let shape = tensor.shape();
        let mut result_shape = shape;
        if let Some(axes) = axes {
            for &axis in axes {
                result_shape[axis] = 1;
            }
        } else {
            for dim in &mut result_shape {
                *dim = 1;
            }
        }

        let result_strides = compute_strides(&result_shape);
        let mut result_data = vec![Scalar::MAX; result_shape.iter().product()];

        for idx in 0..data.len() {
            let mut result_idx = 0;
            let mut tmp = idx;
            for i in 0..NDIM {
                let dim_idx = tmp / tensor.strides()[i];
                tmp %= tensor.strides()[i];
                let res_dim = if let Some(axes) = axes {
                    if axes.contains(&i) { 0 } else { dim_idx }
                } else {
                    0
                };
                result_idx += res_dim * result_strides[i];
            }
            if data[idx] < result_data[result_idx] {
                result_data[result_idx] = data[idx];
            }
        }
        Tensor::new(result_data, result_shape)
    }

    fn argmax<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
    ) -> Tensor<CPUBackend, { NDIM - 1 }> {
        let data = tensor.as_slice();
        let shape = tensor.shape();

        // Build output shape (remove axis)
        let mut out_shape = [0usize; NDIM - 1];
        for i in 0..NDIM {
            if i < axis {
                out_shape[i] = shape[i];
            } else if i > axis {
                out_shape[i - 1] = shape[i];
            }
        }

        let axis_size = shape[axis];
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();

        let mut out = vec![0.0; outer * inner];

        // Contiguous row-major argmax kernel
        for o in 0..outer {
            for i in 0..inner {
                let mut max_val = Scalar::MIN;
                let mut max_idx = 0;

                let base = o * axis_size * inner + i;

                for a in 0..axis_size {
                    let v = data[base + a * inner];
                    if v > max_val {
                        max_val = v;
                        max_idx = a;
                    }
                }

                out[o * inner + i] = max_idx as Scalar;
            }
        }

        Tensor::new(out, out_shape)
    }
}
