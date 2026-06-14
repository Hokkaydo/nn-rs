use crate::backend::Backend;
use std::fmt::Debug;
use std::marker::PhantomData;

pub type Scalar = f32;
pub type TensorId = usize;

#[derive(Clone)]
pub struct Tensor<B: Backend, const NDIM: usize> {
    pub id: TensorId,
    pub _backend: PhantomData<B>,
}

impl<B: Backend, const NDIM: usize> Tensor<B, NDIM> {
    /// Creates a new Tensor with the given data and shape.
    pub fn new(data: Vec<Scalar>, shape: [usize; NDIM]) -> Self {
        assert_eq!(
            shape.len(),
            NDIM,
            "Shape length must match tensor dimension"
        );
        assert_eq!(data.len(), shape.iter().product());
        B::tensor(data, shape)
    }

    /// Creates a Tensor from an existing TensorId.
    /// * `id` - The TensorId of the existing tensor_old.
    /// Returns a new Tensor instance.
    pub fn from_id(id: TensorId) -> Self {
        Self { id, _backend: PhantomData }
    }

    pub fn from_raw_parts(data: Vec<Scalar>, shape: [usize; NDIM], strides: [usize; NDIM]) -> Self {
        B::tensor_from_raw_parts(data, shape, strides)
    }

    /// Creates a tensor_old filled with ones wTensorType = Tensor<B, NDIM>ith the specified shape.
    /// * `shape` - A slice representing the shape of the tensor_old.
    ///
    /// Returns a new tensor_old filled with ones.
    pub fn ones(shape: [usize; NDIM]) -> Self {
        let data_size = shape.iter().product();
        let data = vec![1.0; data_size];
        Self::new(data, shape)
    }

    /// Creates a tensor_old filled with zeros with the specified shape.
    /// * `shape` - A slice representing the shape of the tensor_old.
    ///
    /// Returns a new tensor_old filled with zeros.
    pub fn zeros(shape: [usize; NDIM]) -> Self {
        let data_size = shape.iter().product();
        let data = vec![0.0; data_size];
        Self::new(data, shape)
    }

    /// Computes the shape without dimensions of size one.
    /// # Arguments
    /// * `shape` - A slice representing the original shape.
    /// # Returns
    /// A vector representing the reduced shape.
    pub fn reduce_shape(shape: &[usize]) -> Vec<usize> {
        let reduced_shape = shape
            .iter()
            .cloned()
            .filter(|&d| d != 1)
            .collect::<Vec<usize>>();
        if reduced_shape.is_empty() {
            vec![1]
        } else {
            reduced_shape
        }
    }

    /// Increments multidimensional indices for iterating over a stride tensor.
    /// * `indices` - A mutable slice of indices to increment.
    /// * `shape` - A slice representing the shape of the tensor.
    pub fn increment_indices(indices: &mut [usize], shape: &[usize]) {
        for i in (0..indices.len()).rev() {
            indices[i] += 1;
            if indices[i] < shape[i] {
                break;
            }
            indices[i] = 0;
        }
    }

    // /// Sets the value at the specified multidimensional indices.
    // /// * `indices` - A slice of indices for each dimension of the tensor_old.
    // /// * `value` - The value to set at the specified indices.
    // pub fn set(&mut self, indices: [usize; NDIM], value: Scalar) {
    //     let flat_index = self.compute_flat_index(indices);
    //     let mut_data = B::mut_data(self.clone());
    //     mut_data[flat_index] = value;
    // }

    pub fn compute_flat_index(&self, indices: [usize; NDIM]) -> usize {
        let strides = B::strides(self);
        let offset = B::offset(self);
        let shape = B::shape(self);
        assert_eq!(
            indices.len(),
            shape.len(),
            "Indices length must match tensor dimension"
        );

        let mut flat_index = offset;
        for (i, &idx) in indices.iter().enumerate() {
            assert!(
                idx <= shape[i],
                "Index {idx} out of bounds for dimension {i}",
            );
            flat_index += idx * strides[i];
        }
        flat_index
    }

    /// Gets the value at the specified multidimensional indices.
    /// * `indices` - A slice of indices for each dimension of the tensor_old.
    //
    // Returns the value at the specified indices.
    pub fn get(&self, indices: [usize; NDIM]) -> Scalar {
        let data = B::data(self);
        data[self.compute_flat_index(indices)]
    }

    pub fn set(&self, indices: [usize; NDIM], value: Scalar) {
        let flat_idx = self.compute_flat_index(indices);
        B::with_mut_data(self, |data| {
            data[flat_idx] = value;
        });
    }

    /// Checks if the tensor_old is a scalar (i.e., has shape [1]).
    /// # Returns
    /// True if the tensor_old is a scalar, false otherwise.
    pub fn is_scalar(&self) -> bool {
        self.shape().iter().product::<usize>() == 1
    }

    pub fn as_scalar(&self) -> Scalar {
        if !self.is_scalar() {
            panic!("Tensor is not a scalar")
        }
        self.get([0; NDIM])
    }

    pub fn as_slice(&self) -> Vec<Scalar> {
        B::data(self)
    }

    pub fn with_data(&self, f: impl FnOnce(&[Scalar])) {
        B::with_data(self, f);
    }

    pub fn with_mut_data(&self, f: impl FnOnce(&mut [Scalar])) {
        B::with_mut_data(self, f);
    }

    pub fn shape(&self) -> [usize; NDIM] {
        B::shape(self)
    }

    pub fn strides(&self) -> [usize; NDIM] {
        B::strides(self)
    }

    pub fn offset(&self) -> usize {
        B::offset(self)
    }
}

impl<B: Backend, const NDIM: usize> Tensor<B, NDIM> {
    fn debug_min(&self) -> Scalar {
        *self
            .as_slice()
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn debug_max(&self) -> Scalar {
        *self
            .as_slice()
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }
}

impl<B: Backend, const NDIM: usize> Debug for Tensor<B, NDIM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("id", &self.id)
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("offset", &self.offset())
            .field("Internal", &B::internal_debug(self))
            .field(
                "data",
                &format_args!(
                    "mean: {:.4}, sum: {:.4}, min: {:.4}, max: {:.4}",
                    self.mean_scalar().as_scalar(),
                    self.sum().as_scalar(),
                    self.debug_min(),
                    self.debug_max()
                ),
            )
            .field(
                "data sample (10 elements)",
                &format_args!(
                    "{:?}",
                    &self.as_slice()[0..std::cmp::min(10, self.as_slice().len())]
                ),
            )
            .finish()
    }
}
