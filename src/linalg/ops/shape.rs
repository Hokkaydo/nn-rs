use crate::backend::backend::Backend;
use crate::linalg::tensor::Tensor;

impl<B: Backend, const NDIM: usize> Tensor<B, NDIM> {
    /// Transposes the tensor by reversing the order of its axes.
    /// # Returns
    /// A new tensor with its axes transposed.
    pub fn transpose(&self) -> Self {
        B::transpose(self, None)
    }

    /// Transposes the tensor according to the specified axes.
    /// # Arguments
    /// * `axes` - An array representing the new order of the axes. It must be a permutation of
    /// [0, 1, ..., dim-1]. The i-th axis of the returned array will correspond to the axis numbered
    /// `axes[i]` of the original tensor.
    ///
    /// # Returns
    /// A new tensor with its axes transposed according to the specified order.
    pub fn transpose_axis(&self, axes: [usize; NDIM]) -> Self {
        B::transpose(self, Some(axes))
    }

    /// Reshapes the tensor_old to the specified shape without changing the underlying data.
    /// The total number of elements must remain the same.
    /// # Arguments
    /// * `shape` - A slice representing the new shape of the tensor_old.
    /// # Returns
    /// A new tensor_old with the specified shape.
    pub fn reshape(&self, shape: [usize; NDIM]) -> Self {
        B::reshape(self, shape)
    }

    /// Removes a dimension of size one at the specified position.
    /// # Arguments
    /// * `axis` - The position of the dimension to remove.
    /// # Returns
    /// A new tensor with the specified dimension removed.
    pub fn squeeze(&self, axis: usize) -> Tensor<B, { NDIM - 1 }> {
        B::squeeze(self, axis)
    }

    /// Adds a dimension of size one at the specified position.
    /// # Arguments
    /// * `axis` - The position at which to add the new dimension.
    /// # Returns
    /// A new tensor with an added dimension of size one.
    pub fn unsqueeze(&self, axis: usize) -> Tensor<B, { NDIM + 1 }> {
        B::unsqueeze(self, axis)
    }
}
