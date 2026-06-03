use crate::linalg::tensor::{Scalar, TensorId};

pub struct InternalTensor {
    pub data: Vec<Scalar>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

pub trait Allocator {
    fn allocate(&mut self, data: Vec<Scalar>, shape: Vec<usize>) -> TensorId {
        let strides = compute_strides(&shape);
        self.allocate_with_strides(data, shape, strides)
    }
    fn allocate_with_strides(
        &mut self,
        data: Vec<Scalar>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> TensorId;
    fn shape(&self, id: TensorId) -> Option<&[usize]>;
    fn strides(&self, id: TensorId) -> Option<&[usize]>;
    fn data(&self, id: TensorId) -> Option<&[Scalar]>;
    fn offset(&self, id: TensorId) -> Option<usize>;
    fn data_mut(&mut self, id: TensorId) -> Option<&mut [Scalar]>;
}

#[derive(Default)]
pub struct ArenaAllocator {
    tensors: Vec<InternalTensor>,
    next_id: TensorId,
    generation_start: TensorId,
}

/// Computes the strides for a given shape. Strides are used to calculate the memory offset for each dimension.
/// * `shape` - A slice representing the shape of the tensor_old.
///
/// Returns a vector containing the computed strides.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}

impl ArenaAllocator {
    /// Records the current allocation frontier as the start of a new generation.
    /// Tensors allocated after this point (intermediates) can be freed together.
    pub fn mark_generation_start(&mut self) -> TensorId {
        self.generation_start = self.next_id;
        self.next_id
    }

    /// Frees all tensors allocated after the last `mark_generation_start`.
    /// Parameter tensors allocated before the mark are preserved.
    pub fn truncate_to_generation(&mut self) {
        self.tensors.truncate(self.generation_start);
        self.next_id = self.generation_start;
    }

    pub fn shape_dyn(&self, id: TensorId) -> Option<Vec<usize>> {
        self.tensors.get(id).map(|t| t.shape.clone())
    }
}

impl Allocator for ArenaAllocator {
    fn allocate_with_strides(
        &mut self,
        data: Vec<Scalar>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> TensorId {
        assert_eq!(
            shape.len(),
            strides.len(),
            "Shape and strides must have the same length"
        );
        let tensor = InternalTensor {
            data,
            shape,
            strides,
            offset: 0,
        };
        let id = self.next_id;
        self.tensors.push(tensor);
        self.next_id += 1;
        id
    }

    fn shape(&self, id: TensorId) -> Option<&[usize]> {
        self.tensors.get(id).map(|t| t.shape.as_slice())
    }

    fn strides(&self, id: TensorId) -> Option<&[usize]> {
        self.tensors.get(id).map(|t| t.strides.as_slice())
    }

    fn data(&self, id: TensorId) -> Option<&[Scalar]> {
        self.tensors.get(id).map(|t| t.data.as_slice())
    }

    fn offset(&self, id: TensorId) -> Option<usize> {
        self.tensors.get(id).map(|t| t.offset)
    }

    fn data_mut(&mut self, id: TensorId) -> Option<&mut [Scalar]> {
        self.tensors.get_mut(id).map(|t| t.data.as_mut_slice())
    }
}
