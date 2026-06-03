pub mod stopper;

pub fn broadcast_dimensions<const NDIM: usize>(
    shape_a: &[usize; NDIM],
    shape_b: &[usize; NDIM],
) -> [usize; NDIM] {
    let mut result_shape = [0; NDIM];
    for i in 0..shape_a.len() {
        if shape_a[i] == shape_b[i] {
            result_shape[i] = shape_a[i];
        } else if shape_a[i] == 1 {
            result_shape[i] = shape_b[i];
        } else if shape_b[i] == 1 {
            result_shape[i] = shape_a[i];
        } else {
            panic!(
                "Shapes are not broadcastable: {:?} and {:?}",
                shape_a, shape_b
            );
        }
    }
    result_shape
}

pub fn compute_strides<const NDIM: usize>(shape: &[usize; NDIM]) -> [usize; NDIM] {
    let mut strides = [0; NDIM];
    let mut stride = 1;
    for i in (0..NDIM).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}
