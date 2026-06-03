use crate::backend::backend::MatMulOps;
use crate::backend::cpu::CPUBackend;
use crate::linalg::tensor::Tensor;

impl MatMulOps<Self> for CPUBackend {
    fn matmul_11(a: &Tensor<Self, 1>, b: &Tensor<Self, 1>) -> Tensor<Self, 1> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert_eq!(a_shape[0], b_shape[0], "matmul_11 shape mismatch: {:?} vs {:?}", a_shape, b_shape);
        let a_data = a.as_slice();
        let b_data = b.as_slice();
        let a_s = a.strides();
        let b_s = b.strides();
        let a_off = a.offset();
        let b_off = b.offset();
        let mut dot = 0.0f32;
        for i in 0..a_shape[0] {
            dot += a_data[a_off + i * a_s[0]] * b_data[b_off + i * b_s[0]];
        }
        Tensor::new(vec![dot], [1])
    }

    fn matmul_12(a: &Tensor<Self, 1>, b: &Tensor<Self, 2>) -> Tensor<Self, 1> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert_eq!(a_shape[0], b_shape[0], "matmul_12 shape mismatch: {:?} vs {:?}", a_shape, b_shape);
        let a_data = a.as_slice();
        let b_data = b.as_slice();
        let a_s = a.strides();
        let b_s = b.strides();
        let a_off = a.offset();
        let b_off = b.offset();
        let mut result = vec![0.0f32; b_shape[1]];
        for j in 0..b_shape[1] {
            for i in 0..a_shape[0] {
                result[j] += a_data[a_off + i * a_s[0]] * b_data[b_off + i * b_s[0] + j * b_s[1]];
            }
        }
        Tensor::new(result, [b_shape[1]])
    }

    fn matmul_21(a: &Tensor<Self, 2>, b: &Tensor<Self, 1>) -> Tensor<Self, 1> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert_eq!(a_shape[1], b_shape[0], "matmul_21 shape mismatch: {:?} vs {:?}", a_shape, b_shape);
        let a_data = a.as_slice();
        let b_data = b.as_slice();
        let a_s = a.strides();
        let b_s = b.strides();
        let a_off = a.offset();
        let b_off = b.offset();
        let mut result = vec![0.0f32; a_shape[0]];
        for i in 0..a_shape[0] {
            for j in 0..a_shape[1] {
                result[i] += a_data[a_off + i * a_s[0] + j * a_s[1]] * b_data[b_off + j * b_s[0]];
            }
        }
        Tensor::new(result, [a_shape[0]])
    }

    fn matmul_22(a: &Tensor<Self, 2>, b: &Tensor<Self, 2>) -> Tensor<Self, 2> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert_eq!(a_shape[1], b_shape[0], "matmul_22 shape mismatch: {:?} vs {:?}", a_shape, b_shape);
        let a_data = a.as_slice();
        let b_data = b.as_slice();
        let a_s = a.strides();
        let b_s = b.strides();
        let a_off = a.offset();
        let b_off = b.offset();
        let mut result = vec![0.0f32; a_shape[0] * b_shape[1]];
        for i in 0..a_shape[0] {
            for j in 0..b_shape[1] {
                for k in 0..a_shape[1] {
                    result[i * b_shape[1] + j] +=
                        a_data[a_off + i * a_s[0] + k * a_s[1]]
                            * b_data[b_off + k * b_s[0] + j * b_s[1]];
                }
            }
        }
        Tensor::new(result, [a_shape[0], b_shape[1]])
    }
}
