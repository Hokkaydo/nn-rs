use std::io::{Read, Write};

use nn_autograd::autograd::Autograd;
use nn_autograd::autograd::AutogradTensor;
use nn_la::backend::Backend;
use nn_la::linalg::tensor::{Scalar, Tensor, TensorId};
use rand::Rng;

pub struct Linear<B: Backend> {
    pub weights: Tensor<Autograd<B>, 2>,
    pub bias: Tensor<Autograd<B>, 1>,
}

impl<B: Backend> Linear<B> {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let bound = (1.0 / in_features as Scalar).sqrt();
        let range = rand::distr::Uniform::new(-bound, bound).unwrap();
        let weights: Vec<Scalar> = rand::rng()
            .sample_iter(range)
            .take(in_features * out_features)
            .collect();
        let bias: Vec<Scalar> = rand::rng()
            .sample_iter(range)
            .take(out_features)
            .collect();
        Self {
            weights: Tensor::with_grad(weights, [in_features, out_features]),
            bias: Tensor::with_grad(bias, [out_features]),
        }
    }
}

impl<B: Backend> Linear<B> {
    pub fn forward(&self, input: &Tensor<Autograd<B>, 2>) -> Tensor<Autograd<B>, 2> {
        // input: [batch, in_features], weights: [in_features, out_features] to [batch, out_features]
        let out = input.matmul(&self.weights);
        // bias: [out_features] to broadcast to [batch, out_features]
        out.broadcast_add(&self.bias)
    }

    pub fn parameters(&self) -> Vec<TensorId> {
        vec![self.weights.id, self.bias.id]
    }

    pub fn dump(&self, file: &mut std::io::BufWriter<std::fs::File>) {
        // Write weights
        let weights_data = self.weights.as_slice();
        let weights_len = weights_data.len() as u32;
        file.write_all(&weights_len.to_le_bytes()).unwrap();
        for w in weights_data {
            file.write_all(&w.to_le_bytes()).unwrap();
        }
        // Write bias
        let bias_data = self.bias.as_slice();
        let bias_len = bias_data.len() as u32;
        file.write_all(&bias_len.to_le_bytes()).unwrap();
        for b in bias_data {
            file.write_all(&b.to_le_bytes()).unwrap();
        }
    }

    pub fn restore(file: &mut std::io::BufReader<std::fs::File>) -> Self {
        // Read weights
        let mut weights_len_bytes = [0u8; 4];
        file.read_exact(&mut weights_len_bytes).unwrap();
        let weights_len = u32::from_le_bytes(weights_len_bytes) as usize;
        let mut weights_data = vec![0f32; weights_len];
        for w in &mut weights_data {
            let mut w_bytes = [0u8; 4];
            file.read_exact(&mut w_bytes).unwrap();
            *w = f32::from_le_bytes(w_bytes);
        }
        // Read bias
        let mut bias_len_bytes = [0u8; 4];
        file.read_exact(&mut bias_len_bytes).unwrap();
        let bias_len = u32::from_le_bytes(bias_len_bytes) as usize;
        let mut bias_data = vec![0f32; bias_len];
        for b in &mut bias_data {
            let mut b_bytes = [0u8; 4];
            file.read_exact(&mut b_bytes).unwrap();
            *b = f32::from_le_bytes(b_bytes);
        }
        Self {
            weights: Tensor::with_grad(weights_data, [weights_len / bias_len, bias_len]),
            bias: Tensor::with_grad(bias_data, [bias_len]),
        }
    }
}
