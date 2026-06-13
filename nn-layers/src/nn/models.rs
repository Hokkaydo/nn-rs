use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use nn_autograd::autograd::Autograd;
use nn_core::backend::Backend;
use nn_core::linalg::tensor::{Tensor, TensorId};
use crate::nn::Layer;

pub struct Sequential<B: Backend> {
    layers: Vec<Layer<B>>,
}

impl<B: Backend> Sequential<B> {
    pub fn new(layers: Vec<Layer<B>>) -> Self {
        Self { layers }
    }
}

impl<B: Backend> Sequential<B> {
    pub fn forward(&self, input: &Tensor<Autograd<B>, 2>) -> Tensor<Autograd<B>, 2> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    pub fn parameters(&self) -> Vec<TensorId> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    pub fn dump(&self, file: &str) {
        let file = File::create(file).unwrap();
        let mut writer = BufWriter::new(file);
        let num_layers = self.layers.len() as u32;
        writer.write_all(&num_layers.to_le_bytes()).unwrap();
        for layer in &self.layers {
            layer.dump(&mut writer);
        }
    }

    pub fn restore(file: &str) -> Self {
        let file = File::open(file).expect("Failed to open model file");
        let mut reader = BufReader::new(file);
        let mut num_layers_bytes = [0u8; 4];
        reader.read_exact(&mut num_layers_bytes).unwrap();
        let num_layers = u32::from_le_bytes(num_layers_bytes) as usize;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(Layer::restore(&mut reader));
        }
        Self { layers }
    }
}
