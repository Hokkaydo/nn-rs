pub mod activation;
pub mod linear;
pub mod loss;
pub mod models;
pub mod optimizer;

use std::io::{Read, Write};

use crate::backend::autograd::Autograd;
use crate::backend::backend::Backend;
use crate::linalg::tensor::{Tensor, TensorId};
use crate::nn::activation::*;
use crate::nn::linear::*;


pub enum Layer<B: Backend> {
    Linear(Linear<B>),
    ReLU(ReLU<B>),
    Sigmoid(Sigmoid<B>),
    Softmax(Softmax<B>),
    LogSoftmax(LogSoftmax<B>),
}

#[repr(u8)]
enum LayerKind {
    Linear = 0,
    ReLU = 1,
    Sigmoid = 2,
    Softmax = 3,
    LogSoftmax = 4,
}

impl<B: Backend> Layer<B> {
    pub fn forward(&self, input: &Tensor<Autograd<B>, 2>) -> Tensor<Autograd<B>, 2> {
        match self {
            Layer::Linear(l) => l.forward(input),
            Layer::ReLU(a) => a.forward(input),
            Layer::Sigmoid(a) => a.forward(input),
            Layer::Softmax(a) => a.forward(input),
            Layer::LogSoftmax(a) => a.forward(input),
        }
    }

    pub fn parameters(&self) -> Vec<TensorId> {
        match self {
            Layer::Linear(l) => l.parameters(),
            _ => vec![],
        }
    }

    pub fn dump(&self, file: &mut std::io::BufWriter<std::fs::File>) {
        match self {
            Layer::Linear(l) => {
                file.write_all(&[LayerKind::Linear as u8]).unwrap();
                l.dump(file)
            },
            _ => {
                let tag = match self {
                    Layer::ReLU(_) => LayerKind::ReLU,
                    Layer::Sigmoid(_) => LayerKind::Sigmoid,
                    Layer::Softmax(_) => LayerKind::Softmax,
                    Layer::LogSoftmax(_) => LayerKind::LogSoftmax,
                    _ => unreachable!(),
                };
                file.write_all(&[tag as u8]).unwrap();
            },
        }
    }

    pub fn restore(file: &mut std::io::BufReader<std::fs::File>) -> Self {
        let mut tag = [0u8; 1];
        file.read_exact(&mut tag).unwrap();
        match tag[0] {
            x if x == LayerKind::Linear as u8 => Layer::Linear(Linear::restore(file)),
            x if x == LayerKind::ReLU as u8 => Layer::ReLU(ReLU::new()),
            x if x == LayerKind::Sigmoid as u8 => Layer::Sigmoid(Sigmoid::new()),
            x if x == LayerKind::Softmax as u8 => Layer::Softmax(Softmax::new()),
            x if x == LayerKind::LogSoftmax as u8 => Layer::LogSoftmax(LogSoftmax::new()),
            _ => panic!("Unknown layer type"),
        }
    }
}