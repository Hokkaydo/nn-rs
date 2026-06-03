use crate::backend::autograd::Autograd;
use crate::backend::autograd::engine::ReverseMode;
use crate::backend::backend::Backend;
use crate::helpers::stopper::*;
use crate::linalg::tensor::{Scalar, Tensor};
use crate::nn::loss::mse;
use crate::nn::optimizer::{Optimizer, SGD};
use crate::nn::{linear::Linear, models::Sequential, Layer};
use rand::Rng;
use rand::seq::SliceRandom;
use std::io::Write;

// parametrize tensors

pub fn some<B: Backend>(
    inputs: &Tensor<Autograd<B>, 2>,
    targets: &Tensor<Autograd<B>, 2>,
    epochs: usize,
    mut learning_rate: Scalar,
) -> Sequential<B> {
    let batch_size = inputs.shape()[0];
    let input_size = inputs.shape()[1];
    let output_size = targets.shape()[1];
    let mut net = Sequential::new(vec![Layer::Linear(Linear::new(input_size, output_size))]);

    let mut rng = rand::rng();
    let uniform = rand::distr::Uniform::new(-0.1, 0.1).unwrap();

    let mut file = std::fs::File::create("output.csv").expect("Unable to create file");

    let mut plateau = PlateauDetector::new(
        epochs / 100, // patience
        0.0001,       // threshold
    );

    for epoch in 0..epochs {
        let mut optimizer = SGD::new(learning_rate);
        let mut data: Vec<Scalar> = Vec::new();
        let mut shuffled_indices = (0..batch_size).collect::<Vec<usize>>();
        shuffled_indices.shuffle(&mut rng);

        let noise: Vec<Scalar> = (0..batch_size)
            .map(|_| rng.clone().sample(uniform) as Scalar)
            .collect();
        let noise_tensor = Tensor::<Autograd<B>, 2>::new(noise, [batch_size, 1]);
        let input = inputs.gather(0, &shuffled_indices); // + noise_tensor;
        let target = targets.gather(0, &shuffled_indices);
        let output = net.forward(&input.clone());
        let mut loss = mse(&target, &output);
        let loss_scalar = loss.as_scalar();
        println!("Epoch {epoch}: Loss = {loss_scalar}");
        writeln!(file, "{epoch},{loss_scalar}").expect("Unable to write to file");

        // if epochs > epochs / 10 && plateau.has_plateaued(loss_scalar) {
        //     learning_rate *= 0.8;
        //     println!("Learning rate adjusted to {learning_rate}");
        // }
        // if learning_rate < 0.0001 {
        //     println!("Learning rate too low, stopping training.");
        //     break;
        // }

        loss.backward::<ReverseMode>();
        optimizer.step(&net.parameters());
    }
    // for i in 0..batch_size {
    //     let input = inputs.slice(0, i, 1);
    //     let output = net.forward(input.clone());
    //     // println!("Input: {input:?}, Output: {output:?}");
    // }
    net
}
