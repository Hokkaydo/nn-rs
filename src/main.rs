use nn_rs::backend::backend::Backend;
use nn_rs::backend::cpu::CPUBackend;
use nn_rs::models::mnist::MNIST;
use nn_rs::nn::models::Sequential;
use nn_rs::nn::optimizer::{Adam, Optimizer, SGD};

fn main() {
    let mnist = MNIST::load_mnist();

    println!(
        "MNIST dataset loaded with {} training images and {} test images",
        mnist.train_images.len(),
        mnist.test_images.len()
    );
    let test_batches = mnist.to_batches::<CPUBackend>(&mnist.test_images, &mnist.test_labels, 1000);

    let mut net = if std::env::args().any(|arg| arg == "--load") {
        load()
    } else if std::env::args().any(|arg| arg == "--train") {
        train(&mnist, None)
    } else if std::env::args().any(|arg| arg == "--load-train") {
        let net = load();
        train(&mnist, Some(net))
    } else {
        panic!("Please specify an action: --train, --load, or --load-train");
    };

    let test_accuracy = mnist.test_model(&test_batches, &mut net);
    println!("Test accuracy: {:.2}%", test_accuracy * 100.0);
}

fn train<B: Backend>(mnist: &MNIST, net: Option<Sequential<B>>) -> Sequential<B> {
    let mut train_batches = mnist.to_batches(&mnist.train_images, &mnist.train_labels, 1000);
    println!(
        "Training on {} batches of size {}",
        train_batches.len(),
        train_batches[0].images.shape()[0]
    );
    let optimizer: Box<dyn Optimizer> = if std::env::args().any(|arg| arg == "--adam") {
        Box::from(Adam::new(0.001, 0.9, 0.999, 1e-8))
    } else if std::env::args().any(|arg| arg == "--sgd") {
        Box::from(SGD::new(0.001))
    } else {
        panic!("Please specify an optimizer: --adam or --sgd");
    };

    let net = if let Some(mut trained_net) = net {
        mnist.train(&mut train_batches, 1, optimizer, &mut trained_net);
        trained_net
    } else {
        mnist.train_linear_model(&mut train_batches, 20, optimizer)
        // generates data of scalar from 0 to 1000
        // let inputs = Tensor::new(
        //     (0..100).map(|x| (x as f32)/100.0).collect(),
        //     &[100,1]
        // );
        // let outputs = Tensor::new(
        //     (0..100).map(|x| ((2*x+3) as f32)/100.0).collect(),
        //     &[100,1]
        // );
        //
        // some(
        //     &inputs,
        //     &outputs,
        //     100,
        //     0.01,
        // )
    };

    net.dump("mnist_output.bin");
    println!("Model trained and memory dumped to mnist_output.bin");

    net
}

fn load<B: Backend>() -> Sequential<B> {
    Sequential::restore("mnist_output.bin")
}

// fn main() {
//     let tensor = Tensor::<Autograd<CPUBackend>, 2>::new(vec![1.0, -2.0, 3.0, -4.0], [2, 2]);
//     let result = tensor.abs();
//     let a = &tensor + &tensor;
//     println!("Original Tensor: {:?}", tensor.as_slice());
//     println!("Absolute Tensor: {:?}", result.as_slice());
// }
