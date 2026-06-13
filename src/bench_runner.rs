/// Usage: 
///     cargo run --bin bench_runner --release [matmul|elemwise|forward|backward|all]
///
/// Output columns: op, size, mean_ns, std_ns

use std::hint::black_box;
use std::time::Instant;

use nn_autograd::autograd::engine::ReverseMode;
use nn_autograd::autograd::{clear_grad_storage, clear_tape, Autograd};
use nn_core::backend::{BinaryOps, MatMulOps};
use nn_core::cpu::{mark_generation_start, truncate_to_generation, CPUBackend};
use nn_core::linalg::tensor::Tensor;
use nn_layers::nn::activation::{LogSoftmax, ReLU};
use nn_layers::nn::linear::Linear;
use nn_layers::nn::loss::cross_entropy;
use nn_layers::nn::models::Sequential;
use nn_layers::nn::optimizer::{Optimizer, SGD};
use nn_layers::nn::Layer;
use nn_autograd::autograd::AutogradTensor;

const WARMUP: usize = 5;
const REPS: usize = 20;

fn stats(times_ns: &[u128]) -> (u64, u64) {
    let mean = times_ns.iter().sum::<u128>() as f64 / times_ns.len() as f64;
    let variance = times_ns
        .iter()
        .map(|&t| (t as f64 - mean).powi(2))
        .sum::<f64>()
        / times_ns.len() as f64;
    (mean as u64, variance.sqrt() as u64)
}

// ---------------------------------------------------------------------------
// matmul22
// ---------------------------------------------------------------------------

fn bench_matmul() {
    let sizes = &std::env::args()
        .skip(2)
        .map(|s| s.parse::<usize>().expect("size must be a positive integer"))
        .collect::<Vec<_>>();

    for &sz in sizes {
        let a = Tensor::<CPUBackend, 2>::new(vec![0.5f32; sz * sz], [sz, sz]);
        let b = Tensor::<CPUBackend, 2>::new(vec![0.5f32; sz * sz], [sz, sz]);
        mark_generation_start();

        for _ in 0..WARMUP {
            let _ = black_box(CPUBackend::matmul_22(&a, &b));
            truncate_to_generation();
        }

        let mut times = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t0 = Instant::now();
            let _ = black_box(CPUBackend::matmul_22(&a, &b));
            times.push(t0.elapsed().as_nanos());
            truncate_to_generation();
        }
        let (mean, std) = stats(&times);
        println!("matmul22,{sz},{mean},{std}");
    }
}

// ---------------------------------------------------------------------------
// element-wise add
// ---------------------------------------------------------------------------

fn bench_elemwise() {
    let sizes = &std::env::args()
        .skip(2)
        .map(|s| s.parse::<usize>().expect("size must be a positive integer"))
        .collect::<Vec<_>>();

    for &sz in sizes {
        let a = Tensor::<CPUBackend, 1>::new(vec![1.0f32; sz], [sz]);
        let b = Tensor::<CPUBackend, 1>::new(vec![2.0f32; sz], [sz]);
        mark_generation_start();

        for _ in 0..WARMUP {
            let _ = black_box(CPUBackend::add(&a, &b));
            truncate_to_generation();
        }

        let mut times = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t0 = Instant::now();
            let _ = black_box(CPUBackend::add(&a, &b));
            times.push(t0.elapsed().as_nanos());
            truncate_to_generation();
        }
        let (mean, std) = stats(&times);
        println!("elemwise_add,{sz},{mean},{std}");
    }
}

// ---------------------------------------------------------------------------
// forward pass (MNIST net)
// ---------------------------------------------------------------------------

fn make_net() -> Sequential<CPUBackend> {
    Sequential::new(vec![
        Layer::Linear(Linear::new(784, 256)),
        Layer::ReLU(ReLU::new()),
        Layer::Linear(Linear::new(256, 128)),
        Layer::ReLU(ReLU::new()),
        Layer::Linear(Linear::new(128, 10)),
        Layer::LogSoftmax(LogSoftmax::new()),
    ])
}

fn bench_forward() {
    let sizes = &std::env::args()
        .skip(2)
        .map(|s| s.parse::<usize>().expect("size must be a positive integer"))
        .collect::<Vec<_>>();

    for &bs in sizes {
        let net = make_net();
        mark_generation_start();

        for _ in 0..WARMUP {
            let input = Tensor::<Autograd<CPUBackend>, 2>::new(vec![0.5f32; bs * 784], [bs, 784]);
            let _ = black_box(net.forward(&input));
            clear_tape();
            truncate_to_generation();
        }

        let mut times = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let input = Tensor::<Autograd<CPUBackend>, 2>::new(vec![0.5f32; bs * 784], [bs, 784]);
            let t0 = Instant::now();
            let _ = black_box(net.forward(&input));
            times.push(t0.elapsed().as_nanos());
            clear_tape();
            truncate_to_generation();
        }
        let (mean, std) = stats(&times);
        println!("forward,{bs},{mean},{std}");
    }
}

// ---------------------------------------------------------------------------
// backward pass (MNIST net, includes forward + backward + SGD step)
// ---------------------------------------------------------------------------

fn bench_backward() {
    let sizes = &std::env::args()
        .skip(2)
        .map(|s| s.parse::<usize>().expect("size must be a positive integer"))
        .collect::<Vec<_>>();

    for &bs in sizes {
        let net = make_net();
        let params = net.parameters();
        let mut opt = SGD::new(0.01);

        let target_data: Vec<f32> = (0..bs * 10)
            .map(|i| if i % 10 == 0 { 1.0 } else { 0.0 })
            .collect();

        mark_generation_start();

        for _ in 0..WARMUP {
            let input = Tensor::<Autograd<CPUBackend>, 2>::new(vec![0.5f32; bs * 784], [bs, 784]);
            let target = Tensor::<Autograd<CPUBackend>, 2>::new(target_data.clone(), [bs, 10]);
            let output = net.forward(&input);
            let loss = cross_entropy(&output, &target);
            loss.backward::<ReverseMode>();
            opt.step(&params);
            clear_tape();
            clear_grad_storage();
            truncate_to_generation();
        }

        let mut times = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let input = Tensor::<Autograd<CPUBackend>, 2>::new(vec![0.5f32; bs * 784], [bs, 784]);
            let target = Tensor::<Autograd<CPUBackend>, 2>::new(target_data.clone(), [bs, 10]);
            let t0 = Instant::now();
            let output = net.forward(&input);
            let loss = cross_entropy(&output, &target);
            loss.backward::<ReverseMode>();
            opt.step(&params);
            times.push(t0.elapsed().as_nanos());
            clear_tape();
            clear_grad_storage();
            truncate_to_generation();
        }
        let (mean, std) = stats(&times);
        println!("backward,{bs},{mean},{std}");
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());

    println!("op,size,mean_ns,std_ns");

    match which.as_str() {
        "matmul"   => bench_matmul(),
        "elemwise" => bench_elemwise(),
        "forward"  => bench_forward(),
        "backward" => bench_backward(),
        _          => {
            bench_matmul();
            bench_elemwise();
            bench_forward();
            bench_backward();
        }
    }
}