use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nn_autograd::autograd::{clear_grad_storage, clear_tape, Autograd};
use nn_autograd::autograd::engine::ReverseMode;
use nn_core::cpu::CPUBackend;
use nn_core::cpu::{mark_generation_start, truncate_to_generation};
use nn_core::linalg::tensor::Tensor;
use nn_nn::nn::activation::{LogSoftmax, ReLU};
use nn_nn::nn::linear::Linear;
use nn_nn::nn::loss::cross_entropy;
use nn_nn::nn::models::Sequential;
use nn_nn::nn::optimizer::{Optimizer, SGD};
use nn_nn::nn::Layer;
use nn_autograd::autograd::AutogradTensor;

fn make_mnist_net() -> Sequential<CPUBackend> {
    Sequential::new(vec![
        Layer::Linear(Linear::new(784, 256)),
        Layer::ReLU(ReLU::new()),
        Layer::Linear(Linear::new(256, 128)),
        Layer::ReLU(ReLU::new()),
        Layer::Linear(Linear::new(128, 10)),
        Layer::LogSoftmax(LogSoftmax::new()),
    ])
}

fn bench_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward_pass");
    group.sample_size(20);

    for &batch_size in &[1usize, 32, 64, 128, 256, 512] {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("mnist_net", batch_size),
            &batch_size,
            |b, &bs| {
                let net = make_mnist_net();
                let mut opt = SGD::new(0.01);
                let params = net.parameters();

                // One-hot targets: batch_size rows, 10 columns, class 0 = 1.
                let target_data: Vec<f32> = (0..bs * 10)
                    .map(|i| if i % 10 == 0 { 1.0 } else { 0.0 })
                    .collect();

                mark_generation_start();

                b.iter_custom(|iters| {
                    let start = std::time::Instant::now();
                    for _ in 0..iters {
                        let input = Tensor::<Autograd<CPUBackend>, 2>::new(
                            vec![0.5f32; bs * 784],
                            [bs, 784],
                        );
                        let target = Tensor::<Autograd<CPUBackend>, 2>::new(
                            target_data.clone(),
                            [bs, 10],
                        );
                        let output = net.forward(&input);
                        let loss = cross_entropy(&output, &target);
                        loss.backward::<ReverseMode>();
                        opt.step(&params);

                        // Reset for next iteration.
                        clear_tape();
                        clear_grad_storage();
                        truncate_to_generation();
                    }
                    start.elapsed()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_backward);
criterion_main!(benches);
