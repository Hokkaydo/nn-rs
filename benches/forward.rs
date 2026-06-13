use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nn_autograd::autograd::{Autograd, clear_tape};
use nn_core::cpu::CPUBackend;
use nn_core::cpu::{mark_generation_start, truncate_to_generation};
use nn_core::linalg::tensor::Tensor;
use nn_layers::nn::activation::{LogSoftmax, ReLU};
use nn_layers::nn::linear::Linear;
use nn_layers::nn::models::Sequential;
use nn_layers::nn::Layer;

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

fn bench_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_pass");
    group.sample_size(20);

    for &batch_size in &[1usize, 32, 64, 128, 256, 512, 1024] {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("mnist_net", batch_size),
            &batch_size,
            |b, &bs| {
                let net = make_mnist_net();
                mark_generation_start();
                b.iter_custom(|iters| {
                    let start = std::time::Instant::now();
                    for _ in 0..iters {
                        let input = Tensor::<Autograd<CPUBackend>, 2>::new(
                            vec![0.5f32; bs * 784],
                            [bs, 784],
                        );
                        let _out = std::hint::black_box(net.forward(&input));
                        clear_tape();
                        truncate_to_generation();
                    }
                    start.elapsed()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);
