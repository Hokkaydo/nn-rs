use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nn_la::backend::MatMulOps;
use nn_la::cpu::CPUBackend;
use nn_la::linalg::tensor::Tensor;
use nn_la::cpu::{mark_generation_start, truncate_to_generation};

fn bench_matmul22(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul22");
    group.sample_size(20);

    for &size in &[32usize, 64, 128, 256, 512, 1024] {
        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::new("cpu", size), &size, |b, &sz| {
            let a = Tensor::<CPUBackend, 2>::new(vec![0.5f32; sz * sz], [sz, sz]);
            let b_mat = Tensor::<CPUBackend, 2>::new(vec![0.5f32; sz * sz], [sz, sz]);
            // Save the arena boundary after input matrices so we can free outputs.
            mark_generation_start();
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let _c = std::hint::black_box(CPUBackend::matmul_22(&a, &b_mat));
                    truncate_to_generation();
                }
                start.elapsed()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_matmul22);
criterion_main!(benches);
