use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nn_core::backend::BinaryOps;
use nn_core::cpu::CPUBackend;
use nn_core::linalg::tensor::Tensor;
use nn_core::cpu::{mark_generation_start, truncate_to_generation};

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("elemwise_add");

    for &n in &[1_000usize, 10_000, 100_000, 1_000_000, 10_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("cpu", n), &n, |b, &sz| {
            let a = Tensor::<CPUBackend, 1>::new(vec![1.0f32; sz], [sz]);
            let bv = Tensor::<CPUBackend, 1>::new(vec![2.0f32; sz], [sz]);
            mark_generation_start();
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let _r = std::hint::black_box(CPUBackend::add(&a, &bv));
                    truncate_to_generation();
                }
                start.elapsed()
            });
        });
    }
    group.finish();
}

fn bench_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("elemwise_mul");

    for &n in &[1_000usize, 10_000, 100_000, 1_000_000, 10_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("cpu", n), &n, |b, &sz| {
            let a = Tensor::<CPUBackend, 1>::new(vec![1.0f32; sz], [sz]);
            let bv = Tensor::<CPUBackend, 1>::new(vec![2.0f32; sz], [sz]);
            mark_generation_start();
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let _r = std::hint::black_box(CPUBackend::mul(&a, &bv));
                    truncate_to_generation();
                }
                start.elapsed()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_add, bench_mul);
criterion_main!(benches);
