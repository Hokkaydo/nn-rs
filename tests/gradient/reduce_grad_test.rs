use nn_rs::backend::autograd::Autograd;
use nn_rs::backend::autograd::engine::ReverseMode;
use nn_rs::backend::cpu::CPUBackend;
use nn_rs::linalg::tensor::Tensor;

type AG = Autograd<CPUBackend>;

#[test]
fn test_mean_scalar_grad() {
    let a = Tensor::<AG, 2>::with_grad(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    let mean = a.mean_scalar();
    mean.backward::<ReverseMode>();
    let grad_a = a.grad().unwrap();
    let expected = vec![0.25, 0.25, 0.25, 0.25];
    assert_eq!(grad_a.shape(), [2, 2]);
    assert_eq!(grad_a.as_slice(), expected);
}
