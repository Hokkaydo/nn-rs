use nn_rs::backend::autograd::Autograd;
use nn_rs::backend::autograd::engine::ReverseMode;
use nn_rs::backend::cpu::CPUBackend;
use nn_rs::linalg::tensor::Tensor;

type AG = Autograd<CPUBackend>;

#[test]
fn test_pow_grad() {
    let base = Tensor::<AG, 1>::with_grad(vec![2.0, 3.0, 4.0], [3]);
    let pow = base.pow(3.0);
    pow.backward::<ReverseMode>();
    let grad_base = base.grad().unwrap();
    // d/dx x^3 = 3x^2
    let expected = vec![12.0, 27.0, 48.0];
    assert_eq!(grad_base.shape(), [3]);
    assert_eq!(grad_base.as_slice(), expected);
}
