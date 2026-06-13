use nn_autograd::AutogradTensor;
use nn_autograd::autograd::Autograd;
use nn_autograd::autograd::engine::ReverseMode;
use nn_core::cpu::CPUBackend;
use nn_core::linalg::tensor::Tensor;

type AG = Autograd<CPUBackend>;

#[test]
fn test_broadcast_add_grad() {
    // a:[3,2] + broadcast(b:[3,1]) to c:[3,2], loss = sum(c)
    let a = Tensor::<AG, 2>::with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]);
    let b = Tensor::<AG, 2>::with_grad(vec![7.0, 8.0, 9.0], [3, 1]);
    let c = a.broadcast_add(&b);
    let loss = c.sum();
    loss.backward::<ReverseMode>();

    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();

    assert_eq!(grad_a.shape(), [3, 2]);
    assert_eq!(grad_b.shape(), [3, 1]);
    assert_eq!(grad_a.as_slice(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    assert_eq!(grad_b.as_slice(), vec![2.0, 2.0, 2.0]);
}

#[test]
fn test_sub_grad() {
    let a = Tensor::<AG, 2>::with_grad(vec![10.0, 20.0, 30.0, 40.0], [2, 2]);
    let b = Tensor::<AG, 2>::with_grad(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    let c = &a - &b;
    let loss = c.sum();
    loss.backward::<ReverseMode>();

    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();

    assert_eq!(grad_a.shape(), [2, 2]);
    assert_eq!(grad_b.shape(), [2, 2]);
    assert_eq!(grad_a.as_slice(), vec![1.0, 1.0, 1.0, 1.0]);
    assert_eq!(grad_b.as_slice(), vec![-1.0, -1.0, -1.0, -1.0]);
}
