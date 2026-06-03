use nn_rs::backend::autograd::Autograd;
use nn_rs::backend::autograd::engine::ReverseMode;
use nn_rs::backend::cpu::CPUBackend;
use nn_rs::linalg::tensor::Tensor;

type AG = Autograd<CPUBackend>;

#[test]
fn test_matmul_22_grad() {
    // a:[2,2] @ b:[2,2] to c:[2,2]
    let a = Tensor::<AG, 2>::with_grad(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    let b = Tensor::<AG, 2>::with_grad(vec![5.0, 6.0, 7.0, 8.0], [2, 2]);
    let c = a.matmul(&b);
    let loss = c.sum();
    loss.backward::<ReverseMode>();

    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();

    // ∂L/∂a = ones @ b.T,  ∂L/∂b = a.T @ ones
    let expected_ga = vec![11.0, 15.0, 11.0, 15.0];
    let expected_gb = vec![4.0, 4.0, 6.0, 6.0];
    assert_eq!(grad_a.shape(), [2, 2]);
    assert_eq!(grad_b.shape(), [2, 2]);
    assert_eq!(grad_a.as_slice(), expected_ga);
    assert_eq!(grad_b.as_slice(), expected_gb);
}
