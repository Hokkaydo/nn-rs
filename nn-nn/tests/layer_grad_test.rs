use nn_autograd::autograd::Autograd;
use nn_autograd::autograd::engine::ReverseMode;
use nn_core::cpu::CPUBackend;
use nn_core::linalg::tensor::Tensor;
use nn_nn::nn::linear::Linear;
use nn_autograd::autograd::AutogradTensor;

type AG = Autograd<CPUBackend>;

#[test]
fn test_linear_layer_backward() {
    // input:[1,4] @ weights:[4,2] + bias:[2] to output:[1,2]
    let input = Tensor::<AG, 2>::new(vec![1.0, 2.0, 3.0, 4.0], [1, 4]);
    let linear = Linear::<CPUBackend>::new(4, 2);
    let output = linear.forward(&input);
    let target = Tensor::<AG, 2>::new(vec![0.0, 0.0], [1, 2]);
    // Simple squared-sum loss so grads are predictable
    let loss = (&output - &target).square().sum();
    loss.backward::<ReverseMode>();

    // Verify both parameters have gradients
    let weights_grad = linear.weights.grad();
    let bias_grad = linear.bias.grad();
    assert!(weights_grad.is_some(), "weights gradient should be computed");
    assert!(bias_grad.is_some(), "bias gradient should be computed");

    let wg = weights_grad.unwrap();
    let bg = bias_grad.unwrap();
    assert_eq!(wg.shape(), [4, 2]);
    assert_eq!(bg.shape(), [2]);
}
