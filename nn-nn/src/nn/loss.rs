use nn_autograd::autograd::Autograd;
use nn_core::backend::Backend;
use nn_core::linalg::tensor::Tensor;

/// Mean squared error: mean((pred - target)^2)
pub fn mse<B: Backend, const NDIM: usize>(
    pred: &Tensor<Autograd<B>, NDIM>,
    target: &Tensor<Autograd<B>, NDIM>,
) -> Tensor<Autograd<B>, NDIM> {
    let diff = pred - target;
    diff.square().mean_scalar()
}

/// Cross-entropy loss: nll(log_softmax(pred), target)
/// pred: raw logits [batch, classes], target: one-hot [batch, classes]
pub fn cross_entropy<B: Backend>(
    pred: &Tensor<Autograd<B>, 2>,
    target: &Tensor<Autograd<B>, 2>,
) -> Tensor<Autograd<B>, 2> {
    let log_probs = pred.log_softmax();
    let nll = log_probs * target.clone();
    let neg_mean = nll.mean_scalar();
    -neg_mean
}

/// Binary cross-entropy: mean(-target*log(pred) - (1-target)*log(1-pred))
pub fn binary_cross_entropy<B: Backend>(
    pred: &Tensor<Autograd<B>, 2>,
    target: &Tensor<Autograd<B>, 2>,
) -> Tensor<Autograd<B>, 2> {
    let eps = 1e-7_f32;
    let pred_clipped = pred.clamp(eps, 1.0 - eps);
    let lhs = target * &pred_clipped.log();
    let rhs = (1.0 - target) * (1.0 - &pred_clipped).log();
    -(lhs + rhs).mean_scalar()
}
