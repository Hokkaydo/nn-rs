//! Numeric (finite-difference) gradient checks.
//!
//! Each test builds a scalar loss `L = f(x)` two ways:
//!  - analytically, via the autograd `backward` pass, and
//!  - numerically, via central finite differences on a CPU forward,
//! then asserts the two agree within tolerance.

use nn_autograd::AutogradTensor;
use nn_autograd::autograd::engine::ReverseMode;
use nn_autograd::autograd::{Autograd, clear_grad_storage, clear_tape};
use nn_la::cpu::CPUBackend;
use nn_la::linalg::tensor::Tensor;

type AG = Autograd<CPUBackend>;
type Scalar = f32;

const EPS: Scalar = 1e-3;
const TOL: Scalar = 2e-2;

/// Build a scalar loss from raw `data`/`shape` using plain (tape-free) CPU tensors.
/// `f` must reduce to a single element so `as_scalar()` is well defined.
fn forward_scalar_2d<F>(data: &[Scalar], shape: [usize; 2], f: &F) -> Scalar
where
    F: Fn(Tensor<CPUBackend, 2>) -> Tensor<CPUBackend, 2>,
{
    let t = Tensor::<CPUBackend, 2>::new(data.to_vec(), shape);
    f(t).as_scalar()
}

/// Central-difference gradient of the scalar loss w.r.t. every input element.
fn numeric_grad_2d<F>(data: &[Scalar], shape: [usize; 2], f: &F) -> Vec<Scalar>
where
    F: Fn(Tensor<CPUBackend, 2>) -> Tensor<CPUBackend, 2>,
{
    let mut grad = vec![0.0; data.len()];
    let mut buf = data.to_vec();
    for i in 0..data.len() {
        let orig = buf[i];
        buf[i] = orig + EPS;
        let plus = forward_scalar_2d(&buf, shape, f);
        buf[i] = orig - EPS;
        let minus = forward_scalar_2d(&buf, shape, f);
        buf[i] = orig;
        grad[i] = (plus - minus) / (2.0 * EPS);
    }
    grad
}

fn assert_close(analytic: &[Scalar], numeric: &[Scalar]) {
    assert_eq!(analytic.len(), numeric.len(), "grad length mismatch");
    for (i, (a, n)) in analytic.iter().zip(numeric.iter()).enumerate() {
        assert!(
            (a - n).abs() <= TOL,
            "grad mismatch at {i}: analytic={a}, numeric={n} (tol={TOL})"
        );
    }
}

/// Run a 2D grad check: `ag_f` builds the loss on the autograd tensor, `cpu_f` the
/// identical math on a plain CPU tensor. Both must produce a scalar.
fn check_2d<AgF, CpuF>(data: Vec<Scalar>, shape: [usize; 2], ag_f: AgF, cpu_f: CpuF)
where
    AgF: Fn(&Tensor<AG, 2>) -> Tensor<AG, 2>,
    CpuF: Fn(Tensor<CPUBackend, 2>) -> Tensor<CPUBackend, 2>,
{
    clear_tape();
    clear_grad_storage();

    let x = Tensor::<AG, 2>::with_grad(data.clone(), shape);
    let loss = ag_f(&x);
    loss.backward::<ReverseMode>();
    let analytic = x.grad().expect("input should have a gradient");

    let numeric = numeric_grad_2d(&data, shape, &cpu_f);
    assert_close(&analytic.as_slice(), &numeric);
}

// A simple non-uniform scalar reduction so different output elements get different
// upstream gradients (a plain `sum` would make many Jacobian columns indistinguishable).
fn weighted_loss_2d(t: Tensor<CPUBackend, 2>) -> Tensor<CPUBackend, 2> {
    let shape = t.shape();
    let n = shape[0] * shape[1];
    let weights: Vec<Scalar> = (0..n).map(|i| 1.0 + i as Scalar * 0.5).collect();
    let w = Tensor::<CPUBackend, 2>::new(weights, shape);
    (&t * &w).sum()
}

fn weighted_loss_2d_ag(t: Tensor<AG, 2>) -> Tensor<AG, 2> {
    let shape = t.shape();
    let n = shape[0] * shape[1];
    let weights: Vec<Scalar> = (0..n).map(|i| 1.0 + i as Scalar * 0.5).collect();
    let w = Tensor::<AG, 2>::new(weights, shape);
    (&t * &w).sum()
}

#[test]
fn grad_check_softmax_2d() {
    // [batch=3, classes=4]
    let data = vec![
        0.1, 0.5, -0.3, 0.8, 1.2, -0.7, 0.4, 0.0, -0.2, 0.9, 0.3, -0.5,
    ];
    check_2d(
        data,
        [3, 4],
        |x| weighted_loss_2d_ag(x.softmax()),
        |x| weighted_loss_2d(x.softmax()),
    );
}

#[test]
fn grad_check_log_softmax_2d() {
    let data = vec![
        0.1, 0.5, -0.3, 0.8, 1.2, -0.7, 0.4, 0.0, -0.2, 0.9, 0.3, -0.5,
    ];
    check_2d(
        data,
        [3, 4],
        |x| weighted_loss_2d_ag(x.log_softmax()),
        |x| weighted_loss_2d(x.log_softmax()),
    );
}

#[test]
fn grad_check_sigmoid_2d() {
    let data = vec![-1.5, -0.5, 0.0, 0.5, 1.5, 2.0];
    check_2d(
        data,
        [2, 3],
        |x| weighted_loss_2d_ag(x.sigmoid()),
        |x| weighted_loss_2d(x.sigmoid()),
    );
}

#[test]
fn grad_check_relu_2d() {
    // Avoid exactly 0 where ReLU is non-differentiable.
    let data = vec![-1.5, 0.7, -0.3, 1.2, 0.9, -2.0];
    check_2d(
        data,
        [2, 3],
        |x| weighted_loss_2d_ag(x.relu()),
        |x| weighted_loss_2d(x.relu()),
    );
}

#[test]
fn grad_check_mean_axis1_2d() {
    // mean over the feature axis, then weighted reduce of the [batch,1] result.
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    check_2d(
        data,
        [2, 3],
        |x| weighted_loss_2d_ag(x.mean(&[1])),
        |x| weighted_loss_2d(x.mean(&[1])),
    );
}

#[test]
fn grad_check_sum_axis0_2d() {
    // sum over the batch axis (axis 0), keepdim -> [1, features].
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    check_2d(
        data,
        [2, 3],
        |x| weighted_loss_2d_ag(x.sum_axis(&[0])),
        |x| weighted_loss_2d(x.sum_axis(&[0])),
    );
}

#[test]
fn grad_check_tanh_2d() {
    let data = vec![-1.5, 0.0, 0.7, -0.3, 1.2, 0.9];
    check_2d(
        data,
        [2, 3],
        |x| weighted_loss_2d_ag(x.tanh()),
        |x| weighted_loss_2d(x.tanh()),
    );
}

#[test]
fn grad_check_gelu_2d() {
    let data = vec![-1.5, -0.5, 0.0, 0.5, 1.0, 1.5];
    check_2d(
        data,
        [2, 3],
        |x| weighted_loss_2d_ag(x.gelu()),
        |x| weighted_loss_2d(x.gelu()),
    );
}

#[test]
fn grad_check_softmax_1d() {
    // Rank-1 softmax: exercises the last-axis (not hardcoded axis 1) backward path.
    let data = vec![0.1, 0.5, -0.3, 0.8];
    let w_data = vec![1.0, 2.0, 3.0, 4.0];

    clear_tape();
    clear_grad_storage();
    let x = Tensor::<AG, 1>::with_grad(data.clone(), [4]);
    let w = Tensor::<AG, 1>::new(w_data.clone(), [4]);
    let loss = (&x.softmax() * &w).sum();
    loss.backward::<ReverseMode>();
    let analytic = x.grad().unwrap();

    let mut numeric = vec![0.0; data.len()];
    let mut buf = data.clone();
    let eval = |buf: &[Scalar]| -> Scalar {
        let x = Tensor::<CPUBackend, 1>::new(buf.to_vec(), [4]);
        let w = Tensor::<CPUBackend, 1>::new(w_data.clone(), [4]);
        (&x.softmax() * &w).sum().as_scalar()
    };
    for i in 0..data.len() {
        let orig = buf[i];
        buf[i] = orig + EPS;
        let plus = eval(&buf);
        buf[i] = orig - EPS;
        let minus = eval(&buf);
        buf[i] = orig;
        numeric[i] = (plus - minus) / (2.0 * EPS);
    }

    assert_eq!(analytic.shape(), [4]);
    assert_close(&analytic.as_slice(), &numeric);
}
