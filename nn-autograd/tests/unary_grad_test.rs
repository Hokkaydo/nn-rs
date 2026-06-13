use nn_autograd::AutogradTensor;
use nn_autograd::autograd::{Autograd, clear_grad_storage, clear_tape};
use nn_autograd::autograd::engine::ReverseMode;
use nn_core::cpu::CPUBackend;
use nn_core::linalg::tensor::Tensor;

type AG = Autograd<CPUBackend>;
type Scalar = f32;

const EPS: Scalar = 1e-3;
const TOL: Scalar = 2e-2;

fn numeric_grad_1d<F>(data: &[Scalar], shape: [usize; 1], f: &F) -> Vec<Scalar>
where
    F: Fn(Tensor<CPUBackend, 1>) -> Tensor<CPUBackend, 1>,
{
    let mut grad = vec![0.0; data.len()];
    let mut buf = data.to_vec();
    for i in 0..data.len() {
        let orig = buf[i];
        buf[i] = orig + EPS;
        let plus = Tensor::<CPUBackend, 1>::new(buf.clone(), shape);
        let plus_val = f(plus).as_scalar();
        buf[i] = orig - EPS;
        let minus = Tensor::<CPUBackend, 1>::new(buf.clone(), shape);
        let minus_val = f(minus).as_scalar();
        buf[i] = orig;
        grad[i] = (plus_val - minus_val) / (2.0 * EPS);
    }
    grad
}

fn assert_close(analytic: &[Scalar], numeric: &[Scalar]) {
    assert_eq!(analytic.len(), numeric.len());
    for (i, (a, n)) in analytic.iter().zip(numeric.iter()).enumerate() {
        assert!(
            (a - n).abs() <= TOL,
            "grad mismatch at {i}: analytic={a}, numeric={n} (tol={TOL})"
        );
    }
}

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

#[test]
fn test_tanh_grad_analytic() {
    // d/dx tanh(x) = 1 - tanh^2(x)
    let data = vec![-1.0, 0.0, 0.5, 1.5];
    clear_tape();
    clear_grad_storage();
    let x = Tensor::<AG, 1>::with_grad(data.clone(), [4]);
    x.tanh().sum().backward::<ReverseMode>();
    let grad = x.grad().unwrap();
    assert_eq!(grad.shape(), [4]);
    for (i, &xi) in data.iter().enumerate() {
        let expected = 1.0 - xi.tanh().powi(2);
        assert!((grad.as_slice()[i] - expected).abs() < 1e-5,
            "tanh grad at {xi}: got {}, expected {expected}", grad.as_slice()[i]);
    }
}

#[test]
fn grad_check_tanh_1d() {
    let data = vec![-1.5, 0.0, 0.7, 1.2];
    clear_tape();
    clear_grad_storage();

    let w = vec![1.0, 2.0, 3.0, 4.0];
    let w_ag = Tensor::<AG, 1>::new(w.clone(), [4]);
    let x = Tensor::<AG, 1>::with_grad(data.clone(), [4]);
    let loss = (&x.tanh() * &w_ag).sum();
    loss.backward::<ReverseMode>();
    let analytic = x.grad().unwrap();

    let numeric = numeric_grad_1d(&data, [4], &|t| {
        let w_cpu = Tensor::<CPUBackend, 1>::new(w.clone(), [4]);
        (&t.tanh() * &w_cpu).sum()
    });

    assert_close(&analytic.as_slice(), &numeric);
}

#[test]
fn grad_check_gelu_1d() {
    let data = vec![-1.5, -0.5, 0.0, 0.5, 1.5];
    clear_tape();
    clear_grad_storage();

    let w = vec![1.0, 1.5, 2.0, 2.5, 3.0];
    let w_ag = Tensor::<AG, 1>::new(w.clone(), [5]);
    let x = Tensor::<AG, 1>::with_grad(data.clone(), [5]);
    let loss = (&x.gelu() * &w_ag).sum();
    loss.backward::<ReverseMode>();
    let analytic = x.grad().unwrap();

    let numeric = numeric_grad_1d(&data, [5], &|t| {
        let w_cpu = Tensor::<CPUBackend, 1>::new(w.clone(), [5]);
        (&t.gelu() * &w_cpu).sum()
    });

    assert_close(&analytic.as_slice(), &numeric);
}
