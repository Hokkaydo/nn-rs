use nn_autograd::AutogradTensor;
use nn_autograd::autograd::engine::ReverseMode;
use nn_autograd::autograd::{Autograd, clear_grad_storage, clear_tape};
use nn_core::cpu::CPUBackend;
use nn_core::linalg::tensor::Tensor;

type AG = Autograd<CPUBackend>;
type Scalar = f32;

const EPS: Scalar = 1e-3;
const TOL: Scalar = 2e-2;

/// Central finite-difference gradient of a scalar loss w.r.t. a 2D matrix input,
/// holding everything else (the closure captures) fixed.
fn numeric_grad_matrix<F>(data: &[Scalar], rows: usize, cols: usize, f: &F) -> Vec<Scalar>
where
    F: Fn(&[Scalar]) -> Scalar,
{
    let mut grad = vec![0.0; data.len()];
    let mut buf = data.to_vec();
    for i in 0..data.len() {
        let orig = buf[i];
        buf[i] = orig + EPS;
        let plus = f(&buf);
        buf[i] = orig - EPS;
        let minus = f(&buf);
        buf[i] = orig;
        grad[i] = (plus - minus) / (2.0 * EPS);
    }
    let _ = (rows, cols);
    grad
}

fn assert_close(analytic: &[Scalar], numeric: &[Scalar]) {
    assert_eq!(analytic.len(), numeric.len());
    for (i, (a, n)) in analytic.iter().zip(numeric.iter()).enumerate() {
        assert!(
            (a - n).abs() <= TOL,
            "grad mismatch at {i}: analytic={a}, numeric={n}"
        );
    }
}

#[test]
fn grad_check_matmul_21() {
    // a:[M,N] @ b:[N] -> [M]; weighted-sum to a scalar so each output row differs.
    let m = 3;
    let n = 2;
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![0.5, -1.0];
    let w_data = vec![1.0, 2.0, 3.0]; // per-output-row weights, length M

    clear_tape();
    clear_grad_storage();
    let a = Tensor::<AG, 2>::with_grad(a_data.clone(), [m, n]);
    let b = Tensor::<AG, 1>::new(b_data.clone(), [n]);
    let w = Tensor::<AG, 1>::new(w_data.clone(), [m]);
    let loss = (&a.matmul(&b) * &w).sum();
    loss.backward::<ReverseMode>();
    let analytic = a.grad().unwrap();

    let numeric = numeric_grad_matrix(&a_data, m, n, &|buf| {
        let a = Tensor::<CPUBackend, 2>::new(buf.to_vec(), [m, n]);
        let b = Tensor::<CPUBackend, 1>::new(b_data.clone(), [n]);
        let w = Tensor::<CPUBackend, 1>::new(w_data.clone(), [m]);
        (&a.matmul(&b) * &w).sum().as_scalar()
    });

    assert_eq!(analytic.shape(), [m, n]);
    assert_close(&analytic.as_slice(), &numeric);
}

#[test]
fn grad_check_matmul_12() {
    // a:[M] @ b:[M,N] -> [N]; gradient checked w.r.t. b.
    let m = 3;
    let n = 2;
    let a_data = vec![0.5, -1.0, 2.0];
    let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w_data = vec![1.0, 2.0]; // per-output weights, length N

    clear_tape();
    clear_grad_storage();
    let a = Tensor::<AG, 1>::new(a_data.clone(), [m]);
    let b = Tensor::<AG, 2>::with_grad(b_data.clone(), [m, n]);
    let w = Tensor::<AG, 1>::new(w_data.clone(), [n]);
    let loss = (&a.matmul(&b) * &w).sum();
    loss.backward::<ReverseMode>();
    let analytic = b.grad().unwrap();

    let numeric = numeric_grad_matrix(&b_data, m, n, &|buf| {
        let a = Tensor::<CPUBackend, 1>::new(a_data.clone(), [m]);
        let b = Tensor::<CPUBackend, 2>::new(buf.to_vec(), [m, n]);
        let w = Tensor::<CPUBackend, 1>::new(w_data.clone(), [n]);
        (&a.matmul(&b) * &w).sum().as_scalar()
    });

    assert_eq!(analytic.shape(), [m, n]);
    assert_close(&analytic.as_slice(), &numeric);
}

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
