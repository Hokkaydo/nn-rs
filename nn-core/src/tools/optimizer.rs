use nn_autograd::autograd;
use nn_la::cpu::{data_by_id, with_mut_data_by_id};
use nn_la::linalg::tensor::{Scalar, TensorId};
use nn_autograd::autograd::clear_grad_storage;

pub trait Optimizer {
    fn step(&mut self, param_ids: &[TensorId]);
    fn zero_grad(&self, param_ids: &[TensorId]) {
        let _ = param_ids; // param-level zero_grad clears all for simplicity
        clear_grad_storage();
    }
}

pub struct SGD {
    lr: Scalar,
}

impl SGD {
    pub fn new(lr: Scalar) -> Self {
        Self { lr }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, param_ids: &[TensorId]) {
        let lr = self.lr;
        for &id in param_ids {
            if let Some(gid) = autograd::get_grad_id(id) {
                let grad = data_by_id(gid);
                with_mut_data_by_id(id, |param| {
                    for (p, g) in param.iter_mut().zip(grad.iter()) {
                        *p -= lr * g;
                    }
                });
            }
        }
    }
}

pub struct Adam {
    lr: Scalar,
    beta1: Scalar,
    beta2: Scalar,
    eps: Scalar,
    moments: Vec<Vec<Scalar>>,
    variances: Vec<Vec<Scalar>>,
    t: usize,
}

impl Adam {
    pub fn new(lr: Scalar, beta1: Scalar, beta2: Scalar, eps: Scalar) -> Self {
        Self { lr, beta1, beta2, eps, moments: Vec::new(), variances: Vec::new(), t: 0 }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, param_ids: &[TensorId]) {
        self.t += 1;
        let t = self.t;
        let (lr, b1, b2, eps) = (self.lr, self.beta1, self.beta2, self.eps);

        if self.moments.len() < param_ids.len() {
            for &id in &param_ids[self.moments.len()..] {
                let n = data_by_id(id).len();
                self.moments.push(vec![0.0; n]);
                self.variances.push(vec![0.0; n]);
            }
        }

        for (i, &id) in param_ids.iter().enumerate() {
            if let Some(gid) = autograd::get_grad_id(id) {
                let grad = data_by_id(gid);
                let m = &mut self.moments[i];
                let v = &mut self.variances[i];
                for (j, &g) in grad.iter().enumerate() {
                    m[j] = b1 * m[j] + (1.0 - b1) * g;
                    v[j] = b2 * v[j] + (1.0 - b2) * g * g;
                }
                let m_hat: Vec<Scalar> = m.iter().map(|&x| x / (1.0 - b1.powi(t as i32))).collect();
                let v_hat: Vec<Scalar> = v.iter().map(|&x| x / (1.0 - b2.powi(t as i32))).collect();
                with_mut_data_by_id(id, |param| {
                    for (j, p) in param.iter_mut().enumerate() {
                        *p -= lr * m_hat[j] / (v_hat[j].sqrt() + eps);
                    }
                });
            }
        }
    }
}
