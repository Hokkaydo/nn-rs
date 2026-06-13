use crate::linalg::tensor::Scalar;

pub struct PlateauDetector {
    best_loss: Scalar,
    patience: usize,
    wait: usize,
    min_delta: Scalar,
}

impl PlateauDetector {
    pub(crate) fn new(patience: usize, min_delta: Scalar) -> Self {
        Self {
            best_loss: Scalar::INFINITY,
            patience,
            wait: 0,
            min_delta,
        }
    }

    pub fn has_plateaued(&mut self, current_loss: Scalar) -> bool {
        if current_loss < self.best_loss - self.min_delta {
            self.best_loss = current_loss;
            self.wait = 0;
        } else {
            self.wait += 1;
        }
        self.wait >= self.patience
    }
}
