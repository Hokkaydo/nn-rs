use nn_autograd::autograd::Autograd;
use nn_la::backend::Backend;
use nn_la::linalg::tensor::Tensor;

pub struct ReLU<B: Backend> { 
    _marker: std::marker::PhantomData<B>,
}
pub struct GeLU<B: Backend> { 
    _marker: std::marker::PhantomData<B>,
}
pub struct Sigmoid<B: Backend> { 
    _marker: std::marker::PhantomData<B>,
}
pub struct Softmax<B: Backend> { 
    _marker: std::marker::PhantomData<B>,
}
pub struct LogSoftmax<B: Backend> { 
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> ReLU<B> {
    pub fn forward(&self, input: &Tensor<Autograd<B>, 2>) -> Tensor<Autograd<B>, 2> {
        input.relu()
    }

    pub fn new() -> Self {
        Self { _marker: std::marker::PhantomData }
    }
}

impl<B: Backend> GeLU<B> {
    pub fn forward(&self, input: &Tensor<Autograd<B>, 2>) -> Tensor<Autograd<B>, 2> {
        input.gelu()
    }

    pub fn new() -> Self {
        Self { _marker: std::marker::PhantomData }
    }
}

impl<B: Backend> Sigmoid<B> {
    pub fn forward(&self, input: &Tensor<Autograd<B>, 2>) -> Tensor<Autograd<B>, 2> {
        input.sigmoid()
    }

    pub fn new() -> Self {
        Self { _marker: std::marker::PhantomData }
    }
}

impl<B: Backend> Softmax<B> {
    pub fn forward(&self, input: &Tensor<Autograd<B>, 2>) -> Tensor<Autograd<B>, 2> {
        input.softmax()
    }

    pub fn new() -> Self {
        Self { _marker: std::marker::PhantomData }
    }
}

impl<B: Backend> LogSoftmax<B> {
    pub fn forward(&self, input: &Tensor<Autograd<B>, 2>) -> Tensor<Autograd<B>, 2> {
        input.log_softmax()
    }

    pub fn new() -> Self {
        Self { _marker: std::marker::PhantomData }
    }
}
