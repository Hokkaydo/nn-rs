use crate::backend::autograd::engine::{AutogradStrategy, GradValue};
use crate::backend::autograd::{Autograd, GradNode, GradOp, store_grad};
use crate::backend::backend::Backend;
use crate::linalg::tensor::{Tensor, TensorId};
use std::collections::{HashMap, HashSet};

pub struct ReverseMode;

/// Build a topological order of nodes reachable from `root_id`.
fn build_topology(tape: &[GradNode], root_id: TensorId) -> Vec<GradNode> {
    let producers: HashMap<TensorId, &GradNode> =
        tape.iter().map(|n| (n.output_id, n)).collect();

    let mut visited = HashSet::new();
    let mut order = Vec::new();

    fn dfs(
        id: TensorId,
        producers: &HashMap<TensorId, &GradNode>,
        visited: &mut HashSet<TensorId>,
        order: &mut Vec<GradNode>,
    ) {
        if !visited.insert(id) {
            return;
        }
        if let Some(node) = producers.get(&id) {
            for &inp in &node.input_ids {
                dfs(inp, producers, visited, order);
            }
            order.push((*node).clone());
        }
    }

    dfs(root_id, &producers, &mut visited, &mut order);
    order
}

/// Accumulate `grad` into `grads[target_id]` using inner-backend addition.
fn accumulate<B: Backend>(
    target_id: TensorId,
    grad: GradValue<B>,
    grads: &mut HashMap<TensorId, GradValue<B>>,
) {
    grads
        .entry(target_id)
        .and_modify(|existing| {
            let new = existing.clone().add_into(grad.clone());
            *existing = new;
        })
        .or_insert(grad);
}

impl AutogradStrategy for ReverseMode {
    fn backward<B: Backend, const NDIM: usize>(
        tensor: &Tensor<Autograd<B>, NDIM>,
        tape: Vec<GradNode>,
        requires_grad: HashMap<TensorId, bool>,
    ) {
        let topo = build_topology(&tape, tensor.id);

        let mut grads: HashMap<TensorId, GradValue<B>> = HashMap::new();

        // Seed: ∂L/∂L = ones(output_shape)
        let seed = Tensor::<B, NDIM>::ones(tensor.shape());
        grads.insert(tensor.id, GradValue::from_id(seed.id, NDIM));

        for node in topo.into_iter().rev() {
            let go = match grads.get(&node.output_id) {
                Some(g) => g.clone(),
                None => continue,
            };

            let inp_ndim = node.inputs_ndims.first().copied().unwrap_or(node.output_ndim);

            match &node.grad_op {

                GradOp::Neg => {
                    accumulate(node.input_ids[0], go.neg(), &mut grads);
                }

                GradOp::Exp => {
                    // d/da = go * exp(a)
                    let out = GradValue::<B>::from_id(node.output_id, node.output_ndim);
                    accumulate(node.input_ids[0], go.mul(&out), &mut grads);
                }

                GradOp::Log => {
                    // d/da = go / a
                    let a = GradValue::<B>::from_id(node.input_ids[0], inp_ndim);
                    accumulate(node.input_ids[0], go.div(&a), &mut grads);
                }

                GradOp::Pow { exponent } => {
                    // d/da = go * exponent * a^(exponent-1)
                    let exp = *exponent;
                    let a = GradValue::<B>::from_id(node.input_ids[0], inp_ndim);
                    let ga = go.mul_scalar(exp).mul(&a.pow(exp - 1.0));
                    accumulate(node.input_ids[0], ga, &mut grads);
                }

                GradOp::Abs => {
                    // d/da = go * sign(a)
                    let a = GradValue::<B>::from_id(node.input_ids[0], inp_ndim);
                    accumulate(node.input_ids[0], go.mul(&a.sign()), &mut grads);
                }

                GradOp::Sign => {
                    // Not differentiable, zero gradient
                    let zeros = GradValue::<B>::zeros_like_id(node.input_ids[0], inp_ndim);
                    accumulate(node.input_ids[0], zeros, &mut grads);
                }

                GradOp::Clamp { min, max } => {
                    // ∂/∂a = go * (min < a < max ? 1 : 0)
                    let a = GradValue::<B>::from_id(node.input_ids[0], inp_ndim);
                    let mask = a.clamp_mask(*min, *max);
                    accumulate(node.input_ids[0], go.mul(&mask), &mut grads);
                }

                GradOp::Add => {
                    // d/da = go (summed to input shape if broadcast happened)
                    // d/db = go (same)
                    for (&inp_id, &ndim) in node.input_ids.iter().zip(node.inputs_ndims.iter()) {
                        let ga = go.clone().sum_to_shape_of(inp_id, ndim);
                        accumulate(inp_id, ga, &mut grads);
                    }
                }

                GradOp::Sub => {
                    let ga = go.clone().sum_to_shape_of(node.input_ids[0], node.inputs_ndims[0]);
                    let gb = go.sum_to_shape_of(node.input_ids[1], node.inputs_ndims[1]).neg();
                    accumulate(node.input_ids[0], ga, &mut grads);
                    accumulate(node.input_ids[1], gb, &mut grads);
                }

                GradOp::Mul => {
                    // d/da = go * b,  d/db = go * a
                    let a = GradValue::<B>::from_id(node.input_ids[0], node.inputs_ndims[0]);
                    let b = GradValue::<B>::from_id(node.input_ids[1], node.inputs_ndims[1]);
                    let ga = go.clone().mul(&b);
                    let gb = go.mul(&a);
                    accumulate(node.input_ids[0], ga, &mut grads);
                    accumulate(node.input_ids[1], gb, &mut grads);
                }

                GradOp::Div => {
                    // d/da = go / b,  d/db = -go * a / b*b
                    let a = GradValue::<B>::from_id(node.input_ids[0], node.inputs_ndims[0]);
                    let b = GradValue::<B>::from_id(node.input_ids[1], node.inputs_ndims[1]);
                    let ga = go.clone().div(&b);
                    let gb = go.neg().mul(&a).div(&b.clone().mul(&b));
                    accumulate(node.input_ids[0], ga, &mut grads);
                    accumulate(node.input_ids[1], gb, &mut grads);
                }

                GradOp::AddScalar { .. } | GradOp::SubScalar { .. } => {
                    accumulate(node.input_ids[0], go, &mut grads);
                }

                GradOp::ScalarSub { .. } => {
                    accumulate(node.input_ids[0], go.neg(), &mut grads);
                }

                GradOp::MulScalar { scalar } => {
                    accumulate(node.input_ids[0], go.mul_scalar(*scalar), &mut grads);
                }

                GradOp::DivScalar { scalar } => {
                    accumulate(node.input_ids[0], go.div_scalar(*scalar), &mut grads);
                }

                GradOp::ScalarDiv { scalar } => {
                    // scalar / a : d/da = -scalar * go / a*a
                    let s = *scalar;
                    let a = GradValue::<B>::from_id(node.input_ids[0], inp_ndim);
                    let ga = go.neg().mul_scalar(s).div(&a.clone().mul(&a));
                    accumulate(node.input_ids[0], ga, &mut grads);
                }

                GradOp::MatMul => {
                    match (node.inputs_ndims[0], node.inputs_ndims[1]) {
                        (2, 2) => {
                            // a:[M,K] b:[K,N] to out:[M,N]
                            // d/da = go @ b.T,  d/db = a.T @ go
                            let a: Tensor<B, 2> = Tensor::from_id(node.input_ids[0]);
                            let b: Tensor<B, 2> = Tensor::from_id(node.input_ids[1]);
                            let go2: Tensor<B, 2> = Tensor::from_id(go.id());
                            let b_t = B::transpose(&b, None);
                            let a_t = B::transpose(&a, None);
                            let ga = GradValue::Rank2(B::matmul_22(&go2, &b_t));
                            let gb = GradValue::Rank2(B::matmul_22(&a_t, &go2));
                            accumulate(node.input_ids[0], ga, &mut grads);
                            accumulate(node.input_ids[1], gb, &mut grads);
                        }
                        (2, 1) => {
                            // a:[M,N] b:[N] to out:[M]
                            // d/da = outer(go, b) = go.unsqueeze(1) @ b.unsqueeze(0)
                            // d/db = a.T @ go
                            let a: Tensor<B, 2> = Tensor::from_id(node.input_ids[0]);
                            let b: Tensor<B, 1> = Tensor::from_id(node.input_ids[1]);
                            let go1: Tensor<B, 1> = Tensor::from_id(go.id());
                            let go_col = B::unsqueeze(&go1, 1); // [M,1]
                            let b_row = B::unsqueeze(&b, 0);    // [1,N]
                            let ga = GradValue::Rank2(B::matmul_22(&go_col, &b_row));
                            let gb = GradValue::Rank1(B::matmul_21(&B::transpose(&a, None), &go1));
                            accumulate(node.input_ids[0], ga, &mut grads);
                            accumulate(node.input_ids[1], gb, &mut grads);
                        }
                        (1, 2) => {
                            // a:[M] b:[M,N] to out:[N]
                            // d/da = go @ b.T  (= B::matmul_12(go, b.T))
                            // d/db = outer(a, go) = a.unsqueeze(1) @ go.unsqueeze(0)
                            let a: Tensor<B, 1> = Tensor::from_id(node.input_ids[0]);
                            let b: Tensor<B, 2> = Tensor::from_id(node.input_ids[1]);
                            let go1: Tensor<B, 1> = Tensor::from_id(go.id());
                            let b_t = B::transpose(&b, None);
                            let ga = GradValue::Rank1(B::matmul_12(&go1, &b_t));
                            let a_col = B::unsqueeze(&a, 1);  // [M,1]
                            let go_row = B::unsqueeze(&go1, 0); // [1,N]
                            let gb = GradValue::Rank2(B::matmul_22(&a_col, &go_row));
                            accumulate(node.input_ids[0], ga, &mut grads);
                            accumulate(node.input_ids[1], gb, &mut grads);
                        }
                        (1, 1) => {
                            // dot product: a:[N] b:[N] to scalar
                            // d/da = go_scalar * b,  d/db = go_scalar * a
                            let a: Tensor<B, 1> = Tensor::from_id(node.input_ids[0]);
                            let b: Tensor<B, 1> = Tensor::from_id(node.input_ids[1]);
                            let scale = go.as_scalar();
                            let ga = GradValue::Rank1(B::mul_scalar(&b, scale));
                            let gb = GradValue::Rank1(B::mul_scalar(&a, scale));
                            accumulate(node.input_ids[0], ga, &mut grads);
                            accumulate(node.input_ids[1], gb, &mut grads);
                        }
                        (r0, r1) => panic!("MatMul backward: unsupported ranks ({r0}, {r1})"),
                    }
                }

                GradOp::Sum { .. } => {
                    // d/da = broadcast go back to input shape
                    let ga = go.broadcast_to_id(node.input_ids[0]);
                    accumulate(node.input_ids[0], ga, &mut grads);
                }

                GradOp::Mean { axes } => {
                    // d/da = go / count, broadcast to input shape
                    let input_shape = B::shape_dyn(node.input_ids[0]);
                    let count: usize = axes
                        .as_ref()
                        .map(|ax| ax.iter().map(|&a| input_shape[a]).product())
                        .unwrap_or_else(|| input_shape.iter().product());
                    let ga = go.div_scalar(count as f32).broadcast_to_id(node.input_ids[0]);
                    accumulate(node.input_ids[0], ga, &mut grads);
                }

                GradOp::Max { .. } | GradOp::Min { .. } => {
                    // d/da = go * mask where mask = (a == output broadcast to a.shape)
                    let a = GradValue::<B>::from_id(node.input_ids[0], inp_ndim);
                    let out_bcast = GradValue::<B>::from_id(node.output_id, node.output_ndim)
                        .broadcast_to_id(node.input_ids[0]);
                    let mask = a.eq_mask(out_bcast);
                    let go_bcast = go.broadcast_to_id(node.input_ids[0]);
                    accumulate(node.input_ids[0], go_bcast.mul(&mask), &mut grads);
                }


                GradOp::Reshape { .. } => {
                    // d/da = reshape go back to input shape
                    let ga = go.reshape_to_id(node.input_ids[0]);
                    accumulate(node.input_ids[0], ga, &mut grads);
                }

                GradOp::Transpose { axes } => {
                    // d/da = transpose go with inverse permutation
                    match go {
                        GradValue::Rank1(t) => {
                            accumulate(node.input_ids[0], GradValue::Rank1(t), &mut grads);
                        }
                        GradValue::Rank2(t) => {
                            let inv = axes.as_ref().map(|ax| {
                                let mut inv = [0usize; 2];
                                for (i, &j) in ax.iter().enumerate() {
                                    inv[j] = i;
                                }
                                inv
                            });
                            let transposed = B::transpose(&t, inv);
                            accumulate(node.input_ids[0], GradValue::Rank2(transposed), &mut grads);
                        }
                    }
                }

                GradOp::Squeeze { axis, .. } => {
                    // d/da = unsqueeze go at axis (restores the squeezed dim)
                    let ga = match go {
                        GradValue::Rank1(t) => GradValue::unsqueeze_r1_to_r2(t, *axis),
                        other => other,
                    };
                    accumulate(node.input_ids[0], ga, &mut grads);
                }

                GradOp::Unsqueeze { axis } => {
                    // d/da = squeeze go at axis
                    let ga = match go {
                        GradValue::Rank2(t) => GradValue::squeeze_r2_to_r1(t, *axis),
                        other => other,
                    };
                    accumulate(node.input_ids[0], ga, &mut grads);
                }

                GradOp::Broadcast { .. } => {
                    // d/da = sum go over broadcast dims
                    let ga = go.sum_to_shape_of(node.input_ids[0], inp_ndim);
                    accumulate(node.input_ids[0], ga, &mut grads);
                }


                GradOp::Slice { axis, start, len } => {
                    let zeros = GradValue::<B>::zeros_like_id(node.input_ids[0], inp_ndim);
                    let indices: Vec<usize> = (*start..*start + *len).collect();
                    let ga = zeros.scatter_add_at(go, *axis, &indices);
                    accumulate(node.input_ids[0], ga, &mut grads);
                }

                GradOp::Gather { axis, indices } => {
                    let zeros = GradValue::<B>::zeros_like_id(node.input_ids[0], inp_ndim);
                    let ga = zeros.scatter_add_at(go, *axis, indices);
                    accumulate(node.input_ids[0], ga, &mut grads);
                }

                GradOp::ScatterAdd { .. } => {
                    // d/dtarget = go (grad flows through unchanged)
                    // d/dsrc    = gather(go, axis, indices)
                    // For simplicity: pass gradient through to target only for now
                    accumulate(node.input_ids[0], go, &mut grads);
                }

                GradOp::Sigmoid => {
                    // d/da = go * output * (1 - output)
                    let out = GradValue::<B>::from_id(node.output_id, node.output_ndim);
                    let one_minus = out.clone().neg().add_scalar(1.0);
                    let ga = go.mul(&out).mul(&one_minus);
                    accumulate(node.input_ids[0], ga, &mut grads);
                }

                GradOp::ReLU => {
                    // d/da = go * (a > 0 ? 1 : 0)
                    let a = GradValue::<B>::from_id(node.input_ids[0], inp_ndim);
                    let mask = a.relu().sign();
                    accumulate(node.input_ids[0], go.mul(&mask), &mut grads);
                }

                GradOp::Softmax => {
                    // Jacobian-vector product: output * (go - sum(go * output, keepdim))
                    let out = GradValue::<B>::from_id(node.output_id, node.output_ndim);
                    let dot = go.clone().mul(&out).sum_axes(&[1]);
                    let dot_bcast = dot.broadcast_to_id(out.id());
                    let ga = out.clone().mul(&go.sub(&dot_bcast));
                    accumulate(node.input_ids[0], ga, &mut grads);
                }

                GradOp::LogSoftmax => {
                    // d/da = go - exp(output) * sum(go, keepdim)
                    let out = GradValue::<B>::from_id(node.output_id, node.output_ndim);
                    let softmax = out.exp();
                    let sum_go = go.clone().sum_axes(&[1]);
                    let sum_bcast = sum_go.broadcast_to_id(softmax.id());
                    let ga = go.sub(&softmax.mul(&sum_bcast));
                    accumulate(node.input_ids[0], ga, &mut grads);
                }
            }
        }

        // Write results into GRAD_STORAGE so tensor.grad() works.
        for (param_id, grad_val) in &grads {
            if requires_grad.get(param_id).copied().unwrap_or(false) {
                store_grad(*param_id, grad_val.id());
            }
        }
    }
}
