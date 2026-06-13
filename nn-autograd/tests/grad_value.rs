use nn_autograd::autograd::engine::GradValue;
use nn_core::cpu::CPUBackend;
use nn_core::linalg::tensor::Tensor;

type GV = GradValue<CPUBackend>;

fn r1(data: Vec<f32>, shape: [usize; 1]) -> GV {
    GV::Rank1(Tensor::<CPUBackend, 1>::new(data, shape))
}

fn r2(data: Vec<f32>, shape: [usize; 2]) -> GV {
    GV::Rank2(Tensor::<CPUBackend, 2>::new(data, shape))
}

fn slice1(v: &GV) -> Vec<f32> {
    match v {
        GV::Rank1(t) => t.as_slice(),
        GV::Rank2(t) => t.as_slice(),
        GV::Rank3(t) => t.as_slice(),
        GV::Rank4(t) => t.as_slice(),
    }
}

fn assert_close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < 1e-5, "mismatch at {i}: {x} vs {y}");
    }
}

// --- constructors / metadata ---

#[test]
fn from_id_rank1() {
    let t = Tensor::<CPUBackend, 1>::new(vec![1.0, 2.0], [2]);
    let gv = GV::from_id(t.id, 1);
    assert_eq!(gv.ndim(), 1);
    assert_eq!(gv.shape_vec(), vec![2]);
}

#[test]
fn from_id_rank2() {
    let t = Tensor::<CPUBackend, 2>::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    let gv = GV::from_id(t.id, 2);
    assert_eq!(gv.ndim(), 2);
    assert_eq!(gv.shape_vec(), vec![2, 2]);
}

#[test]
fn from_id_rank3() {
    let t = Tensor::<CPUBackend, 3>::new(vec![1.0; 8], [2, 2, 2]);
    let gv = GV::from_id(t.id, 3);
    assert_eq!(gv.ndim(), 3);
    assert_eq!(gv.shape_vec(), vec![2, 2, 2]);
}

#[test]
fn from_id_rank4() {
    let t = Tensor::<CPUBackend, 4>::new(vec![1.0; 16], [2, 2, 2, 2]);
    let gv = GV::from_id(t.id, 4);
    assert_eq!(gv.ndim(), 4);
    assert_eq!(gv.shape_vec(), vec![2, 2, 2, 2]);
}

#[test]
#[should_panic(expected = "unsupported rank 5")]
fn from_id_invalid_rank() {
    let t = Tensor::<CPUBackend, 1>::new(vec![1.0], [1]);
    GV::from_id(t.id, 5);
}

#[test]
fn ones_like_id_rank1() {
    let t = Tensor::<CPUBackend, 1>::new(vec![5.0, 6.0, 7.0], [3]);
    let ones = GV::ones_like_id(t.id, 1);
    assert_eq!(ones.ndim(), 1);
    assert_eq!(slice1(&ones), vec![1.0, 1.0, 1.0]);
}

#[test]
fn ones_like_id_rank2() {
    let t = Tensor::<CPUBackend, 2>::new(vec![9.0; 6], [2, 3]);
    let ones = GV::ones_like_id(t.id, 2);
    assert_eq!(ones.ndim(), 2);
    assert_eq!(slice1(&ones), vec![1.0; 6]);
}

#[test]
fn zeros_like_id_rank1() {
    let t = Tensor::<CPUBackend, 1>::new(vec![3.0, 4.0], [2]);
    let zeros = GV::zeros_like_id(t.id, 1);
    assert_eq!(slice1(&zeros), vec![0.0, 0.0]);
}

#[test]
fn zeros_like_id_rank2() {
    let t = Tensor::<CPUBackend, 2>::new(vec![1.0; 4], [2, 2]);
    let zeros = GV::zeros_like_id(t.id, 2);
    assert_eq!(slice1(&zeros), vec![0.0; 4]);
}

#[test]
fn as_scalar_rank1() {
    let gv = r1(vec![42.0], [1]);
    assert_eq!(gv.as_scalar(), 42.0);
}

#[test]
fn as_scalar_rank2() {
    let gv = r2(vec![7.0], [1, 1]);
    assert_eq!(gv.as_scalar(), 7.0);
}

// --- arithmetic ---

#[test]
fn add_into_r1() {
    let a = r1(vec![1.0, 2.0], [2]);
    let b = r1(vec![3.0, 4.0], [2]);
    let c = a.add_into(b);
    assert_eq!(slice1(&c), vec![4.0, 6.0]);
}

#[test]
fn add_into_r2() {
    let a = r2(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    let b = r2(vec![4.0, 3.0, 2.0, 1.0], [2, 2]);
    let c = a.add_into(b);
    assert_eq!(slice1(&c), vec![5.0, 5.0, 5.0, 5.0]);
}

#[test]
#[should_panic(expected = "rank mismatch")]
fn add_into_rank_mismatch() {
    let a = r1(vec![1.0], [1]);
    let b = r2(vec![1.0], [1, 1]);
    let _ = a.add_into(b);
}

#[test]
fn sub_r1() {
    let a = r1(vec![5.0, 3.0], [2]);
    let b = r1(vec![1.0, 2.0], [2]);
    let c = a.sub(&b);
    assert_eq!(slice1(&c), vec![4.0, 1.0]);
}

#[test]
fn mul_r1() {
    let a = r1(vec![2.0, 3.0], [2]);
    let b = r1(vec![4.0, 5.0], [2]);
    let c = a.mul(&b);
    assert_eq!(slice1(&c), vec![8.0, 15.0]);
}

#[test]
fn div_r1() {
    let a = r1(vec![6.0, 9.0], [2]);
    let b = r1(vec![2.0, 3.0], [2]);
    let c = a.div(&b);
    assert_eq!(slice1(&c), vec![3.0, 3.0]);
}

#[test]
fn neg_r1() {
    let a = r1(vec![1.0, -2.0, 3.0], [3]);
    let c = a.neg();
    assert_eq!(slice1(&c), vec![-1.0, 2.0, -3.0]);
}

#[test]
fn mul_scalar_r2() {
    let a = r2(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    let c = a.mul_scalar(3.0);
    assert_eq!(slice1(&c), vec![3.0, 6.0, 9.0, 12.0]);
}

#[test]
fn div_scalar_r1() {
    let a = r1(vec![4.0, 6.0], [2]);
    let c = a.div_scalar(2.0);
    assert_eq!(slice1(&c), vec![2.0, 3.0]);
}

#[test]
fn add_scalar_r1() {
    let a = r1(vec![1.0, 2.0], [2]);
    let c = a.add_scalar(10.0);
    assert_eq!(slice1(&c), vec![11.0, 12.0]);
}

// --- unary ops ---

#[test]
fn exp_r1() {
    let a = r1(vec![0.0, 1.0], [2]);
    let c = a.exp();
    assert_close(&slice1(&c), &[1.0, std::f32::consts::E]);
}

#[test]
fn log_r1() {
    let a = r1(vec![1.0, std::f32::consts::E], [2]);
    let c = a.log();
    assert_close(&slice1(&c), &[0.0, 1.0]);
}

#[test]
fn abs_r1() {
    let a = r1(vec![-3.0, 4.0], [2]);
    let c = a.abs();
    assert_eq!(slice1(&c), vec![3.0, 4.0]);
}

#[test]
fn sign_r1() {
    let a = r1(vec![-5.0, 0.0, 3.0], [3]);
    let c = a.sign();
    assert_eq!(slice1(&c), vec![-1.0, 0.0, 1.0]);
}

#[test]
fn relu_r1() {
    let a = r1(vec![-1.0, 0.0, 2.0], [3]);
    let c = a.relu();
    assert_eq!(slice1(&c), vec![0.0, 0.0, 2.0]);
}

#[test]
fn gelu_r1() {
    let data = vec![-1.0, 0.0, 1.0];
    let a = r1(data.clone(), [3]);
    let c = a.gelu();
    let expected: Vec<f32> = data
        .iter()
        .map(|&x| 0.5 * x * (1.0 + (x / 2.0f32.sqrt()).tanh()))
        .collect();
    assert_close(&slice1(&c), &expected);
}

#[test]
fn tanh_r1() {
    let data = vec![-1.0, 0.0, 1.0];
    let a = r1(data.clone(), [3]);
    let c = a.tanh();
    let expected: Vec<f32> = data.iter().map(|&x| x.tanh()).collect();
    assert_close(&slice1(&c), &expected);
}

#[test]
fn pow_r1() {
    let a = r1(vec![2.0, 3.0], [2]);
    let c = a.pow(3.0);
    assert_close(&slice1(&c), &[8.0, 27.0]);
}

// --- reduction ---

#[test]
fn sum_axes_r2() {
    // [2,3] sum over axis 0 keepdim → [1,3]
    let a = r2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let c = a.sum_axes(&[0]);
    assert_eq!(c.ndim(), 2);
    assert_eq!(c.shape_vec(), vec![1, 3]);
    assert_close(&slice1(&c), &[5.0, 7.0, 9.0]);
}

#[test]
fn sum_all_r1() {
    // sum(None) on rank-1 produces a scalar tensor of shape [1]
    let a = r1(vec![1.0, 2.0, 3.0], [3]);
    let c = a.sum_all();
    assert_eq!(c.ndim(), 1);
    assert_eq!(c.shape_vec(), vec![1]);
    assert_close(&slice1(&c), &[6.0]);
}

#[test]
fn sum_all_r2() {
    // sum(None) on rank-2 produces a scalar of shape [1,1]
    let a = r2(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    let c = a.sum_all();
    assert_eq!(c.ndim(), 2);
    assert_eq!(c.shape_vec(), vec![1, 1]);
    assert_close(&slice1(&c), &[10.0]);
}

// --- shape ---

#[test]
fn transpose_default_r1_is_identity() {
    let a = r1(vec![1.0, 2.0, 3.0], [3]);
    let original = slice1(&a);
    let c = a.transpose_default();
    assert_eq!(c.ndim(), 1);
    assert_eq!(slice1(&c), original);
}

#[test]
fn transpose_default_r2() {
    // Input [2,3]: row0=[1,2,3], row1=[4,5,6]
    // After transpose to [3,2]: col0=[1,4], col1=[2,5], col2=[3,6]
    let a = r2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let c = a.transpose_default();
    assert_eq!(c.ndim(), 2);
    assert_eq!(c.shape_vec(), vec![3, 2]);
    // Use element access (respects strides) rather than as_slice (raw buffer order)
    let t = match &c {
        GV::Rank2(t) => t,
        _ => panic!(),
    };
    assert_eq!(t.get([0, 0]), 1.0);
    assert_eq!(t.get([0, 1]), 4.0);
    assert_eq!(t.get([1, 0]), 2.0);
    assert_eq!(t.get([1, 1]), 5.0);
    assert_eq!(t.get([2, 0]), 3.0);
    assert_eq!(t.get([2, 1]), 6.0);
}

#[test]
fn unsqueeze_r1_to_r2_axis0() {
    let t = Tensor::<CPUBackend, 1>::new(vec![1.0, 2.0, 3.0], [3]);
    let gv = GV::unsqueeze_r1_to_r2(t, 0);
    assert_eq!(gv.ndim(), 2);
    assert_eq!(gv.shape_vec(), vec![1, 3]);
}

#[test]
fn squeeze_r2_to_r1() {
    let t = Tensor::<CPUBackend, 2>::new(vec![1.0, 2.0, 3.0], [1, 3]);
    let gv = GV::squeeze_r2_to_r1(t, 0);
    assert_eq!(gv.ndim(), 1);
    assert_eq!(gv.shape_vec(), vec![3]);
    assert_eq!(slice1(&gv), vec![1.0, 2.0, 3.0]);
}

// --- complex helpers ---

#[test]
fn eq_mask_matches() {
    let a = r1(vec![1.0, 2.0, 3.0], [3]);
    let b = r1(vec![1.0, 9.0, 3.0], [3]);
    let mask = a.eq_mask(b);
    assert_eq!(slice1(&mask), vec![1.0, 0.0, 1.0]);
}

#[test]
fn clamp_mask_interior() {
    // Values inside (min, max) get mask 1, values at or outside get 0
    let a = r1(vec![-2.0, 0.0, 1.0, 3.0, 5.0], [5]);
    let mask = a.clamp_mask(0.0, 3.0);
    // -2 < 0 → 0; 0 == min → 0; 1 interior → 1; 3 == max → 0; 5 > max → 0
    assert_eq!(slice1(&mask), vec![0.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn scatter_add_at_r1() {
    let target = r1(vec![0.0, 0.0, 0.0], [3]);
    let src = r1(vec![1.0, 2.0], [2]);
    let result = target.scatter_add_at(src, 0, &[0, 2]);
    assert_eq!(slice1(&result), vec![1.0, 0.0, 2.0]);
}

#[test]
fn gather_at_r2() {
    // [3,2] gather rows 0 and 2
    let a = r2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]);
    let result = a.gather_at(0, &[0, 2]);
    assert_eq!(result.ndim(), 2);
    assert_eq!(result.shape_vec(), vec![2, 2]);
    assert_eq!(slice1(&result), vec![1.0, 2.0, 5.0, 6.0]);
}

#[test]
fn broadcast_to_id_r2() {
    // broadcast [1,3] → [2,3]
    let src = r2(vec![1.0, 2.0, 3.0], [1, 3]);
    let target = Tensor::<CPUBackend, 2>::new(vec![0.0; 6], [2, 3]);
    let result = src.broadcast_to_id(target.id);
    assert_eq!(result.shape_vec(), vec![2, 3]);
    assert_close(&slice1(&result), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn sum_to_shape_of_reduces_axis() {
    // [2,3] sum to [1,3] target
    let src = r2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let target = Tensor::<CPUBackend, 2>::new(vec![0.0; 3], [1, 3]);
    let result = src.sum_to_shape_of(target.id, 2);
    assert_eq!(result.shape_vec(), vec![1, 3]);
    assert_close(&slice1(&result), &[5.0, 7.0, 9.0]);
}

#[test]
fn reshape_to_id_r2() {
    // [2,3] → [3,2]
    let src = r2(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let target = Tensor::<CPUBackend, 2>::new(vec![0.0; 6], [3, 2]);
    let result = src.reshape_to_id(target.id);
    assert_eq!(result.shape_vec(), vec![3, 2]);
    assert_eq!(slice1(&result), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}
