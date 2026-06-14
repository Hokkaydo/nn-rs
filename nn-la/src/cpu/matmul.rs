use std::simd::{Simd, StdFloat};

use rayon::prelude::*;

use crate::backend::MatMulOps;
use crate::cpu::CPUBackend;
use crate::linalg::tensor::Tensor;

// Micro-kernel tile (fit in registers): MR rows x NR cols.
// NR=32 : two AVX-512 registers (two Simd<f32,16>), doubles throughput vs AVX2.
// MR=6  : 12 accumulator registers
const MR: usize = 6;
const NR: usize = 32;
const NR_HALF: usize = NR / 2;

// Cache-blocking panel sizes (MC*KC*4 = 64KB, KC*NC*4 = 1MB):
//   A-panel (MC x KC) fits in L1 (320KB),
//   B-panel (KC x NC) fits in L2 (5MB).
//   C micro-tile (MR×NR) stays in registers.
const MC: usize = 64;
const KC: usize = 256;
const NC: usize = 1024;

// Minimum matrix area (m*n) below which we skip Rayon thread overhead.
// Need at least 4 ic-bands (one per physical core) to amortise thread wake-up,
// so threshold = 4 * MC rows × some N = 256 * 256.
const PARALLEL_THRESHOLD: usize = 256 * 256;

// Thread-local pack buffers: reused across calls, eliminating per-call heap allocation.
thread_local! {
    static A_PACK: std::cell::UnsafeCell<Vec<f32>> = std::cell::UnsafeCell::new(vec![0.0f32; MC.div_ceil(MR) * MR * KC]);
    static B_PACK: std::cell::UnsafeCell<Vec<f32>> = std::cell::UnsafeCell::new(vec![0.0f32; KC * NC.div_ceil(NR) * NR]);
}

// Pack rows [row_start, row_start+rows) x cols [col_start, col_start+cols) of
// `src` into `dst` as column-major panels of width MR.
//
// Layout: for panel p (MR rows), depth d (KC cols):
//   dst[p * cols * MR + d * MR + r] = src[row_start + p*MR + r][col_start + d]
//
// Edge panels (rows % MR != 0) are zero-padded to MR so the micro-kernel can
// always read a full MR slice without a bounds check.
#[inline(always)]
fn pack_a(
    src: &[f32],
    dst: &mut [f32],
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    rs: usize,
    cs: usize,
    off: usize,
) {
    let mut pos = 0;
    // for each panel in the A tile, last panel may have fewer than MR rows
    for panel in 0..rows.div_ceil(MR) {
        let r0 = row_start + panel * MR;
        let real_rows = MR.min(rows - panel * MR);
        for d in 0..cols {
            for r in 0..real_rows {
                dst[pos] = src[off + (r0 + r) * rs + (col_start + d) * cs];
                pos += 1;
            }
            // zero-pad the last panel if real_rows < MR
            for _ in real_rows..MR {
                dst[pos] = 0.0;
                pos += 1;
            }
        }
    }
}

// Pack rows [row_start, row_start+rows) x cols [col_start, col_start+cols) of
// `src` into `dst` as row-major panels of width NR.
//
// Layout: for panel q (NR cols), depth d (KC rows):
//   dst[q * rows * NR + d * NR + n] = src[row_start + d][col_start + q*NR + n]
//
// Edge panels (cols % NR != 0) are zero-padded to NR.
#[inline(always)]
fn pack_b(
    src: &[f32],
    dst: &mut [f32],
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    rs: usize,
    cs: usize,
    off: usize,
) {
    // for each panel in the B tile, last panel may have fewer than NR cols
    for panel in 0..cols.div_ceil(NR) {
        let c0 = col_start + panel * NR;
        let base = panel * rows * NR;
        let real_cols = NR.min(cols - panel * NR);
        for d in 0..rows {
            for n in 0..real_cols {
                dst[base + d * NR + n] = src[off + (row_start + d) * rs + (c0 + n) * cs];
            }
            // zero-pad the last panel if real_cols < NR
            for n in real_cols..NR {
                dst[base + d * NR + n] = 0.0;
            }
        }
    }
}

// Micro-kernel: C[MR_ACTUAL x NR] += A_packed[MR_ACTUAL x kc] * B_packed[kc x NR]
//
// MR_ACTUAL is a compile-time constant, so `for r in 0..MR_ACTUAL` is fully
// unrolled: exactly 2*MR_ACTUAL vfmadd231ps per kc step, no branches.
//
// V::from_slice calls replaced with unsafe pointer loads to eliminate the bounds check (a `je` before every FMA block) that std::simd emits for safe slices.
// Safety: pack_a/pack_b guarantee the packed buffers are padded to MR/NR
// multiples, and c is indexed within [0, m*n).
#[inline(always)]
fn micro_kernel<const MR_ACTUAL: usize>(
    a_pack: &[f32],
    b_pack: &[f32],
    c: &mut [f32],
    kc: usize,
    c_row: usize,
    c_col: usize,
    n: usize,
) {
    type V = Simd<f32, NR_HALF>;

    // AVX2 has 16 registers, allowing 12 accumulator registers + 2 for the B panel + 2 for the A panel
    let mut acc = [[V::splat(0.0f32); 2]; MR_ACTUAL];
    // load C micro-tile into accumulators
    for r in 0..MR_ACTUAL {
        let base = (c_row + r) * n + c_col;
        // Safety: base + NR ≤ (c_row + MR_ACTUAL) * n + c_col + NR ≤ m*n
        unsafe {
            acc[r][0] = V::from_array(*c.as_ptr().add(base).cast());
            acc[r][1] = V::from_array(*c.as_ptr().add(base + NR_HALF).cast());
        }
    }

    // iterate over the kc dimension of the A and B panels, accumulating into the C micro-tile
    for d in 0..kc {
        // Safety: b_pack is padded to NR multiples, d < kc <= KC
        let b0 = unsafe { V::from_array(*b_pack.as_ptr().add(d * NR).cast()) };
        let b1 = unsafe { V::from_array(*b_pack.as_ptr().add(d * NR + NR_HALF).cast()) };
        for r in 0..MR_ACTUAL {
            // Safety: a_pack is padded to MR multiples, d < kc <= KC
            let a = V::splat(unsafe { *a_pack.as_ptr().add(d * MR + r) });
            acc[r][0] = a.mul_add(b0, acc[r][0]);
            acc[r][1] = a.mul_add(b1, acc[r][1]);
        }
    }

    // store the C micro-tile back to memory
    for r in 0..MR_ACTUAL {
        let base = (c_row + r) * n + c_col;
        unsafe {
            *c.as_mut_ptr().add(base).cast() = acc[r][0].to_array();
            *c.as_mut_ptr().add(base + NR_HALF).cast() = acc[r][1].to_array();
        }
    }
}

// Edge micro-kernel for the rightmost tile when nc_actual < NR.
#[inline(always)]
fn micro_kernel_edge<const MR_ACTUAL: usize>(
    a_pack: &[f32],
    b_pack: &[f32],
    c: &mut [f32],
    kc: usize,
    nr_actual: usize,
    c_row: usize,
    c_col: usize,
    n: usize,
) {
    for d in 0..kc {
        for r in 0..MR_ACTUAL {
            let a = a_pack[d * MR + r];
            for col in 0..nr_actual {
                c[(c_row + r) * n + c_col + col] += a * b_pack[d * NR + col];
            }
        }
    }
}

// Dispatch to the correct monomorphized micro_kernel based on runtime mr value.
#[inline(always)]
fn dispatch_micro_kernel(
    a_pack: &[f32],
    b_pack: &[f32],
    c: &mut [f32],
    kc: usize,
    mr: usize,
    nr: usize,
    c_row: usize,
    c_col: usize,
    n: usize,
) {
    macro_rules! mk {
        ($mr:literal) => {
            if nr == NR {
                micro_kernel::<$mr>(a_pack, b_pack, c, kc, c_row, c_col, n)
            } else {
                micro_kernel_edge::<$mr>(a_pack, b_pack, c, kc, nr, c_row, c_col, n)
            }
        };
    }
    match mr {
        1 => mk!(1),
        2 => mk!(2),
        3 => mk!(3),
        4 => mk!(4),
        5 => mk!(5),
        _ => mk!(6),
    }
}

// Inner loop body shared by single-threaded and parallel paths.
// Processes one ic-band: rows [ic, ic+mc) of C, all jc/pc tiles.
// `c_band` is the slice c[ic*n .. (ic+mc)*n], i.e. the rows of C corresponding to this ic-band.
#[inline(always)]
fn compute_ic_band(
    a: &[f32],
    b: &[f32],
    c_band: &mut [f32],
    ic: usize,
    mc: usize,
    n: usize,
    k: usize,
    a_rs: usize,
    a_cs: usize,
    a_off: usize,
    b_rs: usize,
    b_cs: usize,
    b_off: usize,
    a_pack: &mut [f32],
    b_pack: &mut [f32],
) {
    // number of A micro-panels in this ic-band (ceil(mc/MR))
    let mr_panels = mc.div_ceil(MR);

    // for each B panel (of size NC)
    for jc in (0..n).step_by(NC) {
        // panel's number of cols
        let nc = NC.min(n - jc);
        // number of B micro-panels in this panel (ceil(nc/NR))
        let nr_panels = nc.div_ceil(NR);

        // for each k-panel (of size KC)
        for pc in (0..k).step_by(KC) {
            // panel's number of depth rows
            let kc = KC.min(k - pc);

            // pack B panel (kc x nc)
            pack_b(b, b_pack, pc, jc, kc, nc, b_rs, b_cs, b_off);
            // pack A panel (mc x kc)
            pack_a(a, a_pack, ic, pc, mc, kc, a_rs, a_cs, a_off);

            // for each C micro-tile (MR rows x NR cols) in the AxB panel product
            for ir in 0..mr_panels {
                // micro-tile's number of rows (MR for all but the last panel, which may be smaller)
                let mr = MR.min(mc - ir * MR);
                // A micro-panel (mr x kc) starts at a_pack[ir * kc * MR]
                let a_panel = &a_pack[ir * kc * MR..];

                for jr in 0..nr_panels {
                    // micro-tile's number of cols (NR for all but the last panel, which may be smaller)
                    let nr = NR.min(nc - jr * NR);
                    // B micro-panel (kc x nr) starts at b_pack[jr * kc * NR]
                    let b_panel = &b_pack[jr * kc * NR..];
                    // C micro-tile (mr x nr) starts at c_band[ir*MR*n + jc+jr*NR]
                    dispatch_micro_kernel(
                        a_panel, b_panel, c_band,
                        kc, mr, nr,
                        ir * MR, jc + jr * NR, n,
                    );
                }
            }
        }
    }
}

fn matmul_22_impl(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    a_rs: usize,
    a_cs: usize,
    a_off: usize,
    b_rs: usize,
    b_cs: usize,
    b_off: usize,
) {
    // ic-bands of C are disjoint row-ranges of a row-major matrix = contiguous
    // slices. Rayon can own each slice independently with no sharing.
    let ic_bands: Vec<usize> = (0..m).step_by(MC).collect();

    if m * n < PARALLEL_THRESHOLD {
        // Single-threaded path: use thread-local buffers (zero allocation).
        A_PACK.with(|ap| B_PACK.with(|bp| {
            let a_pack = unsafe { &mut *ap.get() };
            let b_pack = unsafe { &mut *bp.get() };
            for &ic in &ic_bands {
                let mc = MC.min(m - ic);
                let c_band = &mut c[ic * n..(ic + mc) * n];
                compute_ic_band(
                    a, b, c_band, ic, mc, n, k,
                    a_rs, a_cs, a_off, b_rs, b_cs, b_off,
                    a_pack, b_pack,
                );
            }
        }));
    } else {
        // Parallel path: split C into ic-bands (disjoint row slices) and process each band on its own thread. Each thread uses its own thread-local pack buffers (avoiding allocation)
        // Safety: the pointer cast to *mut f32 is ok because each band covers a distinct, non-overlapping range of `c`.
        let c_ptr = c.as_mut_ptr() as usize; // usize for Send across thread boundary

        ic_bands.into_par_iter().for_each(|ic| {
            let mc = MC.min(m - ic);
            let c_band = unsafe {
                std::slice::from_raw_parts_mut(
                    (c_ptr as *mut f32).add(ic * n),
                    mc * n,
                )
            };
            A_PACK.with(|ap| B_PACK.with(|bp| {
                let a_pack = unsafe { &mut *ap.get() };
                let b_pack = unsafe { &mut *bp.get() };
                compute_ic_band(
                    a, b, c_band, ic, mc, n, k,
                    a_rs, a_cs, a_off, b_rs, b_cs, b_off,
                    a_pack, b_pack,
                );
            }));
        });
    }
}

impl MatMulOps<Self> for CPUBackend {
    fn matmul_11(a: &Tensor<Self, 1>, b: &Tensor<Self, 1>) -> Tensor<Self, 1> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert_eq!(
            a_shape[0], b_shape[0],
            "matmul_11 shape mismatch: {:?} vs {:?}",
            a_shape, b_shape
        );
        let a_s = a.strides();
        let b_s = b.strides();
        let a_off = a.offset();
        let b_off = b.offset();
        let mut dot = 0.0f32;
        a.with_data(|a_data| {
            b.with_data(|b_data| {
                for i in 0..a_shape[0] {
                    dot += a_data[a_off + i * a_s[0]] * b_data[b_off + i * b_s[0]];
                }
            });
        });
        Tensor::new(vec![dot], [1])
    }

    fn matmul_12(a: &Tensor<Self, 1>, b: &Tensor<Self, 2>) -> Tensor<Self, 1> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert_eq!(
            a_shape[0], b_shape[0],
            "matmul_12 shape mismatch: {:?} vs {:?}",
            a_shape, b_shape
        );
        let a_s = a.strides();
        let b_s = b.strides();
        let a_off = a.offset();
        let b_off = b.offset();
        let mut result = vec![0.0f32; b_shape[1]];
        a.with_data(|a_data| {
            b.with_data(|b_data| {
                for j in 0..b_shape[1] {
                    for i in 0..a_shape[0] {
                        result[j] +=
                            a_data[a_off + i * a_s[0]] * b_data[b_off + i * b_s[0] + j * b_s[1]];
                    }
                }
            });
        });
        Tensor::new(result, [b_shape[1]])
    }

    fn matmul_21(a: &Tensor<Self, 2>, b: &Tensor<Self, 1>) -> Tensor<Self, 1> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert_eq!(
            a_shape[1], b_shape[0],
            "matmul_21 shape mismatch: {:?} vs {:?}",
            a_shape, b_shape
        );
        let a_s = a.strides();
        let b_s = b.strides();
        let a_off = a.offset();
        let b_off = b.offset();
        let mut result = vec![0.0f32; a_shape[0]];
        a.with_data(|a_data| {
            b.with_data(|b_data| {
                for i in 0..a_shape[0] {
                    for j in 0..a_shape[1] {
                        result[i] +=
                            a_data[a_off + i * a_s[0] + j * a_s[1]] * b_data[b_off + j * b_s[0]];
                    }
                }
            });
        });
        Tensor::new(result, [a_shape[0]])
    }

    fn matmul_22(a: &Tensor<Self, 2>, b: &Tensor<Self, 2>) -> Tensor<Self, 2> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        assert_eq!(
            a_shape[1], b_shape[0],
            "matmul_22 shape mismatch: {:?} vs {:?}",
            a_shape, b_shape
        );
        let a_s = a.strides();
        let b_s = b.strides();
        let a_off = a.offset();
        let b_off = b.offset();
        let (m, n, k) = (a_shape[0], b_shape[1], a_shape[1]);
        let mut c = vec![0.0f32; m * n];

        a.with_data(|a_data| {
            b.with_data(|b_data| {
                matmul_22_impl(
                    a_data, b_data, &mut c,
                    m, n, k,
                    a_s[0], a_s[1], a_off,
                    b_s[0], b_s[1], b_off,
                );
            });
        });

        Tensor::new(c, [m, n])
    }

    fn matmul_33(a: &Tensor<Self, 3>, b: &Tensor<Self, 3>) -> Tensor<Self, 3> {
        let [batch, m, k] = a.shape();
        let [b_batch, b_k, n] = b.shape();
        assert_eq!(batch, b_batch, "matmul_33 batch mismatch: {} vs {}", batch, b_batch);
        assert_eq!(k, b_k, "matmul_33 inner dim mismatch: {} vs {}", k, b_k);

        let a_s = a.strides();  // [batch_stride, row_stride, col_stride]
        let b_s = b.strides();
        let a_off = a.offset();
        let b_off = b.offset();
        let mut c = vec![0.0f32; batch * m * n];

        a.with_data(|a_data| {
            b.with_data(|b_data| {
                for i in 0..batch {
                    // Slice into c for this batch element: c[i*m*n .. (i+1)*m*n]
                    let c_slice = &mut c[i * m * n..(i + 1) * m * n];
                    matmul_22_impl(
                        a_data, b_data, c_slice,
                        m, n, k,
                        // row/col strides within the slice are the last two strides of a/b
                        a_s[1], a_s[2], a_off + i * a_s[0],
                        b_s[1], b_s[2], b_off + i * b_s[0],
                    );
                }
            });
        });

        Tensor::new(c, [batch, m, n])
    }
}
