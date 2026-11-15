//! SIMD-tiled coupled-form modal IR synthesis (`1x` full-rate high modes plus
//! `4x` bandlimited sub-rate low modes with 16-tap Blackman-Nuttall polyphase
//! upsampling).

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use piano_simd::F32x4;
use piano_spec::{BodyPreset, PRESET_RADIATOR_ZONES};

use crate::mic_perspectives::ZonePathPair;
use crate::modal_prepare::{prepare_modes_split, PreparedMode};
use crate::subrate_upsample::{upsample_4x_add, DECIM, PAD_LEFT, PAD_RIGHT};

pub(crate) const NUM_IR_PATHS: usize = PRESET_RADIATOR_ZONES * 2;
pub(crate) const TILE: usize = 256;
pub(crate) const TILE_VECS: usize = TILE / 4;
pub(crate) const BATCH: usize = 16;

#[inline]
pub(crate) fn path_index(zone: usize, ch: usize) -> usize {
    zone * 2 + ch
}

pub(crate) struct ModalWorkspace {
    pub(crate) low_modes: Vec<PreparedMode>,
    pub(crate) high_modes: Vec<PreparedMode>,
    pub(crate) sub_flat: Vec<f32>,
    pub(crate) tile_ir: Box<[[F32x4; TILE_VECS]; NUM_IR_PATHS]>,
    pub(crate) q_buf: Box<[[F32x4; TILE_VECS]; BATCH]>,
}

impl ModalWorkspace {
    pub(crate) fn new(ir_len: usize) -> Self {
        let sub_padded = ir_len.div_ceil(DECIM) + PAD_LEFT + PAD_RIGHT;
        Self {
            low_modes: Vec::new(),
            high_modes: Vec::new(),
            sub_flat: alloc::vec![0.0_f32; NUM_IR_PATHS * sub_padded],
            tile_ir: Box::new([[F32x4::zero(); TILE_VECS]; NUM_IR_PATHS]),
            q_buf: Box::new([[F32x4::zero(); TILE_VECS]; BATCH]),
        }
    }

    pub(crate) fn ensure_sub_len(&mut self, sub_padded: usize) {
        let total = NUM_IR_PATHS * sub_padded;
        if self.sub_flat.len() != total {
            self.sub_flat.resize(total, 0.0);
        } else {
            self.sub_flat.fill(0.0);
        }
    }
}

pub(crate) fn synthesize_all_modes(
    preset: &BodyPreset,
    paths: &[ZonePathPair; PRESET_RADIATOR_ZONES],
    sr: f64,
    ir_len: usize,
    ws: &mut ModalWorkspace,
    ir_flat: &mut [f32],
) {
    let sub_len = ir_len.div_ceil(DECIM);
    let sub_padded = sub_len + PAD_LEFT + PAD_RIGHT;
    ws.ensure_sub_len(sub_padded);
    prepare_modes_split(
        preset,
        paths,
        sr,
        ir_len,
        sub_padded,
        &mut ws.low_modes,
        &mut ws.high_modes,
    );

    if !ws.high_modes.is_empty() {
        synthesize_modal_tiles(
            ir_len,
            &mut ws.high_modes,
            &mut ws.tile_ir,
            &mut ws.q_buf,
            ir_flat,
        );
    }

    if !ws.low_modes.is_empty() {
        synthesize_modal_tiles(
            sub_padded,
            &mut ws.low_modes,
            &mut ws.tile_ir,
            &mut ws.q_buf,
            &mut ws.sub_flat,
        );
        upsample_4x_add(&ws.sub_flat, sub_padded, ir_len, ir_flat);
    }
}

fn synthesize_modal_tiles(
    len: usize,
    modes: &mut [PreparedMode],
    tile_ir: &mut [[F32x4; TILE_VECS]; NUM_IR_PATHS],
    q_buf: &mut [[F32x4; TILE_VECS]; BATCH],
    dst_flat: &mut [f32],
) {
    let mut t0 = 0;
    while t0 < len {
        let tile_len = (len - t0).min(TILE);
        let tile_vecs = tile_len.div_ceil(4);
        let mut p = 0;
        while p < NUM_IR_PATHS {
            tile_ir[p][..tile_vecs].fill(F32x4::zero());
            p += 1;
        }

        let mut m_start = 0;
        while m_start < modes.len() {
            if modes[m_start].n_end <= t0 {
                break;
            }
            let batch_end = (m_start + BATCH).min(modes.len());
            let mut active = 0;
            let mut max_span = 0;
            let mut idx = m_start;
            while idx < batch_end {
                if modes[idx].n_end <= t0 {
                    break;
                }
                let span = (modes[idx].n_end - t0).min(tile_len);
                if span > max_span {
                    max_span = span;
                }
                let mode = &mut modes[idx];
                let q = &mut q_buf[active];
                let c0 = mode.c;
                let s0 = mode.s;
                let c0_v = F32x4::splat(c0 as f32);
                let s0_v = F32x4::splat(s0 as f32);
                let mut c_vec = mode.seed_cos * c0_v - mode.seed_sin * s0_v;
                let mut s_vec = mode.seed_cos * s0_v + mode.seed_sin * c0_v;
                let rc4_v = F32x4::splat(mode.rc4);
                let rs4_v = F32x4::splat(mode.rs4);

                let full_vecs = span >> 2;
                let rem = span & 3;
                let mut v = 0;
                if full_vecs > 0 {
                    q[0] = s_vec;
                    v = 1;
                    while v < full_vecs {
                        let c_next = c_vec * rc4_v - s_vec * rs4_v;
                        s_vec = c_vec * rs4_v + s_vec * rc4_v;
                        c_vec = c_next;
                        q[v] = s_vec;
                        v += 1;
                    }
                    if rem > 0 {
                        s_vec = c_vec * rs4_v + s_vec * rc4_v;
                    }
                }
                if rem > 0 {
                    let mut arr = s_vec.to_array();
                    let mut r = rem;
                    while r < 4 {
                        arr[r] = 0.0;
                        r += 1;
                    }
                    q[v] = F32x4::from_array(arr);
                    v += 1;
                }
                if v < tile_vecs {
                    q[v..tile_vecs].fill(F32x4::zero());
                }
                if t0 + TILE < mode.n_end {
                    mode.c = mode.rc_tile * c0 - mode.rs_tile * s0;
                    mode.s = mode.rs_tile * c0 + mode.rc_tile * s0;
                }
                active += 1;
                idx += 1;
            }

            let vecs = max_span.div_ceil(4).min(TILE_VECS);
            accumulate_batch(
                &modes[m_start..m_start + active],
                q_buf,
                active,
                vecs,
                tile_ir,
            );
            m_start += active;
        }

        let full_vecs = tile_len >> 2;
        let rem = tile_len & 3;
        let mut path = 0;
        while path < NUM_IR_PATHS {
            let dst_start = path * len + t0;
            let dst = &mut dst_flat[dst_start..dst_start + tile_len];
            let src = &tile_ir[path];
            let mut v = 0;
            while v < full_vecs {
                src[v].write_to_slice(&mut dst[v * 4..v * 4 + 4]);
                v += 1;
            }
            if rem > 0 {
                let arr = src[v].to_array();
                dst[full_vecs * 4..tile_len].copy_from_slice(&arr[..rem]);
            }
            path += 1;
        }

        t0 += TILE;
    }
}

#[inline]
fn accumulate_batch(
    batch_modes: &[PreparedMode],
    q_buf: &[[F32x4; TILE_VECS]; BATCH],
    count: usize,
    vecs: usize,
    tile_ir: &mut [[F32x4; TILE_VECS]; NUM_IR_PATHS],
) {
    let mut base = 0;
    while base < NUM_IR_PATHS {
        let mut w0 = [F32x4::zero(); BATCH];
        let mut w1 = [F32x4::zero(); BATCH];
        let mut w2 = [F32x4::zero(); BATCH];
        let mut k = 0;
        while k < count {
            let mw = &batch_modes[k].weights;
            w0[k] = F32x4::splat(mw[base]);
            w1[k] = F32x4::splat(mw[base + 1]);
            w2[k] = F32x4::splat(mw[base + 2]);
            k += 1;
        }

        let mut v = 0;
        if count == BATCH {
            while v + 2 <= vecs {
                let mut a0 = tile_ir[base][v];
                let mut a1 = tile_ir[base + 1][v];
                let mut a2 = tile_ir[base + 2][v];
                let mut b0 = tile_ir[base][v + 1];
                let mut b1 = tile_ir[base + 1][v + 1];
                let mut b2 = tile_ir[base + 2][v + 1];
                let mut m = 0;
                while m < BATCH {
                    let q0 = q_buf[m][v];
                    let q1 = q_buf[m][v + 1];
                    let wm0 = w0[m];
                    let wm1 = w1[m];
                    let wm2 = w2[m];
                    a0 = q0 * wm0 + a0;
                    b0 = q1 * wm0 + b0;
                    a1 = q0 * wm1 + a1;
                    b1 = q1 * wm1 + b1;
                    a2 = q0 * wm2 + a2;
                    b2 = q1 * wm2 + b2;
                    m += 1;
                }
                tile_ir[base][v] = a0;
                tile_ir[base + 1][v] = a1;
                tile_ir[base + 2][v] = a2;
                tile_ir[base][v + 1] = b0;
                tile_ir[base + 1][v + 1] = b1;
                tile_ir[base + 2][v + 1] = b2;
                v += 2;
            }
        }
        while v < vecs {
            let mut a0 = tile_ir[base][v];
            let mut a1 = tile_ir[base + 1][v];
            let mut a2 = tile_ir[base + 2][v];
            let mut m = 0;
            while m < count {
                let qv = q_buf[m][v];
                a0 = qv * w0[m] + a0;
                a1 = qv * w1[m] + a1;
                a2 = qv * w2[m] + a2;
                m += 1;
            }
            tile_ir[base][v] = a0;
            tile_ir[base + 1][v] = a1;
            tile_ir[base + 2][v] = a2;
            v += 1;
        }
        base += 3;
    }
}
