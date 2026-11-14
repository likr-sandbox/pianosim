//! Block-level engine telemetry metrics mapped to the shared 8-float ABI layout.

/// Real-time diagnostic metrics computed per rendered audio block.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct EngineTelemetry {
    /// Number of currently active voices.
    pub active_voices: u32,
    /// Peak absolute amplitude on the left output channel in the last block.
    pub peak_l: f32,
    /// Peak absolute amplitude on the right output channel in the last block.
    pub peak_r: f32,
    /// Root-mean-square level of the mono mixdown `0.5 * (L + R)` in the last block.
    pub rms_mono: f32,
}

impl EngineTelemetry {
    /// Serializes telemetry to the shared `[f32; 8]` ABI buffer layout.
    #[must_use]
    #[inline]
    pub fn to_array(self) -> [f32; 8] {
        [
            self.active_voices as f32,
            self.peak_l,
            self.peak_r,
            self.rms_mono,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    }
}

/// Computes block peak and mono RMS metrics without allocation.
#[must_use]
pub fn compute_block_telemetry(
    active_voices: u32,
    out_l: &[f32],
    out_r: &[f32],
) -> EngineTelemetry {
    let frames = out_l.len().min(out_r.len());
    if frames == 0 {
        return EngineTelemetry {
            active_voices,
            peak_l: 0.0,
            peak_r: 0.0,
            rms_mono: 0.0,
        };
    }

    let mut peak_l = 0.0_f32;
    let mut peak_r = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    for i in 0..frames {
        let l = out_l[i];
        let r = out_r[i];
        let al = l.abs();
        let ar = r.abs();
        if al > peak_l {
            peak_l = al;
        }
        if ar > peak_r {
            peak_r = ar;
        }
        let mono = 0.5 * (l + r);
        sum_sq += mono * mono;
    }

    EngineTelemetry {
        active_voices,
        peak_l,
        peak_r,
        rms_mono: (sum_sq / (frames as f32)).sqrt(),
    }
}
