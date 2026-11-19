//! Preallocated time-sliced `VoicingBlob` commit state (`8` keys/block over `11` blocks,
//! zero audio-thread heap allocations).

use piano_spec::{build_scale_design, StringPhysicalParams, VoicingSpec, NUM_KEYS};

use crate::PianoEngine;

/// Number of piano keys updated per audio block during time-sliced `VoicingBlob` commit.
pub(crate) const VOICING_COMMIT_KEYS_PER_BLOCK: u8 = 8;

/// Preallocated state for zero-allocation incremental `VoicingBlob` application (`11` blocks).
#[derive(Clone, Debug)]
pub(crate) struct VoicingCommitState {
    pub active: bool,
    pub spec: VoicingSpec,
    pub next_key: u8,
    pub keys_per_block: u8,
    pub scale_specs: [StringPhysicalParams; NUM_KEYS],
    pub voicing_blob_buf: Vec<f32>,
    pub blob_len: usize,
}

impl VoicingCommitState {
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        let default_spec = VoicingSpec::default();
        let cap = crate::voicing_blob_len(sample_rate);
        Self {
            active: false,
            spec: default_spec.clone(),
            next_key: 0,
            keys_per_block: VOICING_COMMIT_KEYS_PER_BLOCK,
            scale_specs: build_scale_design(&default_spec),
            voicing_blob_buf: vec![0.0_f32; cap],
            blob_len: 0,
        }
    }
}

impl PianoEngine {
    /// Returns `true` while a time-sliced `VoicingBlob` commit is still applying across blocks.
    #[must_use]
    #[inline]
    pub const fn is_voicing_applying(&self) -> bool {
        self.voicing_commit.active
    }

    /// Validates `blob` and latches preallocated `VoicingCommitState` for incremental `8`-key/block
    /// installation on subsequent [`Self::render_block`] calls without heap allocations.
    pub fn begin_voicing_commit(&mut self, spec: VoicingSpec, blob: &[f32]) -> bool {
        if self.voicing_commit.active {
            return false;
        }
        let mut v = spec.sanitized();
        if blob.len() > self.voicing_commit.voicing_blob_buf.len()
            || !self.core.bank.validate_voicing_blob(&v, blob)
        {
            return false;
        }
        self.voicing_commit.voicing_blob_buf[..blob.len()].copy_from_slice(blob);
        self.voicing_commit.blob_len = blob.len();
        v.hammer_hardness = self.voicing_smoother.current_hammer_hardness();
        let scale_specs = build_scale_design(&v);
        self.core.action.set_voicing_scale_specs(&v, &scale_specs);
        if !self.core.bank.begin_voicing_install(
            &v,
            &scale_specs,
            &self.voicing_commit.voicing_blob_buf[..blob.len()],
        ) {
            return false;
        }
        self.core.bridge.set_voicing(&v);
        self.voicing_smoother.set_tuning_applied_full(
            v.unison_detune_scale,
            v.stretch_amount,
            v.string_length_scale,
            v.a4_freq_hz,
            v.temperament,
            v.condition,
        );
        self.voicing_commit.spec = v;
        self.voicing_commit.scale_specs = scale_specs;
        self.voicing_commit.next_key = 0;
        self.voicing_commit.active = true;
        true
    }

    /// Advances an active time-sliced `VoicingBlob` commit by up to `keys_per_block` (`8`) keys,
    /// atomically finalizing WDF bridge port resistances on block `11` (`next_key == 88`).
    pub fn step_voicing_commit(&mut self) -> bool {
        if !self.voicing_commit.active {
            return false;
        }
        let k_start = usize::from(self.voicing_commit.next_key);
        let step = usize::from(self.voicing_commit.keys_per_block.max(1));
        let k_end = (k_start + step).min(NUM_KEYS);
        let mut spec = self.voicing_commit.spec.clone();
        spec.hammer_hardness = self.voicing_smoother.current_hammer_hardness();
        let blob = &self.voicing_commit.voicing_blob_buf[..self.voicing_commit.blob_len];
        for k in k_start..k_end {
            let _ = self.core.bank.install_voicing_key(k, blob);
            self.core
                .action
                .update_voicing_key(k, &spec, &self.voicing_commit.scale_specs[k]);
        }
        self.voicing_commit.next_key = k_end as u8;
        if k_end >= NUM_KEYS {
            let _ = self.core.bank.finish_voicing_ports(blob);
            self.core.bridge.set_voicing(&spec);
            self.core
                .bridge
                .set_port_resistances(self.core.bank.bridge_port_resistances());
            self.core.current_voicing = spec;
            self.voicing_smoother.set_tuning_applied_full(
                self.core.current_voicing.unison_detune_scale,
                self.core.current_voicing.stretch_amount,
                self.core.current_voicing.string_length_scale,
                self.core.current_voicing.a4_freq_hz,
                self.core.current_voicing.temperament,
                self.core.current_voicing.condition,
            );
            self.voicing_commit.active = false;
        }
        true
    }
}
