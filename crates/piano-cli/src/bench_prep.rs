//! Half-pedal + SpaceBlob / VoicingBlob isolation microsecond benchmark (`piano-cli bench`).

use std::hint::black_box;
use std::time::Instant;

use piano_engine::{
    embedded_body_bytes, BodyArchetype, PerfProfile, PianoEngine, SpacePrepContext,
    VoicingPrepContext,
};
use piano_spec::{
    u7_to_u16, EventKind, LidState, ListenerPerspective, PedalKind, SampleEvent, Temperament,
    VoicingSpec,
};

use crate::bench_rt::{compute_percentiles, Percentiles};

const BLOCK_FRAMES: usize = 128;
const SAMPLE_RATE: f32 = 48_000.0;
const BLOB_STREAM_CHUNK: usize = 8_192;
const WARMUP_HALF_PEDAL_BLOCKS: usize = 64;
const STEADY_HALF_PEDAL_BLOCKS: usize = 256;
const PREP_SWAPS: usize = 8;
const BLOCKS_PER_SWAP: usize = 40;
const HALF_PEDAL_CHORD: [u8; 8] = [48, 55, 60, 64, 67, 72, 76, 79];
const HALF_PEDAL_ARP: [u8; 8] = [60, 64, 67, 72, 76, 79, 84, 72];
const SEP: &str =
    "----------------------------------------------------------------------------------------------------";
const EQ_SEP: &str =
    "====================================================================================================";

struct PreparedSpaceEntry {
    body: BodyArchetype,
    persp: ListenerPerspective,
    lid: LidState,
    blob: Vec<f32>,
}

struct PreparedVoicingEntry {
    spec: VoicingSpec,
    blob: Vec<f32>,
}

#[derive(Copy, Clone, Debug)]
pub struct PrepIsolationReport {
    pub steady_render: Percentiles,
    pub space_total: Percentiles,
    pub space_chunk0: Percentiles,
    pub space_chunk_mid: Percentiles,
    pub space_commit: Percentiles,
    pub space_crossfade_render: Percentiles,
    pub space_direct_apply: Percentiles,
    pub space_sync_reconfig: Percentiles,
    pub voicing_total: Percentiles,
    pub voicing_apply: Percentiles,
    pub voicing_commit_render: Percentiles,
    pub combined_total: Percentiles,
    pub combined_space_max_us: f64,
    pub combined_voicing_max_us: f64,
    pub combined_render_max_us: f64,
    pub space_blob_f32: usize,
    pub voicing_blob_f32: usize,
}

fn init_half_pedal_engine(profile: PerfProfile) -> PianoEngine {
    let mut engine = PianoEngine::new_with_profile(SAMPLE_RATE, VoicingSpec::default(), profile);
    let (mut out_l, mut out_r) = ([0.0_f32; BLOCK_FRAMES], [0.0_f32; BLOCK_FRAMES]);
    let noop = SampleEvent {
        sample_offset: 0,
        kind: EventKind::AllNotesOff,
    };
    let mut init_evs = [noop; 9];
    init_evs[0] = SampleEvent {
        sample_offset: 0,
        kind: EventKind::Pedal {
            kind: PedalKind::Sustain,
            value_u16: u7_to_u16(64),
        },
    };
    for (i, &key) in HALF_PEDAL_CHORD.iter().enumerate() {
        init_evs[i + 1] = SampleEvent {
            sample_offset: 0,
            kind: EventKind::NoteOn {
                midi_key: key,
                velocity_u16: u7_to_u16(88),
            },
        };
    }
    engine.render_block(&init_evs, &mut out_l, &mut out_r);
    for _ in 0..WARMUP_HALF_PEDAL_BLOCKS {
        engine.render_block(&[], &mut out_l, &mut out_r);
    }
    black_box((&out_l, &out_r));
    engine
}

fn step_half_pedal_events(block: usize, buf: &mut [SampleEvent; 3]) -> &[SampleEvent] {
    if !block.is_multiple_of(16) {
        return &[];
    }
    let idx = block / 16;
    let prev = HALF_PEDAL_ARP[idx % HALF_PEDAL_ARP.len()];
    let next = HALF_PEDAL_ARP[(idx + 1) % HALF_PEDAL_ARP.len()];
    let cc = 42 + ((idx * 3) % 18) as u8;
    buf[0] = SampleEvent {
        sample_offset: 0,
        kind: EventKind::Pedal {
            kind: PedalKind::Sustain,
            value_u16: u7_to_u16(cc),
        },
    };
    buf[1] = SampleEvent {
        sample_offset: 0,
        kind: EventKind::NoteOff {
            midi_key: prev,
            release_vel_u16: u7_to_u16(48),
        },
    };
    buf[2] = SampleEvent {
        sample_offset: 32,
        kind: EventKind::NoteOn {
            midi_key: next,
            velocity_u16: u7_to_u16(82),
        },
    };
    &buf[..]
}

fn prepare_space_entries(profile: PerfProfile) -> Vec<PreparedSpaceEntry> {
    let mut ctx = SpacePrepContext::new_concert_grand(f64::from(SAMPLE_RATE));
    ctx.set_perf_profile(&profile);
    let specs = [
        (
            BodyArchetype::UprightStudio,
            ListenerPerspective::ConcertHall,
            LidState::Closed,
        ),
        (
            BodyArchetype::ConcertGrand,
            ListenerPerspective::Player,
            LidState::Open,
        ),
        (
            BodyArchetype::VintageSaloon,
            ListenerPerspective::BinauralHrtf,
            LidState::Half,
        ),
        (
            BodyArchetype::ConcertGrand,
            ListenerPerspective::CloseStereo,
            LidState::Open,
        ),
    ];
    specs
        .into_iter()
        .map(|(body, persp, lid)| {
            ctx.set_body_preset_bytes(embedded_body_bytes(body))
                .expect("embedded body preset");
            PreparedSpaceEntry {
                body,
                persp,
                lid,
                blob: ctx.prepare_space(persp, lid).to_vec(),
            }
        })
        .collect()
}

fn prepare_voicing_entries(profile: PerfProfile) -> Vec<PreparedVoicingEntry> {
    let mut ctx = VoicingPrepContext::new_with_profile(SAMPLE_RATE, &profile);
    let specs = [
        VoicingSpec::from_globals_full(
            0.85,
            2.2,
            1.4,
            1.0,
            415.0,
            Temperament::WerckmeisterIII,
            0.2,
        ),
        VoicingSpec::from_globals_full(0.30, 1.0, 1.0, 1.0, 440.0, Temperament::Equal, 0.0),
        VoicingSpec::from_globals_full(
            0.65,
            1.6,
            1.2,
            1.05,
            442.0,
            Temperament::KirnbergerIII,
            0.5,
        ),
        VoicingSpec::from_globals_full(0.50, 1.0, 1.0, 1.0, 440.0, Temperament::Equal, 0.0),
    ];
    specs
        .into_iter()
        .map(|spec| PreparedVoicingEntry {
            blob: ctx.prepare_voicing(&spec).to_vec(),
            spec,
        })
        .collect()
}

fn to_pct(mut v: Vec<f64>) -> Percentiles {
    if v.is_empty() {
        v.push(0.0);
    }
    compute_percentiles(&mut v)
}

fn step_space_stream(
    engine: &mut PianoEngine,
    entry: &PreparedSpaceEntry,
    off: usize,
) -> (Option<usize>, f64, bool) {
    let total = entry.blob.len();
    let n = BLOB_STREAM_CHUNK.min(total - off);
    let is_last = off + n >= total;
    let t0 = Instant::now();
    let ok = engine.stage_precomputed_body_space_chunk(
        entry.body,
        entry.persp,
        entry.lid,
        off,
        &entry.blob[off..off + n],
        total,
    );
    assert!(ok, "stage_precomputed_body_space_chunk failed");
    if is_last {
        let committed = engine.commit_precomputed_body_space(entry.body, entry.persp, entry.lid);
        assert!(committed, "commit_precomputed_body_space failed");
    }
    let us = t0.elapsed().as_secs_f64() * 1e6;
    (if is_last { None } else { Some(off + n) }, us, is_last)
}

pub fn measure_prep_isolation(profile: PerfProfile) -> PrepIsolationReport {
    let spaces = prepare_space_entries(profile);
    let voicings = prepare_voicing_entries(profile);
    let noop = SampleEvent {
        sample_offset: 0,
        kind: EventKind::AllNotesOff,
    };
    let mut ev_buf = [noop; 3];
    let (mut out_l, mut out_r) = ([0.0_f32; BLOCK_FRAMES], [0.0_f32; BLOCK_FRAMES]);

    // 1. Steady Half-pedal baseline
    let mut engine = init_half_pedal_engine(profile);
    let mut steady_us = Vec::with_capacity(STEADY_HALF_PEDAL_BLOCKS);
    for b in 0..STEADY_HALF_PEDAL_BLOCKS {
        let evs = step_half_pedal_events(b, &mut ev_buf);
        let t0 = Instant::now();
        engine.render_block(evs, &mut out_l, &mut out_r);
        steady_us.push(t0.elapsed().as_secs_f64() * 1e6);
        black_box((&out_l, &out_r));
    }

    // 2a. Half-pedal + SpaceBlob chunked stream & commit
    let mut engine = init_half_pedal_engine(profile);
    let mut space_total = Vec::with_capacity(PREP_SWAPS * BLOCKS_PER_SWAP);
    let (mut space_chunk0, mut space_chunk_mid, mut space_commit, mut space_xfade) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut active_space, mut swap_idx) = (None::<(usize, usize)>, 0usize);
    for b in 0..(PREP_SWAPS * BLOCKS_PER_SWAP) {
        if b.is_multiple_of(BLOCKS_PER_SWAP) && active_space.is_none() {
            active_space = Some((swap_idx % spaces.len(), 0));
            swap_idx += 1;
        }
        let evs = step_half_pedal_events(b, &mut ev_buf);
        let was_xfade = engine.space_guard().is_crossfading();
        let t_blk0 = Instant::now();
        if let Some((idx, off)) = active_space {
            if off > 0 || !was_xfade {
                let (next_off, sp_us, is_last) = step_space_stream(&mut engine, &spaces[idx], off);
                if is_last {
                    space_commit.push(sp_us);
                } else if off == 0 {
                    space_chunk0.push(sp_us);
                } else {
                    space_chunk_mid.push(sp_us);
                }
                active_space = next_off.map(|o| (idx, o));
            }
        }
        let t_ren0 = Instant::now();
        engine.render_block(evs, &mut out_l, &mut out_r);
        let ren_us = t_ren0.elapsed().as_secs_f64() * 1e6;
        space_total.push(t_blk0.elapsed().as_secs_f64() * 1e6);
        if was_xfade || engine.space_guard().is_crossfading() {
            space_xfade.push(ren_us);
        }
        black_box((&out_l, &out_r));
    }

    // 2a-ref: Direct apply_precomputed_body_space & synchronous set_body_space
    let mut engine = init_half_pedal_engine(profile);
    let (mut space_direct, mut space_sync) =
        (Vec::with_capacity(PREP_SWAPS), Vec::with_capacity(4));
    for i in 0..PREP_SWAPS {
        let e = &spaces[i % spaces.len()];
        let t0 = Instant::now();
        assert!(engine.apply_precomputed_body_space(e.body, e.persp, e.lid, &e.blob));
        space_direct.push(t0.elapsed().as_secs_f64() * 1e6);
        for _ in 0..8 {
            engine.render_block(&[], &mut out_l, &mut out_r);
        }
    }
    for i in 0..4 {
        let e = &spaces[i % spaces.len()];
        let t0 = Instant::now();
        assert!(engine.set_body_space(e.body, e.persp, e.lid));
        space_sync.push(t0.elapsed().as_secs_f64() * 1e6);
        for _ in 0..8 {
            engine.render_block(&[], &mut out_l, &mut out_r);
        }
    }

    // 2b. Half-pedal + VoicingBlob commit
    let mut engine = init_half_pedal_engine(profile);
    let mut voicing_total = Vec::with_capacity(PREP_SWAPS * BLOCKS_PER_SWAP);
    let (mut voicing_apply, mut voicing_commit_render) = (Vec::new(), Vec::new());
    for b in 0..(PREP_SWAPS * BLOCKS_PER_SWAP) {
        let evs = step_half_pedal_events(b, &mut ev_buf);
        let t_blk0 = Instant::now();
        let is_commit = b.is_multiple_of(BLOCKS_PER_SWAP);
        if is_commit {
            let e = &voicings[(b / BLOCKS_PER_SWAP) % voicings.len()];
            engine.set_hammer_hardness(e.spec.hammer_hardness);
            let t_vc0 = Instant::now();
            assert!(engine.apply_precomputed_voicing(e.spec.clone(), &e.blob));
            voicing_apply.push(t_vc0.elapsed().as_secs_f64() * 1e6);
        }
        let t_ren0 = Instant::now();
        engine.render_block(evs, &mut out_l, &mut out_r);
        let ren_us = t_ren0.elapsed().as_secs_f64() * 1e6;
        voicing_total.push(t_blk0.elapsed().as_secs_f64() * 1e6);
        if is_commit {
            voicing_commit_render.push(ren_us);
        }
        black_box((&out_l, &out_r));
    }

    // 2c. Half-pedal + Combined Prep sweep
    let mut engine = init_half_pedal_engine(profile);
    let mut combined_total = Vec::with_capacity(PREP_SWAPS * BLOCKS_PER_SWAP);
    let (mut comb_sp_max, mut comb_vc_max, mut comb_ren_max) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut active_sp, mut comb_swap) = (None::<(usize, usize)>, 0usize);
    for b in 0..(PREP_SWAPS * BLOCKS_PER_SWAP) {
        let evs = step_half_pedal_events(b, &mut ev_buf);
        if b.is_multiple_of(BLOCKS_PER_SWAP) {
            if active_sp.is_none() {
                active_sp = Some((comb_swap % spaces.len(), 0));
            }
            comb_swap += 1;
        }
        let t_blk0 = Instant::now();
        if b.is_multiple_of(BLOCKS_PER_SWAP) {
            let e = &voicings[(b / BLOCKS_PER_SWAP) % voicings.len()];
            engine.set_hammer_hardness(e.spec.hammer_hardness);
            let t_vc0 = Instant::now();
            assert!(engine.apply_precomputed_voicing(e.spec.clone(), &e.blob));
            comb_vc_max = comb_vc_max.max(t_vc0.elapsed().as_secs_f64() * 1e6);
        }
        if let Some((idx, off)) = active_sp {
            if off > 0 || !engine.space_guard().is_crossfading() {
                let (next_off, sp_us, _) = step_space_stream(&mut engine, &spaces[idx], off);
                comb_sp_max = comb_sp_max.max(sp_us);
                active_sp = next_off.map(|o| (idx, o));
            }
        }
        let t_ren0 = Instant::now();
        engine.render_block(evs, &mut out_l, &mut out_r);
        comb_ren_max = comb_ren_max.max(t_ren0.elapsed().as_secs_f64() * 1e6);
        combined_total.push(t_blk0.elapsed().as_secs_f64() * 1e6);
        black_box((&out_l, &out_r));
    }

    PrepIsolationReport {
        steady_render: to_pct(steady_us),
        space_total: to_pct(space_total),
        space_chunk0: to_pct(space_chunk0),
        space_chunk_mid: to_pct(space_chunk_mid),
        space_commit: to_pct(space_commit),
        space_crossfade_render: to_pct(space_xfade),
        space_direct_apply: to_pct(space_direct),
        space_sync_reconfig: to_pct(space_sync),
        voicing_total: to_pct(voicing_total),
        voicing_apply: to_pct(voicing_apply),
        voicing_commit_render: to_pct(voicing_commit_render),
        combined_total: to_pct(combined_total),
        combined_space_max_us: comb_sp_max,
        combined_voicing_max_us: comb_vc_max,
        combined_render_max_us: comb_ren_max,
        space_blob_f32: spaces[0].blob.len(),
        voicing_blob_f32: voicings[0].blob.len(),
    }
}

fn print_iso_row(label: &str, pct: &Percentiles, budget: f64) {
    let tag = if pct.max > budget {
        "SPIKE (>2667 us)"
    } else if pct.p99 > budget * 0.70 {
        "WARN (>70%)"
    } else {
        "INFO (<70%)"
    };
    println!(
        "{:<44} | {:>6.1} us | {:>5.1} us | {:>5.1} us | {:>5.1} us | {:>5.1} us | {tag}",
        label, budget, pct.p50, pct.p95, pct.p99, pct.max
    );
}

pub fn run_prep_isolation_section(profile: PerfProfile) {
    let r = measure_prep_isolation(profile);
    let b = 2666.7_f64;
    println!(
        "Half-pedal + Prep Sweep Isolation (128f @ 48 kHz; SpaceBlob={} f32, VoicingBlob={} f32)\n{SEP}",
        r.space_blob_f32, r.voicing_blob_f32
    );
    print_iso_row("Half-pedal steady render (baseline)", &r.steady_render, b);
    print_iso_row("Half-pedal + SpaceBlob (total block)", &r.space_total, b);
    print_iso_row(
        "Half-pedal + VoicingBlob (total block)",
        &r.voicing_total,
        b,
    );
    print_iso_row(
        "Half-pedal + Combined Prep (total block)",
        &r.combined_total,
        b,
    );
    println!("{SEP}");
    print_iso_row("  SpaceBlob stage chunk0 (incl. reset)", &r.space_chunk0, b);
    print_iso_row("  SpaceBlob stage mid chunk", &r.space_chunk_mid, b);
    print_iso_row("  SpaceBlob last chunk + commit", &r.space_commit, b);
    print_iso_row(
        "  SpaceBlob crossfade render_block",
        &r.space_crossfade_render,
        b,
    );
    print_iso_row(
        "  SpaceBlob direct apply_precomputed",
        &r.space_direct_apply,
        b,
    );
    print_iso_row(
        "  SpaceBlob sync set_body_space (ref)",
        &r.space_sync_reconfig,
        b,
    );
    print_iso_row(
        "  VoicingBlob apply_precomputed_voicing",
        &r.voicing_apply,
        b,
    );
    print_iso_row(
        "  VoicingBlob render_block on commit",
        &r.voicing_commit_render,
        b,
    );
    println!(
        "Combined max breakdown: space_stage_commit={:.1} us, voicing_apply={:.1} us, render_block={:.1} us\n{EQ_SEP}",
        r.combined_space_max_us, r.combined_voicing_max_us, r.combined_render_max_us
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prep_isolation_bench_runs_and_returns_finite_timings() {
        let r = measure_prep_isolation(PerfProfile::realtime());
        assert!(r.space_blob_f32 > 1000 && r.voicing_blob_f32 > 1000);
        assert!(r.steady_render.p50.is_finite() && r.steady_render.max > 0.0);
        assert!(r.space_commit.max.is_finite() && r.space_commit.max > 0.0);
        assert!(r.voicing_apply.max.is_finite() && r.voicing_apply.max > 0.0);
        assert!(r.combined_total.max.is_finite() && r.combined_total.max > 0.0);
    }
}
