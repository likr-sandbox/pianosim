use piano_radiator::{embedded_body_bytes, BodyArchetype, SpacePrepContext};
use piano_spec::{
    midi_to_key, u7_to_u16, EventKind, LidState, ListenerPerspective, PedalKind, SampleEvent,
    VoicingSpec,
};
use piano_strings::VoicingPrepContext;

use crate::m3_tests::{band_power, hann_windowed_spectrum, mono_mix};
use crate::m4_tests::{note_off, note_on};
use crate::wave4_tests::render_stage4_blocks;
use crate::PianoEngine;

const SR: f32 = 48_000.0;
const SR64: f64 = 48_000.0;

fn band_db(spec: &[piano_simd::Complex], n: usize, lo: f64, hi: f64) -> f64 {
    10.0 * band_power(spec, n, SR64, lo, hi).max(1e-30).log10()
}

fn observe_d3(pl: &mut [f32; 3], pr: &mut [f32; 3], sl: &[f32], sr: &[f32], max_d3: &mut f32) {
    for (&xl, &xr) in sl.iter().zip(sr.iter()) {
        assert!(xl.is_finite() && xr.is_finite());
        *max_d3 = (*max_d3)
            .max((xl - 3.0 * pl[0] + 3.0 * pl[1] - pl[2]).abs())
            .max((xr - 3.0 * pr[0] + 3.0 * pr[1] - pr[2]).abs());
        *pl = [xl, pl[0], pl[1]];
        *pr = [xr, pr[0], pr[1]];
    }
}

fn pedal_ev(kind: PedalKind, cc7: u8, sample_offset: u32) -> SampleEvent {
    SampleEvent {
        sample_offset,
        kind: EventKind::Pedal {
            kind,
            value_u16: u7_to_u16(cc7),
        },
    }
}

#[test]
fn half_pedal_bach_arpeggio_demotes_voices_and_suppresses_grazing_buzz() {
    let mut eng = PianoEngine::new(SR);
    eng.core.set_reverb(0.0, 0.0);

    // 1. Half-pedal C-major triad post-release spectral cleanliness across CC64 = 42, 52, 78.
    for &cc in &[42_u8, 52, 78] {
        eng.reset();
        eng.core.set_reverb(0.0, 0.0);
        let (mut l, mut r) = ([0.0_f32; 128], [0.0_f32; 128]);
        eng.render_block(
            &[
                pedal_ev(PedalKind::Sustain, cc, 0),
                note_on(48, u7_to_u16(80), 0),
                note_on(52, u7_to_u16(80), 16),
                note_on(55, u7_to_u16(80), 32),
                note_on(60, u7_to_u16(80), 48),
                note_on(64, u7_to_u16(80), 64),
            ],
            &mut l,
            &mut r,
        );
        for _ in 0..49 {
            eng.render_block(&[], &mut l, &mut r);
        }
        eng.render_block(
            &[
                note_off(52, u7_to_u16(48), 0),
                note_off(55, u7_to_u16(48), 0),
                note_off(60, u7_to_u16(48), 0),
                note_off(64, u7_to_u16(48), 0),
            ],
            &mut l,
            &mut r,
        );
        for _ in 0..10 {
            eng.render_block(&[], &mut l, &mut r);
        }
        let (tail_l, tail_r) = render_stage4_blocks(&mut eng, 8_192, &[]);
        let spec = hann_windowed_spectrum(&mono_mix(&tail_l, &tail_r));
        let (low_db, hf25_db, hf10_db) = (
            band_db(&spec, 8_192, 80.0, 2_000.0),
            band_db(&spec, 8_192, 2_500.0, 10_000.0),
            band_db(&spec, 8_192, 10_000.0, 20_000.0),
        );
        assert!(
            low_db - hf25_db >= 40.0 && low_db - hf10_db >= 52.0,
            "CC64={cc}: buzz margin low={low_db:.1}, 2.5-10k={hf25_db:.1}, >10k={hf10_db:.1}"
        );
    }

    // 1b. Individual C3 (48) and E4 (64) grazing at CC64 = 42 also suppress 2.5-10 kHz & >10 kHz buzz.
    for &midi in &[48_u8, 64] {
        eng.reset();
        eng.core.set_reverb(0.0, 0.0);
        let (mut l, mut r) = ([0.0_f32; 128], [0.0_f32; 128]);
        eng.render_block(
            &[
                pedal_ev(PedalKind::Sustain, 42, 0),
                note_on(midi, u7_to_u16(82), 0),
            ],
            &mut l,
            &mut r,
        );
        for _ in 0..30 {
            eng.render_block(&[], &mut l, &mut r);
        }
        eng.render_block(&[note_off(midi, u7_to_u16(48), 0)], &mut l, &mut r);
        for _ in 0..10 {
            eng.render_block(&[], &mut l, &mut r);
        }
        let (tail_l, tail_r) = render_stage4_blocks(&mut eng, 8_192, &[]);
        let spec = hann_windowed_spectrum(&mono_mix(&tail_l, &tail_r));
        let (low_db, hf25_db, hf10_db) = (
            band_db(&spec, 8_192, 80.0, 2_000.0),
            band_db(&spec, 8_192, 2_500.0, 10_000.0),
            band_db(&spec, 8_192, 10_000.0, 20_000.0),
        );
        assert!(
            low_db - hf25_db >= 32.0 && low_db - hf10_db >= 50.0,
            "midi={midi} grazing at CC64=42: low={low_db:.1}, 2.5-10k={hf25_db:.1}, >10k={hf10_db:.1}"
        );
    }

    // 2. Bach Prelude style half-pedaled arpeggio: decayed keys demote tail-damper coupling & stay click-free.
    eng.reset();
    let chords: [[u8; 5]; 3] = [
        [48, 52, 55, 60, 64],
        [48, 50, 57, 62, 65],
        [47, 50, 55, 62, 65],
    ];
    let pat = [0_usize, 1, 2, 3, 4, 2, 3, 4];
    let (mut bl, mut br) = ([0.0_f32; 128], [0.0_f32; 128]);
    let (mut pl, mut pr, mut max_d3) = ([0.0_f32; 3], [0.0_f32; 3], 0.0_f32);
    eng.render_block(&[], &mut [], &mut []);
    for (m, ch) in chords.iter().enumerate() {
        let cc = 42 + (m as u8) * 3;
        eng.render_block(
            &[
                pedal_ev(PedalKind::Sustain, cc, 0),
                note_on(36, u7_to_u16(68), 0),
                note_on(ch[0], u7_to_u16(78), 0),
            ],
            &mut bl,
            &mut br,
        );
        observe_d3(&mut pl, &mut pr, &bl, &br, &mut max_d3);
        eng.render_block(&[note_off(36, u7_to_u16(40), 0)], &mut bl, &mut br);
        observe_d3(&mut pl, &mut pr, &bl, &br, &mut max_d3);
        for (idx, &st) in pat.iter().enumerate() {
            let key = ch[st];
            if idx > 0 {
                eng.render_block(&[note_on(key, u7_to_u16(74), 0)], &mut bl, &mut br);
                observe_d3(&mut pl, &mut pr, &bl, &br, &mut max_d3);
            }
            for b in 0..46 {
                let ev = if idx > 0 && b == 44 {
                    &[note_off(key, u7_to_u16(48), 0)][..]
                } else {
                    &[]
                };
                eng.render_block(ev, &mut bl, &mut br);
                observe_d3(&mut pl, &mut pr, &bl, &br, &mut max_d3);
            }
        }
        eng.render_block(&[note_off(ch[0], u7_to_u16(34), 0)], &mut bl, &mut br);
        observe_d3(&mut pl, &mut pr, &bl, &br, &mut max_d3);
        for b in 0..15 {
            let ev = if b == 0 {
                &[pedal_ev(PedalKind::Sustain, 0, 0)][..]
            } else {
                &[]
            };
            eng.render_block(ev, &mut bl, &mut br);
            observe_d3(&mut pl, &mut pr, &bl, &br, &mut max_d3);
        }
    }
    for &cc in &[0_u8, 14, 15, 42, 44, 45, 78, 124, 127, 0] {
        eng.render_block(&[pedal_ev(PedalKind::Sustain, cc, 0)], &mut bl, &mut br);
        observe_d3(&mut pl, &mut pr, &bl, &br, &mut max_d3);
    }
    assert!(max_d3 < 0.025, "unexpected half-pedal click d3x={max_d3}");
    assert!(
        !eng.core
            .bank
            .key_has_transverse_tail(midi_to_key(64).unwrap()),
        "decayed Measure-0 key E4 kept active tail-damper coupling"
    );
}

#[test]
fn closed_damper_release_after_half_pedal_damps_e4_tails_without_ghost_partials() {
    let mut eng = PianoEngine::new(SR);
    eng.core.set_reverb(0.0, 0.0);
    let (mut bl, mut br) = ([0.0_f32; 128], [0.0_f32; 128]);
    eng.render_block(
        &[
            pedal_ev(PedalKind::Sustain, 64, 0),
            note_on(64, u7_to_u16(78), 0),
        ],
        &mut bl,
        &mut br,
    );
    for _ in 0..10 {
        eng.render_block(&[], &mut bl, &mut br);
    }
    let (pre_l, pre_r) = render_stage4_blocks(&mut eng, 4_096, &[]);
    let pre_spec = hann_windowed_spectrum(&mono_mix(&pre_l, &pre_r));
    let pre_tail_db = 10.0
        * band_power(&pre_spec, 4_096, SR64, 1_550.0, 5_000.0)
            .max(1e-30)
            .log10();

    eng.render_block(
        &[
            note_off(64, u7_to_u16(50), 0),
            pedal_ev(PedalKind::Sustain, 0, 0),
        ],
        &mut bl,
        &mut br,
    );
    for _ in 0..130 {
        eng.render_block(&[], &mut bl, &mut br);
    }
    let (post_l, post_r) = render_stage4_blocks(&mut eng, 4_096, &[]);
    let post_spec = hann_windowed_spectrum(&mono_mix(&post_l, &post_r));
    let post_tail_db = 10.0
        * band_power(&post_spec, 4_096, SR64, 1_550.0, 5_000.0)
            .max(1e-30)
            .log10();
    assert!(
        pre_tail_db - post_tail_db >= 55.0,
        "E4 j>=5 tail left undamped after pedal release: pre={pre_tail_db:.1} dB, post={post_tail_db:.1} dB"
    );
}

#[test]
fn zero_crossing_string_under_shallow_preload_and_straddling_pedal_is_click_free() {
    let mut eng = PianoEngine::new(SR);
    eng.core.set_reverb(0.0, 0.0);
    let (mut bl, mut br) = ([0.0_f32; 128], [0.0_f32; 128]);
    let (mut pl, mut pr, mut max_d3) = ([0.0_f32; 3], [0.0_f32; 3], 0.0_f32);

    eng.render_block(
        &[
            pedal_ev(PedalKind::Sustain, 64, 0),
            note_on(48, u7_to_u16(96), 0),
            note_on(60, u7_to_u16(92), 17),
        ],
        &mut bl,
        &mut br,
    );
    observe_d3(&mut pl, &mut pr, &bl, &br, &mut max_d3);
    eng.render_block(
        &[
            note_off(48, u7_to_u16(48), 0),
            note_off(60, u7_to_u16(48), 0),
        ],
        &mut bl,
        &mut br,
    );
    observe_d3(&mut pl, &mut pr, &bl, &br, &mut max_d3);

    // Sweep shallow preload & zero-straddling CC64 across varying string phases.
    for step in 0..80_u8 {
        let cc = match step % 8 {
            0 | 1 => 38,
            2 | 3 => 43,
            4 | 5 => 46,
            _ => 41,
        };
        let off = u32::from(step.wrapping_mul(23) & 127);
        eng.render_block(&[pedal_ev(PedalKind::Sustain, cc, off)], &mut bl, &mut br);
        observe_d3(&mut pl, &mut pr, &bl, &br, &mut max_d3);
    }
    assert!(
        max_d3 < 0.025,
        "zero-crossing / straddling half-pedal click d3x={max_d3}"
    );
}

#[test]
fn live_multibody_swap_under_sostenuto_half_pedal_and_una_corda_has_no_pop() {
    let mut eng = PianoEngine::new(SR);
    let mut sp_prep = SpacePrepContext::new_from_preset_bytes(
        embedded_body_bytes(BodyArchetype::VintageSaloon),
        SR64,
    )
    .expect("saloon prep");
    let mut vc_prep = VoicingPrepContext::new(SR);
    let (mut bl, mut br) = ([0.0_f32; 128], [0.0_f32; 128]);

    eng.render_block(
        &[
            note_on(36, u7_to_u16(96), 0),
            note_on(43, u7_to_u16(92), 8),
            note_on(48, u7_to_u16(90), 16),
            note_on(52, u7_to_u16(88), 24),
            note_on(60, u7_to_u16(82), 32),
            note_on(64, u7_to_u16(80), 40),
        ],
        &mut bl,
        &mut br,
    );
    for _ in 0..8 {
        eng.render_block(&[], &mut bl, &mut br);
    }
    // Latch [36, 43] with Sostenuto after releasing [48, 52, 60, 64] into half-pedal.
    eng.render_block(
        &[
            pedal_ev(PedalKind::Sustain, 45, 0),
            note_off(48, u7_to_u16(48), 16),
            note_off(52, u7_to_u16(48), 16),
            note_off(60, u7_to_u16(48), 16),
            note_off(64, u7_to_u16(48), 16),
            pedal_ev(PedalKind::Sostenuto, 127, 48),
            pedal_ev(PedalKind::UnaCorda, 96, 64),
        ],
        &mut bl,
        &mut br,
    );
    eng.render_block(
        &[
            note_off(36, u7_to_u16(40), 0),
            note_off(43, u7_to_u16(40), 0),
        ],
        &mut bl,
        &mut br,
    );
    for _ in 0..12 {
        eng.render_block(&[], &mut bl, &mut br);
    }

    let mut pl = [bl[127], bl[126], bl[125]];
    let mut pr = [br[127], br[126], br[125]];
    let (mut max_d1, mut max_d3) = (0.0_f32, 0.0_f32);
    let mut observe = |sl: &[f32], sr: &[f32]| {
        for (&xl, &xr) in sl.iter().zip(sr.iter()) {
            assert!(xl.is_finite() && xr.is_finite());
            max_d1 = max_d1.max((xl - pl[0]).abs()).max((xr - pr[0]).abs());
            max_d3 = max_d3
                .max((xl - 3.0 * pl[0] + 3.0 * pl[1] - pl[2]).abs())
                .max((xr - 3.0 * pr[0] + 3.0 * pr[1] - pr[2]).abs());
            pl = [xl, pl[0], pl[1]];
            pr = [xr, pr[0], pr[1]];
        }
    };

    // 1. Direct body swap ConcertGrand -> UprightStudio while Sostenuto + half-pedal chord rings.
    eng.set_body_space(
        BodyArchetype::UprightStudio,
        ListenerPerspective::ConcertHall,
        LidState::Half,
    );
    for b in 0..24_u8 {
        let cc = 41 + ((b * 3) % 14);
        eng.render_block(&[pedal_ev(PedalKind::Sustain, cc, 0)], &mut bl, &mut br);
        observe(&bl, &br);
    }

    // 2. Chunked Prep body+space commit UprightStudio -> VintageSaloon + live Voicing swap.
    let body_blob = sp_prep
        .prepare_space(ListenerPerspective::Player, LidState::Open)
        .to_vec();
    for (idx, chunk) in body_blob.chunks(16_384).enumerate() {
        assert!(eng.stage_precomputed_body_space_chunk(
            BodyArchetype::VintageSaloon,
            ListenerPerspective::Player,
            LidState::Open,
            idx * 16_384,
            chunk,
            body_blob.len(),
        ));
    }
    assert!(eng.commit_precomputed_body_space(
        BodyArchetype::VintageSaloon,
        ListenerPerspective::Player,
        LidState::Open,
    ));
    let vc_spec = VoicingSpec::from_globals_with_a4(0.62, 1.15, 1.05, 1.0, 441.0);
    let vc_blob = vc_prep.prepare_voicing(&vc_spec).to_vec();
    assert!(eng.apply_precomputed_voicing(vc_spec, &vc_blob));

    for _ in 0..24 {
        eng.render_block(&[], &mut bl, &mut br);
        observe(&bl, &br);
    }
    assert!(
        max_d1 < 0.06 && max_d3 < 0.06,
        "live body/voicing swap pop under half-pedal + sostenuto: d1={max_d1:.4}, d3={max_d3:.4}"
    );
}

#[test]
fn item4_phase_invariant_tail_and_mid_block_prefix_damping() {
    let mut eng = PianoEngine::new(SR);
    eng.core.set_reverb(0.0, 0.0);
    let (mut bl, mut br) = ([0.0_f32; 128], [0.0_f32; 128]);

    // (a) Chunk-boundary phase invariance under CC64 = 42, 45 (shallow grazing without 375 Hz flicker).
    let k_e4 = midi_to_key(64).unwrap();
    for &cc in &[42_u8, 45] {
        for &offset in &[0_u32, 37] {
            eng.reset();
            eng.core.set_reverb(0.0, 0.0);
            let (mut pl, mut pr, mut max_d3) = ([0.0_f32; 3], [0.0_f32; 3], 0.0_f32);
            eng.render_block(
                &[
                    pedal_ev(PedalKind::Sustain, cc, 0),
                    note_on(64, u7_to_u16(82), offset),
                    note_on(48, u7_to_u16(82), offset),
                ],
                &mut bl,
                &mut br,
            );
            for b in 0..48 {
                let ev = if b == 12 {
                    &[
                        note_off(64, u7_to_u16(48), offset),
                        note_off(48, u7_to_u16(48), offset),
                    ][..]
                } else {
                    &[]
                };
                eng.render_block(ev, &mut bl, &mut br);
                if b == 11 {
                    (pl, pr) = ([bl[127], bl[126], bl[125]], [br[127], br[126], br[125]]);
                } else if b >= 12 {
                    observe_d3(&mut pl, &mut pr, &bl, &br, &mut max_d3);
                }
            }
            let (mut prev, mut flips) = (eng.core.bank.key_has_transverse_tail(k_e4), 0_usize);
            for _ in 0..64 {
                eng.render_block(&[], &mut bl, &mut br);
                let cur = eng.core.bank.key_has_transverse_tail(k_e4);
                if cur != prev {
                    flips += 1;
                    prev = cur;
                }
                observe_d3(&mut pl, &mut pr, &bl, &br, &mut max_d3);
            }
            assert!(max_d3 < 0.025, "CC64={cc} off={offset}: d3x={max_d3}");
            assert!(
                flips <= 1,
                "CC64={cc} off={offset}: tail flicker ({flips} flips)"
            );
        }
    }

    // (b) Mid-block NoteOff + Pedal(0) at sample_offset = 37 damps a sympathetically excited
    // prefix-only key (may_be_nonzero && !tail_nonzero) cleanly without waking tail_nonzero.
    eng.reset();
    eng.core.set_reverb(0.0, 0.0);
    let k_c4 = midi_to_key(60).unwrap();
    eng.render_block(
        &[
            pedal_ev(PedalKind::Sustain, 127, 0),
            note_on(48, u7_to_u16(100), 0),
        ],
        &mut bl,
        &mut br,
    );
    for _ in 0..24 {
        eng.render_block(&[], &mut bl, &mut br);
    }
    let e_symp_pre = eng.core.bank.key_sav_energy_joules(k_c4);
    assert!(e_symp_pre > 0.0);
    assert!(!eng.core.bank.key_has_transverse_tail(k_c4));

    eng.render_block(
        &[
            note_off(48, u7_to_u16(64), 37),
            pedal_ev(PedalKind::Sustain, 0, 37),
        ],
        &mut bl,
        &mut br,
    );
    for _ in 0..160 {
        eng.render_block(&[], &mut bl, &mut br);
    }
    let e_symp_post = eng.core.bank.key_sav_energy_joules(k_c4);
    assert!(!eng.core.bank.key_has_transverse_tail(k_c4));
    assert!(
        e_symp_post < e_symp_pre * 1.0e-3,
        "mid-block Pedal(0) failed to damp prefix-only sympathetic key: pre={e_symp_pre:.3e}, post={e_symp_post:.3e}"
    );
}
