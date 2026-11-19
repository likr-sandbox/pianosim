//! Integration tests verifying Stage 3 duplex voicing sync and bridge subnormal retirement.

use piano_spec::VoicingSpec;

use crate::PianoEngine;

#[test]
fn engine_syncs_duplex_voicing_on_init_precomputed_and_dirty_tuning() {
    let mut init_spec = VoicingSpec::from_globals(0.5, 1.0, 1.75, 1.0);
    init_spec.note_offsets[87].pitch_cents = 15.0;
    let eng_init = PianoEngine::new_with_voicing(48_000.0, &init_spec);
    assert_eq!(eng_init.core.bridge.duplex().voicing().stretch_amount, 1.75);
    assert_eq!(
        eng_init.core.bridge.duplex().voicing().note_offsets[87].pitch_cents,
        15.0
    );

    let mut eng = PianoEngine::new(48_000.0);
    eng.set_voicing_with_a4(0.5, 1.0, 1.5, 1.0, 415.0);
    let (mut l, mut r) = ([0.0_f32; 128], [0.0_f32; 128]);
    for _ in 0..80 {
        eng.render_block(&[], &mut l, &mut r);
    }
    assert!((eng.core.bridge.duplex().a4_freq_hz() - 415.0).abs() < 1e-4);
    assert!((eng.core.bridge.duplex().voicing().stretch_amount - 1.5).abs() < 1e-3);

    // apply_precomputed_voicing also updates BridgeBank's DuplexBank with full VoicingSpec.
    let mut vc_prep = piano_strings::VoicingPrepContext::new(48_000.0);
    let mut pre_spec = VoicingSpec::from_globals_with_a4(0.6, 1.2, 0.75, 1.0, 442.0);
    pre_spec.note_offsets[63].pitch_cents = -18.0;
    let blob = vc_prep.prepare_voicing(&pre_spec);
    assert!(eng.apply_precomputed_voicing(pre_spec, blob));
    assert!((eng.core.bridge.duplex().a4_freq_hz() - 442.0).abs() < 1e-4);
    assert!((eng.core.bridge.duplex().voicing().stretch_amount - 0.75).abs() < 1e-4);
    assert!((eng.core.bridge.duplex().voicing().note_offsets[63].pitch_cents + 18.0).abs() < 1e-4);

    // Once StringBank is silent, EngineCore::render_event_slice calls
    // self.bridge.flush_subnormals() so BridgeBank's idle bypass re-engages after decay.
    let mut a = [0.0_f32; piano_spec::TOTAL_STRINGS];
    a[140] = 1.0e-14;
    let mut b = [0.0_f32; piano_spec::TOTAL_STRINGS];
    let mut z = [0.0_f32; piano_bridge::NUM_BRIDGE_ZONES];
    let _ = eng.core.bridge.step_bridge_sample(
        &a,
        &[0.0; piano_bridge::NUM_BRIDGE_ZONES],
        &mut b,
        &mut z,
    );
    assert!(!eng.core.bridge.is_quiescent());
    for _ in 0..400 {
        eng.render_block(&[], &mut l, &mut r);
        if eng.core.bridge.is_quiescent() {
            break;
        }
    }
    assert!(
        eng.core.bridge.is_quiescent(),
        "EngineCore::render_event_slice did not return BridgeBank to quiescent idle bypass (ssm={}, hertz={}, duplex={})",
        eng.core.bridge.ssm().is_quiescent(),
        eng.core.bridge.hertz().is_quiescent(),
        eng.core.bridge.duplex().is_quiescent(),
    );
}
