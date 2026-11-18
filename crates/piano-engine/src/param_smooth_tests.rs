use super::*;
use piano_spec::{LidState, ListenerPerspective};

#[test]
fn voicing_smoother_settles_and_idles() {
    let mut sm = VoicingSmoother::new();
    sm.set_target_with_a4(0.8, 1.5, 1.2, 0.9, 442.0);
    let sr = 48_000.0;
    let mut steps = 0;
    while sm.is_active() && steps < 1_000 {
        let spec = sm.step_block(128, sr).expect("spec expected");
        assert!(spec.hammer_hardness >= 0.5 && spec.hammer_hardness <= 0.8);
        assert!(spec.a4_freq_hz >= 440.0 && spec.a4_freq_hz <= 442.0);
        steps += 1;
    }
    assert!(!sm.is_active());
    assert_eq!(sm.current_hammer_hardness(), 0.8);
    assert_eq!(sm.current_unison_detune(), 1.5);
    assert_eq!(sm.current_stretch(), 1.2);
    assert_eq!(sm.current_length_scale(), 0.9);
    assert_eq!(sm.current_a4_freq_hz(), 442.0);
    assert!(sm.step_block(128, sr).is_none());
}

#[test]
fn spatial_crossfade_instant_on_silence() {
    let mut guard = SpaceCrossfadeGuard::new(ListenerPerspective::Player, LidState::Open);
    guard.update_envelope(0.0);
    let act = guard.request_space(ListenerPerspective::ConcertHall, LidState::Half, 48_000.0);
    assert_eq!(
        act,
        SpaceTransitionAction::InstantSwitch {
            target_idx: 0,
            perspective: ListenerPerspective::ConcertHall,
            lid: LidState::Half,
        }
    );
    guard.update_envelope(0.05);
    let act = guard.request_space(ListenerPerspective::Player, LidState::Open, 48_000.0);
    assert!(matches!(act, SpaceTransitionAction::StartCrossfade { .. }));
    assert!(guard.is_crossfading());

    let (w_out_0, w_in_0) = raised_cosine_weights(0, 384);
    assert_eq!(w_out_0, 1.0);
    assert_eq!(w_in_0, 0.0);
    let (w_out_mid, w_in_mid) = raised_cosine_weights(192, 384);
    assert!((w_out_mid - 0.5).abs() < 1e-6);
    assert!((w_in_mid - 0.5).abs() < 1e-6);
    let (w_out_end, w_in_end) = raised_cosine_weights(384, 384);
    assert_eq!(w_out_end, 0.0);
    assert_eq!(w_in_end, 1.0);
}

#[test]
fn voicing_dirty_mask_isolates_hammer_from_tuning_and_a4() {
    let mut sm = VoicingSmoother::new();
    sm.set_target(0.85, 1.0, 1.0, 1.0);
    let (_, d_h) = sm.step_block_with_dirty(128, 48_000.0).expect("dirty");
    assert_eq!(d_h, DIRTY_HAMMER);

    let mut sm_a4 = VoicingSmoother::new();
    sm_a4.set_a4_target(415.0);
    let (_, d_a4) = sm_a4.step_block_with_dirty(128, 48_000.0).expect("dirty");
    assert_eq!(d_a4, DIRTY_TUNING);

    sm.set_target(0.85, 1.25, 1.0, 1.0);
    let (_, d_u) = sm.step_block_with_dirty(128, 48_000.0).expect("dirty");
    assert_ne!(d_u & DIRTY_TUNING, 0);

    // Temperament change triggers DIRTY_TUNING immediately and returns to idle next block.
    let mut sm_temp = VoicingSmoother::new();
    sm_temp.set_temperament(Temperament::KirnbergerIII);
    let (spec_t, d_t) = sm_temp.step_block_with_dirty(128, 48_000.0).expect("dirty");
    assert_eq!(d_t, DIRTY_TUNING);
    assert_eq!(spec_t.temperament, Temperament::KirnbergerIII);
    assert!(!sm_temp.is_active());
    assert!(sm_temp.step_block_with_dirty(128, 48_000.0).is_none());

    // Condition change triggers both DIRTY_TUNING and DIRTY_HAMMER and clamps [0.0, 1.0].
    let mut sm_cond = VoicingSmoother::new();
    sm_cond.set_condition_target(2.5);
    let (spec_c, d_c) = sm_cond.step_block_with_dirty(128, 48_000.0).expect("dirty");
    assert_eq!(d_c, DIRTY_TUNING | DIRTY_HAMMER);
    assert!(spec_c.condition > 0.0 && spec_c.condition <= 1.0);
}
