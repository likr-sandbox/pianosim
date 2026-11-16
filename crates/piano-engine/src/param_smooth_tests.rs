use super::*;

#[test]
fn voicing_smoother_settles_and_idles() {
    let mut sm = VoicingSmoother::new();
    sm.set_target(0.8, 1.5, 1.2, 0.9);
    let sr = 48_000.0;
    let mut steps = 0;
    while sm.is_active() && steps < 1_000 {
        let spec = sm.step_block(128, sr).expect("spec expected");
        assert!(spec.hammer_hardness >= 0.5 && spec.hammer_hardness <= 0.8);
        steps += 1;
    }
    assert!(!sm.is_active());
    assert_eq!(sm.current_hammer_hardness(), 0.8);
    assert_eq!(sm.current_unison_detune(), 1.5);
    assert_eq!(sm.current_stretch(), 1.2);
    assert_eq!(sm.current_length_scale(), 0.9);
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
fn voicing_dirty_mask_isolates_hammer_from_tuning() {
    let mut sm = VoicingSmoother::new();
    sm.set_target(0.85, 1.0, 1.0, 1.0);
    let (_, d_h) = sm.step_block_with_dirty(128, 48_000.0).expect("dirty");
    assert_eq!(d_h, DIRTY_HAMMER);
    sm.set_target(0.85, 1.25, 1.0, 1.0);
    let (_, d_u) = sm.step_block_with_dirty(128, 48_000.0).expect("dirty");
    assert_ne!(d_u & DIRTY_TUNING, 0);
}
