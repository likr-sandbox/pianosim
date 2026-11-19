//! Permanent `#[global_allocator]` regression test proving zero heap allocations
//! across real-time `PianoEngine` paths (`render_block`, `set_perf_profile_runtime`,
//! 11-block `VoicingCommitState`, and `SpaceBlob` chunked staging/commit).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use piano_engine::{
    embedded_body_bytes, BodyArchetype, PerfProfile, PianoEngine, SpacePrepContext,
    VoicingPrepContext,
};
use piano_spec::{
    u7_to_u16, EventKind, LidState, ListenerPerspective, PedalKind, SampleEvent, Temperament,
    VoicingSpec,
};

struct CountingAlloc;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL_ALLOC: CountingAlloc = CountingAlloc;

// SAFETY: Delegates all allocation operations directly to `std::alloc::System` and only
// updates lock-free atomics when `COUNTING` is enabled around real-time audio paths.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

fn measure_zero_alloc(label: &str, f: impl FnOnce()) {
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    f();
    COUNTING.store(false, Ordering::SeqCst);
    let allocs = ALLOC_COUNT.load(Ordering::SeqCst);
    assert_eq!(allocs, 0, "{label} performed {allocs} heap allocations");
}

#[test]
fn rt_paths_perform_zero_heap_allocations() {
    let sr = 48_000.0;
    let mut eng = PianoEngine::new_with_profile(sr, VoicingSpec::default(), PerfProfile::full());

    let alt_voicing = VoicingSpec::from_globals_full(
        0.78,
        1.45,
        1.15,
        0.97,
        432.0,
        Temperament::KirnbergerIII,
        0.25,
    );
    let mut vc_prep = VoicingPrepContext::new_with_profile(sr, &PerfProfile::full());
    let vc_blob = vc_prep.prepare_voicing(&alt_voicing).to_vec();

    let mut sp_prep = SpacePrepContext::new_with_profile(f64::from(sr), &PerfProfile::full());
    sp_prep
        .set_body_preset_bytes(embedded_body_bytes(BodyArchetype::UprightStudio))
        .expect("embedded body decodes");
    let sp_blob = sp_prep
        .prepare_space(ListenerPerspective::ConcertHall, LidState::Half)
        .to_vec();

    let mut l = [0.0_f32; 128];
    let mut r = [0.0_f32; 128];

    // 1. render_block with NoteOn, Pedal (Sustain/UnaCorda/Sostenuto), and NoteOff.
    measure_zero_alloc("render_block note/pedal/release", || {
        let on_evs = [
            SampleEvent {
                sample_offset: 0,
                kind: EventKind::Pedal {
                    kind: PedalKind::Sustain,
                    value_u16: u16::MAX / 2,
                },
            },
            SampleEvent {
                sample_offset: 8,
                kind: EventKind::NoteOn {
                    midi_key: 48,
                    velocity_u16: u7_to_u16(96),
                },
            },
            SampleEvent {
                sample_offset: 32,
                kind: EventKind::NoteOn {
                    midi_key: 64,
                    velocity_u16: u7_to_u16(100),
                },
            },
        ];
        eng.render_block(&on_evs, &mut l, &mut r);
        for _ in 0..8 {
            eng.render_block(&[], &mut l, &mut r);
        }
        let off_evs = [
            SampleEvent {
                sample_offset: 16,
                kind: EventKind::NoteOff {
                    midi_key: 48,
                    release_vel_u16: 32_768,
                },
            },
            SampleEvent {
                sample_offset: 64,
                kind: EventKind::Pedal {
                    kind: PedalKind::Sustain,
                    value_u16: 0,
                },
            },
        ];
        eng.render_block(&off_evs, &mut l, &mut r);
    });
    assert!(l.iter().chain(r.iter()).all(|x| x.is_finite()));

    // 2. set_perf_profile_runtime switching across Full / Realtime / Balanced / Full.
    measure_zero_alloc("set_perf_profile_runtime", || {
        let _ = eng.set_perf_profile_runtime(PerfProfile::realtime());
        eng.render_block(&[], &mut l, &mut r);
        let _ = eng.set_perf_profile_runtime(PerfProfile::balanced());
        eng.render_block(&[], &mut l, &mut r);
        let _ = eng.set_perf_profile_runtime(PerfProfile::full());
        eng.render_block(&[], &mut l, &mut r);
    });

    // 3. 11-block VoicingCommitState + concurrent set_hammer_hardness.
    measure_zero_alloc("VoicingCommitState 11-block commit", || {
        assert!(eng.begin_voicing_commit(alt_voicing.clone(), &vc_blob));
        eng.set_hammer_hardness(0.85);
        while eng.is_voicing_applying() {
            eng.render_block(&[], &mut l, &mut r);
        }
    });
    assert!(!eng.is_voicing_applying());

    // 4. SpaceBlob chunked staging + commit + crossfade rendering.
    measure_zero_alloc("SpaceBlob chunk staging and commit", || {
        let mut off = 0_usize;
        while off < sp_blob.len() {
            let n = 2_048.min(sp_blob.len() - off);
            assert!(eng.stage_precomputed_body_space_chunk(
                BodyArchetype::UprightStudio,
                ListenerPerspective::ConcertHall,
                LidState::Half,
                off,
                &sp_blob[off..off + n],
                sp_blob.len(),
            ));
            off += n;
        }
        assert!(eng.commit_precomputed_body_space(
            BodyArchetype::UprightStudio,
            ListenerPerspective::ConcertHall,
            LidState::Half,
        ));
        for _ in 0..8 {
            eng.render_block(&[], &mut l, &mut r);
        }
    });
    assert_eq!(eng.body_archetype(), BodyArchetype::UprightStudio);
    assert!(l.iter().chain(r.iter()).all(|x| x.is_finite()));
}
