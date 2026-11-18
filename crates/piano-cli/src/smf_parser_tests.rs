//! Unit tests for the pure-Rust Standard MIDI File parser.

use std::path::Path;

use super::*;
use crate::check_lines::find_repo_root;

fn make_chunk(tag: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + payload.len());
    out.extend_from_slice(tag);
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    out.extend_from_slice(payload);
    out
}

fn make_mthd(format: u16, tracks: u16, division: u16) -> Vec<u8> {
    let mut body = Vec::with_capacity(6);
    body.extend_from_slice(&format.to_be_bytes());
    body.extend_from_slice(&tracks.to_be_bytes());
    body.extend_from_slice(&division.to_be_bytes());
    make_chunk(b"MThd", &body)
}

#[test]
fn parses_format0_running_status_noteoff_vel0_pedals_and_tiebreak() {
    // division = 480, default 120 BPM (500_000 uspqn -> 960 ticks/sec)
    let mut trk = Vec::new();
    // Tick 0: NoteOn C4 vel=100, then running status NoteOn E4 vel=90, CC64=64, CC66=100, CC67=32
    trk.extend_from_slice(&[0x00, 0x90, 60, 100]);
    trk.extend_from_slice(&[0x00, 64, 90]); // running status 0x90
    trk.extend_from_slice(&[0x00, 0xB0, 64, 64]);
    trk.extend_from_slice(&[0x00, 66, 100]); // running status 0xB0
    trk.extend_from_slice(&[0x00, 67, 32]); // running status 0xB0
                                            // Tick 480 (0x83 0x60 = 480 -> 0.5 s): NoteOn C4 vel=110 followed in stream by NoteOff C4 vel=40
                                            // Tie-break must put NoteOff before NoteOn!
    trk.extend_from_slice(&[0x83, 0x60, 0x90, 60, 110]);
    trk.extend_from_slice(&[0x00, 0x80, 60, 40]);
    // Tick 960 (delta 480 -> 1.0 s): NoteOn E4 vel=0 (maps to NoteOff release=0)
    trk.extend_from_slice(&[0x83, 0x60, 0x90, 64, 0]);
    trk.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

    let mut smf = make_mthd(0, 1, 480);
    smf.extend_from_slice(&make_chunk(b"MTrk", &trk));

    let parsed = parse_smf_bytes(&smf, 48_000).expect("valid format 0");
    assert_eq!(parsed.format, 0);
    assert_eq!(parsed.track_count, 1);
    assert_eq!(parsed.division, 480);
    assert!((parsed.last_event_s - 1.0).abs() < 1e-6);
    assert_eq!(parsed.events.len(), 8);

    // Tick 0 tie-break: Pedals (prio 1) before NoteOns (prio 2)
    assert_eq!(
        parsed.events[0],
        (
            0,
            EventKind::Pedal {
                kind: PedalKind::Sustain,
                value_u16: u7_to_u16(64)
            }
        )
    );
    assert_eq!(
        parsed.events[1],
        (
            0,
            EventKind::Pedal {
                kind: PedalKind::Sostenuto,
                value_u16: u16::MAX
            }
        )
    );
    assert_eq!(
        parsed.events[2],
        (
            0,
            EventKind::Pedal {
                kind: PedalKind::UnaCorda,
                value_u16: u7_to_u16(32)
            }
        )
    );
    assert_eq!(
        parsed.events[3],
        (
            0,
            EventKind::NoteOn {
                midi_key: 60,
                velocity_u16: u7_to_u16(100)
            }
        )
    );
    assert_eq!(
        parsed.events[4],
        (
            0,
            EventKind::NoteOn {
                midi_key: 64,
                velocity_u16: u7_to_u16(90)
            }
        )
    );
    // Tick 480 = 24_000 samples: NoteOff (prio 0) before NoteOn (prio 2)
    assert_eq!(
        parsed.events[5],
        (
            24_000,
            EventKind::NoteOff {
                midi_key: 60,
                release_vel_u16: u7_to_u16(40)
            }
        )
    );
    assert_eq!(
        parsed.events[6],
        (
            24_000,
            EventKind::NoteOn {
                midi_key: 60,
                velocity_u16: u7_to_u16(110)
            }
        )
    );
    // Tick 960 = 48_000 samples: 0x90 vel=0 -> NoteOff with release_vel=0
    assert_eq!(
        parsed.events[7],
        (
            48_000,
            EventKind::NoteOff {
                midi_key: 64,
                release_vel_u16: 0
            }
        )
    );
}

#[test]
fn parses_format1_multi_track_with_tempo_changes() {
    // division = 1000.
    // Track 0: tempo 500_000 uspqn at tick 0; tempo 1_000_000 uspqn (60 BPM) at tick 1000 (t = 0.5 s).
    let mut trk0 = Vec::new();
    trk0.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]); // 500_000
    trk0.extend_from_slice(&[0x87, 0x68, 0xFF, 0x51, 0x03, 0x0F, 0x42, 0x40]); // delta 1000 -> 1_000_000
    trk0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

    // Track 1: NoteOn at tick 1000 (0.5 s), NoteOff at tick 2000 (0.5 + 1.0 = 1.5 s)
    let mut trk1 = Vec::new();
    trk1.extend_from_slice(&[0x87, 0x68, 0x90, 69, 100]); // tick 1000
    trk1.extend_from_slice(&[0x87, 0x68, 0x80, 69, 34]); // tick 2000
    trk1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

    let mut smf = make_mthd(1, 2, 1000);
    smf.extend_from_slice(&make_chunk(b"MTrk", &trk0));
    smf.extend_from_slice(&make_chunk(b"MTrk", &trk1));

    let parsed = parse_smf_bytes(&smf, 48_000).expect("valid format 1");
    assert_eq!(parsed.format, 1);
    assert_eq!(parsed.track_count, 2);
    assert!((parsed.last_event_s - 1.5).abs() < 1e-6);
    assert_eq!(parsed.events[0].0, 24_000);
    assert_eq!(parsed.events[1].0, 72_000);
}

#[test]
fn rejects_corrupt_and_unsupported_smf_inputs() {
    assert!(parse_smf_bytes(b"short", 48_000).is_err());
    assert!(parse_smf_bytes(b"BAD!\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0", 48_000).is_err());
    // Format 2 unsupported
    assert!(parse_smf_bytes(&make_mthd(2, 1, 480), 48_000).is_err());
    // SMPTE division unsupported
    assert!(parse_smf_bytes(&make_mthd(0, 1, 0xE728), 48_000).is_err());

    // Running status without prior status
    let mut bad_rs = make_mthd(0, 1, 480);
    bad_rs.extend_from_slice(&make_chunk(b"MTrk", &[0x00, 0x3C, 0x40]));
    assert!(parse_smf_bytes(&bad_rs, 48_000).is_err());

    // VLQ exceeding 4 bytes
    let mut bad_vlq = make_mthd(0, 1, 480);
    bad_vlq.extend_from_slice(&make_chunk(b"MTrk", &[0x81, 0x81, 0x81, 0x81, 0x00]));
    assert!(parse_smf_bytes(&bad_vlq, 48_000).is_err());
}

#[test]
fn parses_all_six_canonical_demo_mids_and_verifies_bach_beat3_bass_fix() {
    let cwd = std::env::current_dir().expect("cwd");
    let root = find_repo_root(&cwd);
    let demos_dir = root.join("apps/web/public/demos");
    let files = [
        "01-bach-prelude-bwv846.mid",
        "02-chopin-nocturne-op9-no2.mid",
        "03-beethoven-moonlight-op27-no2.mid",
        "04-debussy-clair-de-lune.mid",
        "05-satie-gymnopedie-no1.mid",
        "06-rachmaninoff-prelude-op3-no2.mid",
    ];

    for file in files {
        let path = demos_dir.join(file);
        assert!(path.is_file(), "missing canonical demo {}", path.display());
        let parsed = parse_smf_file(&path, 48_000).expect(file);
        assert_eq!(parsed.format, 0, "{file}");
        assert_eq!(parsed.track_count, 1, "{file}");
        assert_eq!(parsed.division, 1000, "{file}");
        assert!(parsed.last_event_s > 13.0, "{file}");
        assert!(parsed.events.len() >= 40, "{file}");
        for w in parsed.events.windows(2) {
            let (s0, k0) = w[0];
            let (s1, k1) = w[1];
            assert!(
                s0 < s1 || (s0 == s1 && event_priority(k0) <= event_priority(k1)),
                "{file}: out of order at sample {s0} vs {s1}"
            );
        }
    }

    // Verify Bach Prelude beat-3 (idx == 8) bass notes hold >= 0.89 s and release with off_vel=34
    let bach = parse_smf_file(Path::new(&demos_dir.join(files[0])), 48_000).expect("bach");
    let bass_keys: [u8; 16] = [
        48, 48, 47, 48, 48, 48, 47, 47, 45, 38, 43, 43, 41, 41, 40, 36,
    ];
    for (m, &bass_key) in bass_keys.iter().enumerate() {
        let beat3_on_approx = ((m as f64 * 2.0 + 1.004) * 48_000.0).round() as usize;
        let on_idx = bach
            .events
            .iter()
            .position(|&(s, k)| {
                s.abs_diff(beat3_on_approx) <= 48
                    && matches!(k, EventKind::NoteOn { midi_key, .. } if midi_key == bass_key)
            })
            .unwrap_or_else(|| panic!("missing measure {m} beat-3 NoteOn for key {bass_key}"));
        let on_sample = bach.events[on_idx].0;
        let off_ev = bach.events[on_idx + 1..]
            .iter()
            .find(
                |&&(_, k)| matches!(k, EventKind::NoteOff { midi_key, .. } if midi_key == bass_key),
            )
            .unwrap_or_else(|| panic!("missing measure {m} beat-3 NoteOff for key {bass_key}"));
        let dur_s = (off_ev.0 - on_sample) as f64 / 48_000.0;
        assert!(
            dur_s >= 0.89,
            "measure {m} beat-3 bass key {bass_key} held only {dur_s:.4}s (< 0.89s)"
        );
        assert_eq!(
            off_ev.1,
            EventKind::NoteOff {
                midi_key: bass_key,
                release_vel_u16: u7_to_u16(34),
            },
            "measure {m} beat-3 release velocity mismatch"
        );
    }
}
