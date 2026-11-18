//! Binary Standard MIDI File (SMF) parser supporting Format 0 & 1 with VLQ, running status, and tempo maps.

use std::cmp::Ordering;
use std::fs;
use std::path::Path;

use piano_spec::{u7_to_u16, EventKind, PedalKind};

const DEFAULT_US_PER_QUARTER: u32 = 500_000;

/// Sample-quantized events and metadata parsed from a Standard MIDI File.
#[derive(Clone, Debug)]
pub struct ParsedSmfScore {
    #[cfg_attr(not(test), allow(dead_code))]
    pub format: u16,
    #[cfg_attr(not(test), allow(dead_code))]
    pub track_count: u16,
    #[cfg_attr(not(test), allow(dead_code))]
    pub division: u16,
    pub last_event_s: f64,
    pub events: Vec<(usize, EventKind)>,
}

#[derive(Copy, Clone, Debug)]
struct RawEvent {
    tick: u64,
    order: usize,
    kind: EventKind,
}

#[derive(Copy, Clone, Debug)]
struct TempoPoint {
    tick: u64,
    time_s: f64,
    uspqn: u32,
}

struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    const fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    const fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn seek(&mut self, pos: usize) -> Result<(), String> {
        if pos > self.data.len() {
            return Err(String::from("invalid SMF: seek out of bounds"));
        }
        self.pos = pos;
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        let &b = self
            .data
            .get(self.pos)
            .ok_or_else(|| String::from("invalid SMF: unexpected end of stream"))?;
        self.pos += 1;
        Ok(b)
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        let b0 = self.read_u8()?;
        let b1 = self.read_u8()?;
        Ok(u16::from_be_bytes([b0, b1]))
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let b0 = self.read_u8()?;
        let b1 = self.read_u8()?;
        let b2 = self.read_u8()?;
        let b3 = self.read_u8()?;
        Ok(u32::from_be_bytes([b0, b1, b2, b3]))
    }

    fn read_vlq(&mut self) -> Result<u32, String> {
        let mut val = 0_u32;
        for _ in 0..4 {
            let b = self.read_u8()?;
            val = (val << 7) | u32::from(b & 0x7F);
            if (b & 0x80) == 0 {
                return Ok(val);
            }
        }
        Err(String::from(
            "invalid SMF: corrupted VLQ delta time (exceeds 4 bytes)",
        ))
    }

    fn skip(&mut self, n: usize) -> Result<(), String> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| String::from("invalid SMF: truncated chunk"))?;
        self.pos = end;
        Ok(())
    }
}

#[inline]
const fn event_priority(kind: EventKind) -> u8 {
    match kind {
        EventKind::NoteOff { .. } => 0,
        EventKind::Pedal { .. } => 1,
        EventKind::NoteOn { .. } => 2,
        EventKind::AllNotesOff => 3,
    }
}

/// Reads and parses an SMF file on disk into sample-quantized events at `sample_rate`.
pub fn parse_smf_file(path: &Path, sample_rate: u32) -> Result<ParsedSmfScore, String> {
    let bytes =
        fs::read(path).map_err(|e| format!("failed to read MIDI '{}': {e}", path.display()))?;
    parse_smf_bytes(&bytes, sample_rate)
}

/// Parses binary Standard MIDI File bytes (Format 0 or 1) at `sample_rate`.
pub fn parse_smf_bytes(bytes: &[u8], sample_rate: u32) -> Result<ParsedSmfScore, String> {
    if sample_rate == 0 {
        return Err(String::from("invalid sample rate 0"));
    }
    if bytes.len() < 14 {
        return Err(String::from(
            "invalid SMF: file too small for MThd header chunk",
        ));
    }

    let mut reader = ByteReader::new(bytes);
    let magic = [
        reader.read_u8()?,
        reader.read_u8()?,
        reader.read_u8()?,
        reader.read_u8()?,
    ];
    if &magic != b"MThd" {
        return Err(String::from("invalid SMF: missing MThd header chunk"));
    }

    let header_len = reader.read_u32()? as usize;
    if header_len < 6 {
        return Err(format!(
            "invalid SMF: MThd length must be >= 6, got {header_len}"
        ));
    }

    let format = reader.read_u16()?;
    if format > 1 {
        return Err(format!(
            "unsupported SMF format {format}: only Format 0 and 1 are supported"
        ));
    }

    let track_count = reader.read_u16()?;
    let division_raw = reader.read_u16()?;
    if (division_raw & 0x8000) != 0 {
        return Err(String::from(
            "unsupported SMF: SMPTE time code division is not supported",
        ));
    }
    let division = if division_raw == 0 { 480 } else { division_raw };
    if header_len > 6 {
        reader.skip(header_len - 6)?;
    }

    let mut raw_events = Vec::new();
    let mut tempo_changes = vec![(0_u64, DEFAULT_US_PER_QUARTER)];
    let mut tracks_parsed = 0_u16;

    while tracks_parsed < track_count && reader.remaining() >= 8 {
        let chunk_magic = [
            reader.read_u8()?,
            reader.read_u8()?,
            reader.read_u8()?,
            reader.read_u8()?,
        ];
        let chunk_len = reader.read_u32()? as usize;
        if &chunk_magic != b"MTrk" {
            reader.skip(chunk_len)?;
            continue;
        }

        let track_end = reader
            .pos
            .checked_add(chunk_len)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| String::from("invalid SMF: truncated track chunk"))?;
        parse_track(
            &mut reader,
            bytes,
            track_end,
            &mut raw_events,
            &mut tempo_changes,
        )?;
        tracks_parsed += 1;
        reader.seek(track_end)?;
    }

    if tracks_parsed < track_count {
        return Err(format!(
            "invalid SMF: expected {track_count} tracks, found {tracks_parsed}"
        ));
    }

    let tempo_points = build_tempo_points(&mut tempo_changes, division);
    let sr = f64::from(sample_rate);
    let mut last_event_s = 0.0_f64;
    let mut quantized: Vec<(usize, u8, usize, EventKind)> = Vec::with_capacity(raw_events.len());

    for ev in raw_events {
        let time_s = tick_to_seconds(&tempo_points, division, ev.tick);
        if time_s > last_event_s {
            last_event_s = time_s;
        }
        let sample = (time_s.max(0.0) * sr).round() as usize;
        quantized.push((sample, event_priority(ev.kind), ev.order, ev.kind));
    }

    quantized.sort_by(|a, b| match a.0.cmp(&b.0) {
        Ordering::Equal => match a.1.cmp(&b.1) {
            Ordering::Equal => a.2.cmp(&b.2),
            other => other,
        },
        other => other,
    });

    let events = quantized.into_iter().map(|(s, _, _, k)| (s, k)).collect();
    Ok(ParsedSmfScore {
        format,
        track_count,
        division,
        last_event_s,
        events,
    })
}

fn parse_track(
    reader: &mut ByteReader<'_>,
    bytes: &[u8],
    track_end: usize,
    raw_events: &mut Vec<RawEvent>,
    tempo_changes: &mut Vec<(u64, u32)>,
) -> Result<(), String> {
    let mut current_tick = 0_u64;
    let mut running_status = 0_u8;

    while reader.pos < track_end {
        let delta = u64::from(reader.read_vlq()?);
        current_tick = current_tick.saturating_add(delta);
        if reader.pos >= track_end {
            break;
        }

        let peek = bytes[reader.pos];
        let status = if (peek & 0x80) != 0 {
            let s = reader.read_u8()?;
            running_status = if s < 0xF0 { s } else { 0 };
            s
        } else {
            if running_status == 0 {
                return Err(String::from(
                    "invalid SMF: running status without previous status",
                ));
            }
            running_status
        };

        if status == 0xFF {
            let meta_type = reader.read_u8()?;
            let meta_len = reader.read_vlq()? as usize;
            if meta_type == 0x51 && meta_len == 3 {
                let b0 = reader.read_u8()?;
                let b1 = reader.read_u8()?;
                let b2 = reader.read_u8()?;
                let uspqn = u32::from_be_bytes([0, b0, b1, b2]);
                tempo_changes.push((
                    current_tick,
                    if uspqn > 0 {
                        uspqn
                    } else {
                        DEFAULT_US_PER_QUARTER
                    },
                ));
            } else {
                reader.skip(meta_len)?;
            }
        } else if status == 0xF0 || status == 0xF7 {
            let sysex_len = reader.read_vlq()? as usize;
            reader.skip(sysex_len)?;
        } else {
            parse_channel_event(reader, status, current_tick, raw_events)?;
        }
    }
    Ok(())
}

fn parse_channel_event(
    reader: &mut ByteReader<'_>,
    status: u8,
    tick: u64,
    raw_events: &mut Vec<RawEvent>,
) -> Result<(), String> {
    let order = raw_events.len();
    match status & 0xF0 {
        0x80 => {
            let key = reader.read_u8()? & 0x7F;
            let vel = reader.read_u8()? & 0x7F;
            raw_events.push(RawEvent {
                tick,
                order,
                kind: EventKind::NoteOff {
                    midi_key: key,
                    release_vel_u16: u7_to_u16(vel),
                },
            });
        }
        0x90 => {
            let key = reader.read_u8()? & 0x7F;
            let vel = reader.read_u8()? & 0x7F;
            let kind = if vel == 0 {
                EventKind::NoteOff {
                    midi_key: key,
                    release_vel_u16: 0,
                }
            } else {
                EventKind::NoteOn {
                    midi_key: key,
                    velocity_u16: u7_to_u16(vel),
                }
            };
            raw_events.push(RawEvent { tick, order, kind });
        }
        0xB0 => {
            let cc = reader.read_u8()? & 0x7F;
            let val = reader.read_u8()? & 0x7F;
            let pedal = match cc {
                64 => Some((PedalKind::Sustain, u7_to_u16(val))),
                66 => Some((PedalKind::Sostenuto, if val >= 64 { u16::MAX } else { 0 })),
                67 => Some((PedalKind::UnaCorda, u7_to_u16(val))),
                _ => None,
            };
            if let Some((kind, value_u16)) = pedal {
                raw_events.push(RawEvent {
                    tick,
                    order,
                    kind: EventKind::Pedal { kind, value_u16 },
                });
            }
        }
        0xA0 | 0xE0 => reader.skip(2)?,
        0xC0 | 0xD0 => reader.skip(1)?,
        _ => {}
    }
    Ok(())
}

fn build_tempo_points(tempo_changes: &mut [(u64, u32)], division: u16) -> Vec<TempoPoint> {
    tempo_changes.sort_by_key(|&(tick, _)| tick);
    let mut points = Vec::with_capacity(tempo_changes.len());
    let mut current_s = 0.0_f64;
    let mut last_tick = 0_u64;
    let mut current_uspqn = DEFAULT_US_PER_QUARTER;
    let div = f64::from(division);

    for &(tick, uspqn) in tempo_changes.iter() {
        if tick > last_tick {
            let dt = (tick - last_tick) as f64;
            current_s += (dt * f64::from(current_uspqn)) / (div * 1_000_000.0);
            last_tick = tick;
        }
        current_uspqn = uspqn;
        points.push(TempoPoint {
            tick,
            time_s: current_s,
            uspqn: current_uspqn,
        });
    }
    points
}

fn tick_to_seconds(points: &[TempoPoint], division: u16, tick: u64) -> f64 {
    let mut active = points[0];
    for &p in &points[1..] {
        if p.tick <= tick {
            active = p;
        } else {
            break;
        }
    }
    let dt = (tick - active.tick) as f64;
    active.time_s + (dt * f64::from(active.uspqn)) / (f64::from(division) * 1_000_000.0)
}

#[cfg(test)]
#[path = "smf_parser_tests.rs"]
mod tests;
