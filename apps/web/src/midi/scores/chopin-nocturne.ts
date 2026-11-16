/** Procedural Chopin Nocturne in E-flat Major (Op. 9 No. 2) — ~28 seconds. */

import type { MidiCcEvent, MidiNoteEvent, MidiPlaybackEvent, ParsedSmf } from '../smf-parser';
import { sortEvents } from './types';

export function createChopinNocturne(): ParsedSmf {
  const events: MidiPlaybackEvent[] = [];
  const beatSec = 0.454; // ~132 BPM for 8th note in 12/8 time
  const ticksPerBeat = 240;

  let time = 0.0;
  let tick = 0;

  const measures = [
    // Measure 1: Eb major
    { bass: 39, chord: [51, 55, 58], melody: [70, 70, 72, 70, 67, 63] },
    // Measure 2: C minor / Ab major
    { bass: 36, chord: [48, 51, 55], melody: [65, 67, 68, 70, 68, 65] },
    // Measure 3: Fm7 -> Bb7
    { bass: 41, chord: [50, 53, 56], melody: [63, 65, 67, 65, 62, 58] },
    // Measure 4: Eb major cadence
    { bass: 39, chord: [51, 55, 58], melody: [63, 67, 70, 72, 75, 70] },
    // Measure 5: High lyrical variation
    { bass: 39, chord: [51, 55, 58], melody: [77, 75, 74, 75, 72, 70] },
  ];

  for (const m of measures) {
    // Damper pedal down at start of measure
    events.push({
      type: 'cc',
      timeSeconds: time,
      tick,
      channel: 0,
      controller: 64,
      value: 127,
    } as MidiCcEvent);

    // Left hand bass on beat 1
    events.push({
      type: 'noteOn',
      timeSeconds: time,
      tick,
      channel: 0,
      midiKey: m.bass,
      velocity: 96,
    } as MidiNoteEvent);
    events.push({
      type: 'noteOff',
      timeSeconds: time + beatSec * 0.9,
      tick: tick + Math.trunc(ticksPerBeat * 0.9),
      channel: 0,
      midiKey: m.bass,
      velocity: 40,
    } as MidiNoteEvent);

    // Accompaniment chords on beats 2, 3, 5, 6
    const chordBeats = [1, 2, 4, 5];
    for (const cb of chordBeats) {
      const cTime = time + cb * beatSec;
      const cTick = tick + cb * ticksPerBeat;
      for (const k of m.chord) {
        events.push({
          type: 'noteOn',
          timeSeconds: cTime,
          tick: cTick,
          channel: 0,
          midiKey: k,
          velocity: 72,
        } as MidiNoteEvent);
        events.push({
          type: 'noteOff',
          timeSeconds: cTime + beatSec * 0.8,
          tick: cTick + Math.trunc(ticksPerBeat * 0.8),
          channel: 0,
          midiKey: k,
          velocity: 40,
        } as MidiNoteEvent);
      }
    }

    // Right hand melody notes (6 notes per measure)
    for (let mi = 0; mi < m.melody.length; mi++) {
      const mTime = time + mi * beatSec;
      const mTick = tick + mi * ticksPerBeat;
      const melKey = m.melody[mi];
      events.push({
        type: 'noteOn',
        timeSeconds: mTime,
        tick: mTick,
        channel: 0,
        midiKey: melKey,
        velocity: 96,
      } as MidiNoteEvent);
      events.push({
        type: 'noteOff',
        timeSeconds: mTime + beatSec * 0.95,
        tick: mTick + Math.trunc(ticksPerBeat * 0.95),
        channel: 0,
        midiKey: melKey,
        velocity: 50,
      } as MidiNoteEvent);
    }

    // Pedal release and re-catch before measure ends
    events.push({
      type: 'cc',
      timeSeconds: time + beatSec * 5.8,
      tick: tick + Math.trunc(ticksPerBeat * 5.8),
      channel: 0,
      controller: 64,
      value: 0,
    } as MidiCcEvent);

    time += beatSec * 6.0;
    tick += ticksPerBeat * 6;
  }

  sortEvents(events);

  return {
    format: 0,
    trackCount: 1,
    division: 480,
    durationSeconds: time + 1.5,
    events,
  };
}
