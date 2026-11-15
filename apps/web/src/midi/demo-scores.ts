/** Compact procedural demo scores for Bach Prelude in C (BWV 846) and Chopin Nocturne (Op. 9 No. 2). */

import type { MidiCcEvent, MidiNoteEvent, MidiPlaybackEvent, ParsedSmf } from './smf-parser';

/** Procedural Bach Prelude in C Major (BWV 846) — 16 measures, 32 seconds. */
export function createBachPrelude(): ParsedSmf {
  const chords: number[][] = [
    [48, 52, 55, 60, 64], // C major
    [48, 50, 57, 62, 65], // Dm7/C
    [47, 50, 55, 62, 65], // G7/B
    [48, 52, 55, 60, 64], // C major
    [48, 52, 57, 64, 69], // Am/C
    [48, 50, 54, 57, 62], // D7/C
    [47, 50, 55, 62, 67], // G major
    [47, 48, 52, 55, 60], // Cmaj7/B
    [45, 48, 52, 55, 60], // Am7
    [38, 45, 50, 54, 60], // D7
    [43, 47, 50, 55, 59], // G major
    [43, 46, 52, 55, 61], // Gdim
    [41, 45, 50, 57, 62], // Dm/F
    [41, 44, 50, 53, 59], // G7/F
    [40, 43, 48, 55, 60], // C/E
    [36, 43, 48, 52, 55], // C major finale
  ];

  const events: MidiPlaybackEvent[] = [];
  const noteDuration = 0.125; // 16th note at 120 BPM = 125 ms
  const pattern = [0, 1, 2, 3, 4, 2, 3, 4, 0, 1, 2, 3, 4, 2, 3, 4];
  let time = 0.0;
  let tick = 0;
  const tickStep = 120; // 480 ticks/quarter => 120 ticks/16th

  for (const chord of chords) {
    for (const step of pattern) {
      const midiKey = chord[step];
      const velocity = step === 0 ? 92 : step === 1 ? 84 : 76;
      events.push({
        type: 'noteOn',
        timeSeconds: time,
        tick,
        channel: 0,
        midiKey,
        velocity,
      });
      events.push({
        type: 'noteOff',
        timeSeconds: time + 0.12,
        tick: tick + 115,
        channel: 0,
        midiKey,
        velocity: 64,
      });
      time += noteDuration;
      tick += tickStep;
    }
  }

  // Sort events chronologically, noteOff before noteOn on ties
  events.sort((a, b) => {
    if (Math.abs(a.timeSeconds - b.timeSeconds) > 1e-6) {
      return a.timeSeconds - b.timeSeconds;
    }
    return a.type === 'noteOff' ? -1 : 1;
  });

  return {
    format: 0,
    trackCount: 1,
    division: 480,
    durationSeconds: time + 1.0,
    events,
  };
}

/** Procedural Chopin Nocturne in E-flat Major (Op. 9 No. 2) — ~28 seconds. */
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
      velocity: 80,
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
          velocity: 60,
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
        velocity: 88,
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

  events.sort((a, b) => {
    if (Math.abs(a.timeSeconds - b.timeSeconds) > 1e-6) {
      return a.timeSeconds - b.timeSeconds;
    }
    const prioA = a.type === 'noteOff' ? 0 : a.type === 'cc' ? 1 : 2;
    const prioB = b.type === 'noteOff' ? 0 : b.type === 'cc' ? 1 : 2;
    return prioA - prioB;
  });

  return {
    format: 0,
    trackCount: 1,
    division: 480,
    durationSeconds: time + 1.5,
    events,
  };
}

export function getDemoScore(id: string): ParsedSmf {
  if (id === 'chopin-nocturne') {
    return createChopinNocturne();
  }
  return createBachPrelude();
}
