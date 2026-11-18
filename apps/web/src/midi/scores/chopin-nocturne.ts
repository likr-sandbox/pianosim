/** Procedural Chopin Nocturne in E-flat Major (Op. 9 No. 2) — humanized rubato, rolled chords & half-pedaling. */

import type { MidiPlaybackEvent, ParsedSmf } from '../smf-parser';
import { addCc, addNote, addRolledChord, sortEvents } from './types';

export function createChopinNocturne(): ParsedSmf {
  const events: MidiPlaybackEvent[] = [];
  const beatSec = 0.454; // ~132 BPM for 8th note in 12/8 time
  let time = 0.0;

  const measures = [
    { bass: 39, chord: [51, 55, 58], melody: [70, 70, 72, 70, 67, 63] },
    { bass: 36, chord: [48, 51, 55], melody: [65, 67, 68, 70, 68, 65] },
    { bass: 41, chord: [50, 53, 56], melody: [63, 65, 67, 65, 62, 58] },
    { bass: 39, chord: [51, 55, 58], melody: [63, 67, 70, 72, 75, 70] },
    { bass: 39, chord: [51, 55, 58], melody: [77, 75, 74, 75, 72, 70] },
  ];

  for (let mi = 0; mi < measures.length; mi++) {
    const m = measures[mi];
    // Continuous Chopin half-pedal: deeper on bass beat 1 (92), lighter flutter on beat 4 (48 -> 84)
    addCc(events, time + 0.01, 64, 92);
    addCc(events, time + beatSec * 2.85, 64, 46);
    addCc(events, time + beatSec * 3.05, 64, 84);

    // Warm left-hand bass root on beat 1
    addNote(events, time, beatSec * 0.92, m.bass, 86, 36);

    // Rolled inner accompaniment chords (gentle 11 ms micro-roll, subdued 56..64 velocity)
    const chordBeats = [1, 2, 4, 5];
    for (const cb of chordBeats) {
      const cTime = time + cb * beatSec;
      const baseVel = cb === 1 || cb === 4 ? 62 : 56;
      addRolledChord(events, cTime, beatSec * 0.82, m.chord, baseVel, 0.011, 4, 34);
    }

    // Expressive singing right-hand bel canto melody (vel 90..104) with rubato nuance
    for (let i = 0; i < m.melody.length; i++) {
      const rubato = i === 0 ? 0 : Math.sin((i / 5) * Math.PI) * 0.010;
      const mTime = time + i * beatSec + rubato;
      const contourVel = 92 + Math.round(Math.sin((i / 5) * Math.PI) * 12) + (mi === 4 ? 4 : 0);
      addNote(events, mTime, beatSec * 0.96, m.melody[i], contourVel, 46);
    }

    // Half-pedal lift before harmonic resolution
    addCc(events, time + beatSec * 5.75, 64, 40);
    addCc(events, time + beatSec * 5.88, 64, 0);

    time += beatSec * 6.0;
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
