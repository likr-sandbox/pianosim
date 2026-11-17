/** Procedural Beethoven Moonlight Sonata (Op. 27 No. 2, 1st Mvt) — ~38 seconds.
 * Showcases deep C# minor bass octaves, sustained damper pedal wash, and massive bridge sympathetic resonance.
 */

import type { MidiPlaybackEvent, ParsedSmf } from '../smf-parser';
import { addCc, addNote, sortEvents } from './types';

export function createBeethovenMoonlight(): ParsedSmf {
  const events: MidiPlaybackEvent[] = [];
  const bpm = 54;
  const quarterSec = 60 / bpm; // ~1.111 s per beat
  const tripletSec = quarterSec / 3; // ~0.370 s per triplet 8th
  const measureSec = quarterSec * 4; // ~4.444 s per 4/4 measure

  let time = 0.0;

  // 8 measures of Adagio sostenuto in C# minor
  const scoreData = [
    // M1: Intro C# minor triplets
    { bass: [25, 37], trips: [56, 61, 64], melody: [] as Array<{ beat: number; key: number; dur: number; vel: number }> },
    // M2: Triplets continue
    { bass: [25, 37], trips: [56, 61, 64], melody: [] },
    // M3: Melody enters with dotted rhythm
    {
      bass: [25, 37],
      trips: [56, 61, 64],
      melody: [
        { beat: 2.66, key: 68, dur: 0.8, vel: 82 }, // G#4
        { beat: 3.66, key: 68, dur: 0.8, vel: 84 }, // G#4
      ],
    },
    // M4: Harmony shifts to B dominant
    {
      bass: [23, 35], // B0, B1
      trips: [56, 59, 64], // G#3, B3, E4
      melody: [
        { beat: 0.0, key: 68, dur: 2.2, vel: 86 }, // G#4 (tied)
        { beat: 2.0, key: 68, dur: 0.7, vel: 82 },
        { beat: 2.66, key: 69, dur: 0.7, vel: 84 }, // A4
        { beat: 3.33, key: 68, dur: 0.7, vel: 82 }, // G#4
      ],
    },
    // M5: A major / F# minor
    {
      bass: [21, 33], // A0, A1
      trips: [57, 61, 64], // A3, C#4, E4
      melody: [
        { beat: 0.0, key: 66, dur: 2.0, vel: 84 }, // F#4
        { beat: 2.0, key: 64, dur: 1.0, vel: 80 }, // E4
        { beat: 3.0, key: 63, dur: 1.0, vel: 82 }, // D#4
      ],
    },
    // M6: G# dominant 7th tension
    {
      bass: [32, 44], // G#1, G#2
      trips: [56, 59, 62], // G#3, B3, D4
      melody: [
        { beat: 0.0, key: 64, dur: 2.0, vel: 84 }, // E4
        { beat: 2.0, key: 63, dur: 2.0, vel: 82 }, // D#4
      ],
    },
    // M7: Cadence resolution to C# minor
    {
      bass: [25, 37], // C#1, C#2
      trips: [56, 61, 64],
      melody: [
        { beat: 0.0, key: 61, dur: 3.8, vel: 86 }, // C#4
      ],
    },
    // M8: Deep resonant stillness
    {
      bass: [25, 37],
      trips: [49, 56, 61], // Low C#3, G#3, C#4
      melody: [
        { beat: 0.0, key: 64, dur: 4.0, vel: 78 }, // E4
        { beat: 2.0, key: 73, dur: 4.0, vel: 80 }, // C#5
      ],
    },
  ];

  for (const m of scoreData) {
    // Continuous Damper Pedal (CC 64) wash: caught cleanly at measure boundaries
    addCc(events, time + 0.02, 64, 127);

    // Deep resonant bass octaves on beat 1
    for (const bKey of m.bass) {
      addNote(events, time, measureSec * 0.92, bKey, 88, 30);
    }

    // Triplet accompaniment: 12 notes per measure (4 beats * 3)
    for (let t = 0; t < 12; t++) {
      const tripKey = m.trips[t % 3];
      const tripTime = time + t * tripletSec;
      const tripVel = t % 3 === 0 ? 66 : 58;
      addNote(events, tripTime, tripletSec * 1.05, tripKey, tripVel, 30);
    }

    // Melodic notes
    for (const mel of m.melody) {
      const melTime = time + mel.beat * quarterSec;
      addNote(events, melTime, mel.dur, mel.key, mel.vel, 35);
    }

    // Half-pedal breath / catch right before next measure
    addCc(events, time + measureSec - 0.1, 64, 0);

    time += measureSec;
  }

  // Final sustained chord tail
  addCc(events, time, 64, 127);
  addNote(events, time + 0.1, 4.0, 25, 80, 20); // C#1
  addNote(events, time + 0.1, 4.0, 37, 82, 20); // C#2
  addNote(events, time + 0.1, 4.0, 49, 76, 20); // C#3
  addNote(events, time + 0.1, 4.0, 61, 74, 20); // C#4
  addNote(events, time + 0.1, 4.0, 64, 72, 20); // E4
  addNote(events, time + 0.1, 4.0, 73, 75, 20); // C#5
  time += 4.5;
  addCc(events, time, 64, 0);

  sortEvents(events);

  return {
    format: 0,
    trackCount: 1,
    division: 480,
    durationSeconds: time + 1.0,
    events,
  };
}
