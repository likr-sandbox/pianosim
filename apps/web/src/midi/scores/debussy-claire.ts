/** Procedural Debussy Clair de Lune (Suite bergamasque) — ~35 seconds.
 * Showcases delicate pianissimo dynamics, high-register shimmer, and Una Corda soft pedal (CC 67).
 */

import type { MidiPlaybackEvent, ParsedSmf } from '../smf-parser';
import { addCc, addNote, sortEvents } from './types';

export function createDebussyClaire(): ParsedSmf {
  const events: MidiPlaybackEvent[] = [];
  const beatSec = 0.42; // ~142 BPM for eighth notes in 9/8 time (dotted quarter = 1.26s)
  const measureSec = beatSec * 9; // ~3.78 s per 9/8 measure

  let time = 0.0;

  // Engage Una Corda soft pedal (CC 67 = 127) for delicate impressionist timbre
  addCc(events, 0.0, 67, 127);

  const measures = [
    // M1: Opening ethereal third in Db major (F5 + Eb5 resolution)
    {
      bass: 49, // Db3
      chords: [
        { beat: 2, keys: [61, 65, 68], vel: 52 }, // Db4, F4, Ab4
        { beat: 5, keys: [61, 65, 68], vel: 50 },
      ],
      melody: [
        { beat: 0, key: 77, dur: beatSec * 2.5, vel: 62 }, // F5
        { beat: 2.5, key: 75, dur: beatSec * 3.0, vel: 58 }, // Eb5
        { beat: 6, key: 73, dur: beatSec * 2.5, vel: 60 }, // Db5
      ],
    },
    // M2: Gb / Db harmony
    {
      bass: 46, // Bb2
      chords: [
        { beat: 2, keys: [58, 65, 70], vel: 52 }, // Bb3, F4, Bb4
        { beat: 5, keys: [58, 65, 70], vel: 50 },
      ],
      melody: [
        { beat: 0, key: 72, dur: beatSec * 2.5, vel: 60 }, // C5
        { beat: 3, key: 70, dur: beatSec * 3.0, vel: 56 }, // Bb4
        { beat: 6, key: 68, dur: beatSec * 2.8, vel: 58 }, // Ab4
      ],
    },
    // M3: Flowing upward arpeggio towards higher octave
    {
      bass: 42, // Gb2
      chords: [
        { beat: 1, keys: [54, 61, 66], vel: 54 }, // Gb3, Db4, Gb4
        { beat: 4, keys: [61, 66, 70], vel: 52 }, // Db4, Gb4, Bb4
        { beat: 7, keys: [66, 70, 73], vel: 55 }, // Gb4, Bb4, Db5
      ],
      melody: [
        { beat: 0, key: 66, dur: beatSec * 2.0, vel: 56 }, // Gb4
        { beat: 3, key: 70, dur: beatSec * 2.0, vel: 62 }, // Bb4
        { beat: 6, key: 73, dur: beatSec * 2.5, vel: 66 }, // Db5
      ],
    },
    // M4: High-register bell tones (Ab5 -> F6)
    {
      bass: 44, // Ab2
      chords: [
        { beat: 2, keys: [56, 60, 68], vel: 52 }, // Ab3, C4, Ab4
        { beat: 5, keys: [63, 68, 72], vel: 54 }, // Eb4, Ab4, C5
      ],
      melody: [
        { beat: 0, key: 80, dur: beatSec * 3.0, vel: 68 }, // Ab5
        { beat: 3, key: 85, dur: beatSec * 2.5, vel: 70 }, // Db6
        { beat: 6, key: 89, dur: beatSec * 2.8, vel: 72 }, // F6 (shimmer)
      ],
    },
    // M5: Calming descent in thirds
    {
      bass: 49, // Db3
      chords: [
        { beat: 2, keys: [61, 65, 68], vel: 50 },
        { beat: 5, keys: [61, 65, 68], vel: 48 },
      ],
      melody: [
        { beat: 0, key: 85, dur: beatSec * 2.0, vel: 64 }, // Db6
        { beat: 2, key: 80, dur: beatSec * 2.0, vel: 60 }, // Ab5
        { beat: 4, key: 77, dur: beatSec * 2.0, vel: 58 }, // F5
        { beat: 6, key: 73, dur: beatSec * 2.5, vel: 56 }, // Db5
      ],
    },
    // M6: Subdued contemplative cadence
    {
      bass: 42, // Gb2
      chords: [
        { beat: 2, keys: [58, 65, 68], vel: 48 },
        { beat: 5, keys: [56, 63, 68], vel: 46 },
      ],
      melody: [
        { beat: 0, key: 70, dur: beatSec * 3.0, vel: 54 }, // Bb4
        { beat: 3, key: 68, dur: beatSec * 3.0, vel: 52 }, // Ab4
        { beat: 6, key: 65, dur: beatSec * 2.8, vel: 50 }, // F4
      ],
    },
    // M7: Final quiet Db major chord in pianissimo
    {
      bass: 37, // Db2
      chords: [
        { beat: 2, keys: [61, 65, 68, 73], vel: 45 }, // Db4, F4, Ab4, Db5
      ],
      melody: [
        { beat: 0, key: 85, dur: beatSec * 8.0, vel: 58 }, // High Db6
      ],
    },
  ];

  for (const m of measures) {
    // Gentle damper pedal with breath
    addCc(events, time + 0.05, 64, 110);

    // Warm bass root
    addNote(events, time, measureSec * 0.9, m.bass, 68, 30);

    // Harmonic accompaniment
    for (const c of m.chords) {
      const cTime = time + c.beat * beatSec;
      for (const k of c.keys) {
        addNote(events, cTime, beatSec * 2.2, k, c.vel, 25);
      }
    }

    // Singing melody line
    for (const mel of m.melody) {
      const melTime = time + mel.beat * beatSec;
      addNote(events, melTime, mel.dur, mel.key, mel.vel, 30);
    }

    // Pedal release right before next measure
    addCc(events, time + measureSec - 0.1, 64, 0);

    time += measureSec;
  }

  // Release Una Corda and Damper pedal at end of score
  addCc(events, time, 67, 0);
  addCc(events, time, 64, 0);

  sortEvents(events);

  return {
    format: 0,
    trackCount: 1,
    division: 480,
    durationSeconds: time + 1.5,
    events,
  };
}
