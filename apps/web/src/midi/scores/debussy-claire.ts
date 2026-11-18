/** Procedural Debussy Clair de Lune — humanized rolled chords, melody shading & continuous half-pedaling. */

import type { MidiPlaybackEvent, ParsedSmf } from '../smf-parser';
import { addCc, addNote, addRolledChord, sortEvents } from './types';

export function createDebussyClaire(): ParsedSmf {
  const events: MidiPlaybackEvent[] = [];
  const beatSec = 0.42;
  const measureSec = beatSec * 9;
  let time = 0.0;

  // Continuous Una Corda shading (soft impressionist veil, CC67 = 104)
  addCc(events, 0.0, 67, 104);

  const measures = [
    {
      bass: 49,
      chords: [
        { beat: 2, keys: [61, 65, 68], vel: 48 },
        { beat: 5, keys: [61, 65, 68], vel: 45 },
      ],
      melody: [
        { beat: 0, key: 77, dur: beatSec * 2.5, vel: 68 },
        { beat: 2.5, key: 75, dur: beatSec * 3.0, vel: 64 },
        { beat: 6, key: 73, dur: beatSec * 2.5, vel: 66 },
      ],
    },
    {
      bass: 46,
      chords: [
        { beat: 2, keys: [58, 65, 70], vel: 48 },
        { beat: 5, keys: [58, 65, 70], vel: 46 },
      ],
      melody: [
        { beat: 0, key: 72, dur: beatSec * 2.5, vel: 66 },
        { beat: 3, key: 70, dur: beatSec * 3.0, vel: 62 },
        { beat: 6, key: 68, dur: beatSec * 2.8, vel: 64 },
      ],
    },
    {
      bass: 42,
      chords: [
        { beat: 1, keys: [54, 61, 66], vel: 50 },
        { beat: 4, keys: [61, 66, 70], vel: 49 },
        { beat: 7, keys: [66, 70, 73], vel: 52 },
      ],
      melody: [
        { beat: 0, key: 66, dur: beatSec * 2.0, vel: 62 },
        { beat: 3, key: 70, dur: beatSec * 2.0, vel: 68 },
        { beat: 6, key: 73, dur: beatSec * 2.5, vel: 72 },
      ],
    },
    {
      bass: 44,
      chords: [
        { beat: 2, keys: [56, 60, 68], vel: 48 },
        { beat: 5, keys: [63, 68, 72], vel: 50 },
      ],
      melody: [
        { beat: 0, key: 80, dur: beatSec * 3.0, vel: 74 },
        { beat: 3, key: 85, dur: beatSec * 2.5, vel: 78 },
        { beat: 6, key: 89, dur: beatSec * 2.8, vel: 80 },
      ],
    },
    {
      bass: 49,
      chords: [
        { beat: 2, keys: [61, 65, 68], vel: 46 },
        { beat: 5, keys: [61, 65, 68], vel: 44 },
      ],
      melody: [
        { beat: 0, key: 85, dur: beatSec * 2.0, vel: 70 },
        { beat: 2, key: 80, dur: beatSec * 2.0, vel: 66 },
        { beat: 4, key: 77, dur: beatSec * 2.0, vel: 64 },
        { beat: 6, key: 73, dur: beatSec * 2.5, vel: 62 },
      ],
    },
    {
      bass: 42,
      chords: [
        { beat: 2, keys: [58, 65, 68], vel: 44 },
        { beat: 5, keys: [56, 63, 68], vel: 42 },
      ],
      melody: [
        { beat: 0, key: 70, dur: beatSec * 3.0, vel: 60 },
        { beat: 3, key: 68, dur: beatSec * 3.0, vel: 58 },
        { beat: 6, key: 65, dur: beatSec * 2.8, vel: 56 },
      ],
    },
    {
      bass: 37,
      chords: [{ beat: 2, keys: [61, 65, 68, 73], vel: 42 }],
      melody: [{ beat: 0, key: 85, dur: beatSec * 8.0, vel: 64 }],
    },
  ];

  for (let mi = 0; mi < measures.length; mi++) {
    const m = measures[mi];
    // Continuous half-pedal depth (68..86) + subtle Una Corda modulation
    addCc(events, time + 0.03, 64, mi === 3 ? 88 : 74);
    addCc(events, time + measureSec * 0.5, 64, 52);
    addCc(events, time + measureSec * 0.55, 64, 78);

    addNote(events, time, measureSec * 0.9, m.bass, 66, 28);

    for (const c of m.chords) {
      const cTime = time + c.beat * beatSec;
      addRolledChord(events, cTime, beatSec * 2.2, c.keys, c.vel, 0.018, 6, 24);
    }

    for (const mel of m.melody) {
      const melTime = time + mel.beat * beatSec + (mel.beat > 0 ? 0.008 : 0);
      addNote(events, melTime, mel.dur, mel.key, mel.vel, 28);
    }

    addCc(events, time + measureSec - 0.14, 64, 38);
    addCc(events, time + measureSec - 0.05, 64, 0);

    time += measureSec;
  }

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
