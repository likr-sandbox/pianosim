/** Procedural Beethoven Moonlight Sonata (Op. 27 No. 2, 1st Mvt) — humanized voicing & half-pedaling. */

import type { MidiPlaybackEvent, ParsedSmf } from '../smf-parser';
import { addCc, addNote, addRolledChord, sortEvents } from './types';

export function createBeethovenMoonlight(): ParsedSmf {
  const events: MidiPlaybackEvent[] = [];
  const bpm = 54;
  const quarterSec = 60 / bpm; // ~1.111 s per beat
  const tripletSec = quarterSec / 3; // ~0.370 s per triplet 8th
  const measureSec = quarterSec * 4; // ~4.444 s per 4/4 measure

  let time = 0.0;

  const scoreData = [
    { bass: [25, 37], trips: [56, 61, 64], melody: [] as Array<{ beat: number; key: number; dur: number; vel: number }> },
    { bass: [25, 37], trips: [56, 61, 64], melody: [] },
    {
      bass: [25, 37],
      trips: [56, 61, 64],
      melody: [
        { beat: 2.66, key: 68, dur: 0.85, vel: 88 },
        { beat: 3.66, key: 68, dur: 0.85, vel: 92 },
      ],
    },
    {
      bass: [23, 35],
      trips: [56, 59, 64],
      melody: [
        { beat: 0.0, key: 68, dur: 2.2, vel: 94 },
        { beat: 2.0, key: 68, dur: 0.72, vel: 84 },
        { beat: 2.66, key: 69, dur: 0.72, vel: 90 },
        { beat: 3.33, key: 68, dur: 0.72, vel: 86 },
      ],
    },
    {
      bass: [21, 33],
      trips: [57, 61, 64],
      melody: [
        { beat: 0.0, key: 66, dur: 2.05, vel: 90 },
        { beat: 2.0, key: 64, dur: 1.05, vel: 84 },
        { beat: 3.0, key: 63, dur: 1.05, vel: 88 },
      ],
    },
    {
      bass: [32, 44],
      trips: [56, 59, 62],
      melody: [
        { beat: 0.0, key: 64, dur: 2.05, vel: 92 },
        { beat: 2.0, key: 63, dur: 2.05, vel: 86 },
      ],
    },
    {
      bass: [25, 37],
      trips: [56, 61, 64],
      melody: [{ beat: 0.0, key: 61, dur: 3.8, vel: 94 }],
    },
    {
      bass: [25, 37],
      trips: [49, 56, 61],
      melody: [
        { beat: 0.0, key: 64, dur: 4.0, vel: 82 },
        { beat: 2.0, key: 73, dur: 4.0, vel: 86 },
      ],
    },
  ];

  for (let mi = 0; mi < scoreData.length; mi++) {
    const m = scoreData[mi];
    // Continuous half-pedal shimmer (72..92) with mid-measure partial pedal catch (44..52)
    addCc(events, time + 0.015, 64, 86);
    addCc(events, time + measureSec * 0.5, 64, 54);
    addCc(events, time + measureSec * 0.55, 64, 82);

    // Gently rolled deep bass octaves on beat 1 (beat 0 starts at exact time)
    addRolledChord(events, time, measureSec * 0.92, m.bass, 78, 0.012, 4, 28);

    // Subdued triplet accompaniment (vel 44..54) so melody sings cleanly above
    for (let t = 0; t < 12; t++) {
      const tripKey = m.trips[t % 3];
      const rubato = t === 0 ? 0 : Math.sin((t / 12) * Math.PI) * 0.006;
      const tripTime = time + t * tripletSec + rubato;
      const tripVel = t % 3 === 0 ? 54 : 45 + (t % 2) * 2;
      addNote(events, tripTime, tripletSec * 1.06, tripKey, tripVel, 28);
    }

    // Cantabile singing melody voice
    for (const mel of m.melody) {
      const melTime = time + mel.beat * quarterSec + (mel.beat > 0 ? 0.008 : 0);
      addNote(events, melTime, mel.dur, mel.key, mel.vel, 32);
    }

    // Half-pedal breath -> clean catch right before next measure
    addCc(events, time + measureSec - 0.14, 64, 38);
    addCc(events, time + measureSec - 0.06, 64, 0);

    time += measureSec;
  }

  // Final rolled C# minor chord tail
  addCc(events, time + 0.02, 64, 94);
  addRolledChord(events, time + 0.08, 4.0, [25, 37, 49, 61, 64, 73], 74, 0.022, 12, 20);
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
