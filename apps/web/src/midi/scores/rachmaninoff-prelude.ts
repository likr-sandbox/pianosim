/** Procedural Sergei Rachmaninoff Prelude in C# Minor (Op. 3 No. 2) — humanized chord voicing & half-pedaling. */

import type { MidiPlaybackEvent, ParsedSmf } from '../smf-parser';
import { addCc, addNote, addRolledChord, sortEvents } from './types';

export function createRachmaninoffPrelude(): ParsedSmf {
  const events: MidiPlaybackEvent[] = [];
  const beatSec = 1.35;
  let time = 0.0;

  // Immediate beat-0 fortissimo octave A with top voice emphasis
  addCc(events, time, 64, 96);
  addNote(events, time, beatSec * 1.5, 69, 120, 40);
  addNote(events, time + 0.006, beatSec * 1.5, 57, 112, 40);
  time += beatSec * 1.5;

  // G# descending motif
  addNote(events, time, beatSec * 1.5, 68, 122, 40);
  addNote(events, time + 0.006, beatSec * 1.5, 56, 114, 40);
  time += beatSec * 1.5;

  // Massive low C# octaves with subtle 8 ms micro-roll & half-pedal catch
  addCc(events, time - 0.05, 64, 0);
  addCc(events, time + 0.02, 64, 110);
  addRolledChord(events, time, beatSec * 3.0, [25, 37, 49], 122, 0.008, 4, 48);
  time += beatSec * 3.2;

  addCc(events, time, 64, 44);
  addCc(events, time + 0.25, 64, 0);
  time += 0.8;

  // Answering pianissimo chords with rolled voicing, top melody emphasis, and half-pedaling
  const pppChords = [
    { keys: [56, 61, 64], dur: beatSec * 1.1, vel: 50 },
    { keys: [57, 61, 66], dur: beatSec * 1.1, vel: 54 },
    { keys: [60, 63, 68], dur: beatSec * 1.2, vel: 58 },
    { keys: [56, 61, 64], dur: beatSec * 1.8, vel: 48 },
  ];

  for (const c of pppChords) {
    addCc(events, time, 64, 68);
    addRolledChord(events, time, c.dur, c.keys, c.vel, 0.014, 10, 28);
    time += c.dur + 0.15;
    addCc(events, time - 0.08, 64, 36);
    addCc(events, time - 0.03, 64, 0);
  }

  time += 0.4;
  addCc(events, time, 64, 104);
  addNote(events, time, beatSec * 1.0, 81, 124, 40);
  addNote(events, time + 0.006, beatSec * 1.0, 69, 118, 40);
  time += beatSec * 1.0;

  addNote(events, time, beatSec * 1.0, 80, 125, 40);
  addNote(events, time + 0.006, beatSec * 1.0, 68, 120, 40);
  time += beatSec * 1.0;

  // Thunderous 4-octave rolled C# minor crash
  addCc(events, time - 0.05, 64, 0);
  addCc(events, time + 0.02, 64, 116);
  addRolledChord(events, time, 5.0, [25, 37, 49, 61, 64, 73], 122, 0.009, 5, 48);

  time += 5.5;
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
