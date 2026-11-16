/** Procedural Sergei Rachmaninoff Prelude in C# Minor (Op. 3 No. 2) — ~25 seconds.
 * Showcases fortissimo dynamic stress, hammer felt hardening, and Zone 9 keybed thump.
 */

import type { MidiPlaybackEvent, ParsedSmf } from '../smf-parser';
import { addCc, addNote, sortEvents } from './types';

export function createRachmaninoffPrelude(): ParsedSmf {
  const events: MidiPlaybackEvent[] = [];
  const beatSec = 1.35; // Lento (~45 BPM)

  let time = 0.0;

  // Initial iconic descending motif: A -> G# -> C# fortissimo
  // Note 1: Fortissimo A
  addCc(events, time, 64, 127);
  addNote(events, time, beatSec * 1.5, 69, 118, 40); // A4
  addNote(events, time, beatSec * 1.5, 57, 115, 40); // A3
  time += beatSec * 1.5;

  // Note 2: G#
  addNote(events, time, beatSec * 1.5, 68, 120, 40); // G#4
  addNote(events, time, beatSec * 1.5, 56, 116, 40); // G#3
  time += beatSec * 1.5;

  // Note 3: Low massive C# octaves (fff, vel 126) triggering extreme hammer compression & keybed thump
  addCc(events, time - 0.05, 64, 0); // Catch pedal
  addCc(events, time + 0.02, 64, 127);
  addNote(events, time, beatSec * 3.0, 25, 126, 50); // C#1
  addNote(events, time, beatSec * 3.0, 37, 126, 50); // C#2
  addNote(events, time, beatSec * 3.0, 49, 122, 50); // C#3
  time += beatSec * 3.2;

  // Dramatic silence / decay breath
  addCc(events, time, 64, 0);
  time += 0.8;

  // Answering solemn pianissimo chords in the middle register
  const pppChords = [
    // Chord 1: E major / C#m (G#3, C#4, E4)
    { keys: [56, 61, 64], dur: beatSec * 1.1, vel: 55 },
    // Chord 2: F# minor (A3, C#4, F#4)
    { keys: [57, 61, 66], dur: beatSec * 1.1, vel: 58 },
    // Chord 3: G# major 7 (B#3=C4=60, D#4=63, G#4=68)
    { keys: [60, 63, 68], dur: beatSec * 1.2, vel: 60 },
    // Chord 4: Resolution C# minor (G#3, C#4, E4)
    { keys: [56, 61, 64], dur: beatSec * 1.8, vel: 52 },
  ];

  for (const c of pppChords) {
    addCc(events, time, 64, 100);
    for (const k of c.keys) {
      addNote(events, time, c.dur, k, c.vel, 30);
    }
    time += c.dur + 0.15;
    addCc(events, time - 0.05, 64, 0);
  }

  // Second repetition with heavier bass fortissimo
  time += 0.4;
  addCc(events, time, 64, 127);
  // High octave A
  addNote(events, time, beatSec * 1.0, 81, 124, 40); // A5
  addNote(events, time, beatSec * 1.0, 69, 122, 40); // A4
  time += beatSec * 1.0;

  // G#
  addNote(events, time, beatSec * 1.0, 80, 125, 40); // G#5
  addNote(events, time, beatSec * 1.0, 68, 124, 40); // G#4
  time += beatSec * 1.0;

  // Thunderous C# crash across 4 octaves
  addCc(events, time - 0.05, 64, 0);
  addCc(events, time + 0.02, 64, 127);
  addNote(events, time, 5.0, 25, 127, 50); // C#1
  addNote(events, time, 5.0, 37, 127, 50); // C#2
  addNote(events, time, 5.0, 49, 125, 50); // C#3
  addNote(events, time, 5.0, 61, 125, 50); // C#4
  addNote(events, time, 5.0, 64, 122, 50); // E4
  addNote(events, time, 5.0, 73, 125, 50); // C#5

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
