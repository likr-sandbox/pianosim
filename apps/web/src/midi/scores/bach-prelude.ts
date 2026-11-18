/** Procedural Bach Prelude in C Major (BWV 846) — 16 measures, humanized phrasing & half-pedaling. */

import type { MidiPlaybackEvent, ParsedSmf } from '../smf-parser';
import { addCc, addNote, addRolledChord, sortEvents } from './types';

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
  const noteDuration = 0.125; // Nominal 16th note at 120 BPM = 125 ms
  const pattern = [0, 1, 2, 3, 4, 2, 3, 4, 0, 1, 2, 3, 4, 2, 3, 4];
  let time = 0.0;

  for (let m = 0; m < chords.length; m++) {
    const chord = chords[m];
    const phraseArch = Math.sin((m / (chords.length - 1)) * Math.PI); // gentle dynamic swell
    const halfPedalDepth = m === chords.length - 1 ? 78 : 42 + Math.round(phraseArch * 16);

    // Subtle harpsichord-to-piano half-pedal bloom per harmony, lifted cleanly at measure change
    addCc(events, time + 0.015, 64, halfPedalDepth);

    // Sustain bass anchor gently on first note of each figure
    addNote(events, time, noteDuration * 7.2, chord[0], 78 + Math.round(phraseArch * 6), 34);

    for (let idx = 0; idx < pattern.length; idx++) {
      const step = pattern[idx];
      const midiKey = chord[step];
      // Micro-rubato wave across 16th-note figure (0 at beat 0)
      const microOffset = idx === 0 ? 0 : Math.sin((idx / pattern.length) * Math.PI) * 0.004;
      // Emphasize upper melodic voice (step 4) and bass (step 0) over inner accompaniment (1, 2)
      let vel = 58;
      if (step === 4) {
        vel = 80 + Math.round(phraseArch * 8);
      } else if (step === 3) {
        vel = 72 + Math.round(phraseArch * 5);
      } else if (step === 0) {
        vel = 76 + Math.round(phraseArch * 4);
      } else {
        vel = 56 + (idx % 2 === 0 ? 4 : 0);
      }
      if (idx > 0) {
        addNote(events, time + microOffset, 0.122, midiKey, vel, 48);
      }
      time += noteDuration;
    }

    // Half-pedal release breath before harmonic shift
    addCc(events, time - 0.04, 64, 0);
  }

  // Final rolled C major cadence chord with warm pedal wash
  addCc(events, time + 0.01, 64, 88);
  addRolledChord(events, time + 0.02, 1.8, [36, 48, 52, 55, 60, 64], 76, 0.018, 10, 28);
  time += 2.0;
  addCc(events, time, 64, 0);

  sortEvents(events);

  return {
    format: 0,
    trackCount: 1,
    division: 480,
    durationSeconds: time + 0.8,
    events,
  };
}
