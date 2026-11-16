/** Procedural Bach Prelude in C Major (BWV 846) — 16 measures, 32 seconds. */

import type { MidiPlaybackEvent, ParsedSmf } from '../smf-parser';
import { sortEvents } from './types';

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
      const velocity = step === 0 ? 80 : step === 1 ? 74 : 68;
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

  sortEvents(events);

  return {
    format: 0,
    trackCount: 1,
    division: 480,
    durationSeconds: time + 1.0,
    events,
  };
}
