/** Procedural Erik Satie Gymnopédie No. 1 — humanized waltz chord rolls, melody prominence & half-pedaling. */

import type { MidiPlaybackEvent, ParsedSmf } from '../smf-parser';
import { addCc, addNote, addRolledChord, sortEvents } from './types';

export function createSatieGymnopedie(): ParsedSmf {
  const events: MidiPlaybackEvent[] = [];
  const beatSec = 0.88;
  const measureSec = beatSec * 3;
  let time = 0.0;

  const scoreData = [
    { bass: 43, chord: [59, 62, 66], mel: null },
    { bass: 50, chord: [57, 62, 66], mel: null },
    { bass: 43, chord: [59, 62, 66], mel: null },
    { bass: 50, chord: [57, 62, 66], mel: null },
    { bass: 43, chord: [59, 62, 66], mel: { beat: 0, key: 78, dur: beatSec * 2.8, vel: 84 } },
    { bass: 50, chord: [57, 62, 66], mel: { beat: 0, key: 76, dur: beatSec * 2.8, vel: 80 } },
    {
      bass: 43,
      chord: [59, 62, 66],
      mel: { beat: 0, key: 74, dur: beatSec * 1.8, vel: 78 },
      mel2: { beat: 2, key: 71, dur: beatSec * 0.9, vel: 76 },
    },
    {
      bass: 50,
      chord: [57, 62, 66],
      mel: { beat: 0, key: 73, dur: beatSec * 1.8, vel: 78 },
      mel2: { beat: 2, key: 74, dur: beatSec * 0.9, vel: 76 },
    },
    { bass: 43, chord: [59, 62, 66], mel: { beat: 0, key: 76, dur: beatSec * 2.8, vel: 82 } },
    { bass: 50, chord: [57, 61, 66], mel: { beat: 0, key: 71, dur: beatSec * 2.8, vel: 78 } },
    { bass: 43, chord: [55, 59, 62, 67], mel: null },
    { bass: 38, chord: [50, 57, 62, 66], mel: null },
  ];

  for (const m of scoreData) {
    // Luminous continuous half-pedal (76) with partial transition (44) before clean catch (0)
    addCc(events, time + 0.03, 64, 76);

    addNote(events, time, measureSec * 0.92, m.bass, 72, 28);

    // Rolled waltz accompaniment chords (15 ms stagger, soft 50..56 velocity)
    const chordTime1 = time + beatSec * 1.0;
    const chordTime2 = time + beatSec * 2.0;
    addRolledChord(events, chordTime1, beatSec * 0.9, m.chord, 54, 0.015, 5, 24);
    addRolledChord(events, chordTime2, beatSec * 0.85, m.chord, 48, 0.015, 4, 24);

    if (m.mel) {
      const mTime = time + m.mel.beat * beatSec + (m.mel.beat > 0 ? 0.008 : 0);
      addNote(events, mTime, m.mel.dur, m.mel.key, m.mel.vel, 32);
    }
    if ('mel2' in m && m.mel2) {
      const mTime2 = time + m.mel2.beat * beatSec + 0.008;
      addNote(events, mTime2, m.mel2.dur, m.mel2.key, m.mel2.vel, 32);
    }

    addCc(events, time + measureSec - 0.14, 64, 42);
    addCc(events, time + measureSec - 0.06, 64, 0);

    time += measureSec;
  }

  addCc(events, time + 0.02, 64, 92);
  addRolledChord(events, time + 0.08, 4.0, [38, 50, 62, 66], 62, 0.02, 8, 20);
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
