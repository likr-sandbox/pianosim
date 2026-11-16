/** Procedural Erik Satie Gymnopédie No. 1 — ~36 seconds.
 * Showcases spacious acoustic decay, two-stage string damping, and ambient clarity.
 */

import type { MidiPlaybackEvent, ParsedSmf } from '../smf-parser';
import { addCc, addNote, sortEvents } from './types';

export function createSatieGymnopedie(): ParsedSmf {
  const events: MidiPlaybackEvent[] = [];
  const beatSec = 0.88; // Lent et douloureux (~68 BPM)
  const measureSec = beatSec * 3; // ~2.64 s per 3/4 measure

  let time = 0.0;

  // 12 measures of serene waltz texture
  const scoreData = [
    // M1: Intro G major 7
    { bass: 43, chord: [59, 62, 66], mel: null }, // G2, [B3, D4, F#4]
    // M2: D major 7
    { bass: 50, chord: [57, 62, 66], mel: null }, // D3, [A3, D4, F#4]
    // M3: G major 7
    { bass: 43, chord: [59, 62, 66], mel: null },
    // M4: D major 7
    { bass: 50, chord: [57, 62, 66], mel: null },
    // M5: Melody begins on high F#5
    { bass: 43, chord: [59, 62, 66], mel: { beat: 0, key: 78, dur: beatSec * 2.8, vel: 78 } }, // F#5
    // M6: Descending to E5
    { bass: 50, chord: [57, 62, 66], mel: { beat: 0, key: 76, dur: beatSec * 2.8, vel: 74 } }, // E5
    // M7: Melody D5 -> B4
    {
      bass: 43,
      chord: [59, 62, 66],
      mel: { beat: 0, key: 74, dur: beatSec * 1.8, vel: 72 }, // D5
      mel2: { beat: 2, key: 71, dur: beatSec * 0.9, vel: 70 }, // B4
    },
    // M8: C#5 -> D5
    {
      bass: 50,
      chord: [57, 62, 66],
      mel: { beat: 0, key: 73, dur: beatSec * 1.8, vel: 72 }, // C#5
      mel2: { beat: 2, key: 74, dur: beatSec * 0.9, vel: 70 }, // D5
    },
    // M9: Up to E5
    { bass: 43, chord: [59, 62, 66], mel: { beat: 0, key: 76, dur: beatSec * 2.8, vel: 75 } },
    // M10: Falling back to B4
    { bass: 50, chord: [57, 61, 66], mel: { beat: 0, key: 71, dur: beatSec * 2.8, vel: 72 } },
    // M11: Resolution chord
    { bass: 43, chord: [55, 59, 62, 67], mel: null }, // G2, [G3, B3, D4, G4]
    // M12: Deep closing silence
    { bass: 38, chord: [50, 57, 62, 66], mel: null }, // D2, [D3, A3, D4, F#4]
  ];

  for (const m of scoreData) {
    // Damper pedal engaged for rich ambient reverberation across each chord
    addCc(events, time + 0.05, 64, 120);

    // Beat 1: Solitary bass root
    addNote(events, time, measureSec * 0.92, m.bass, 78, 30);

    // Beats 2 & 3: Sustained mid-range impressionist chord
    const chordTime1 = time + beatSec * 1.0;
    const chordTime2 = time + beatSec * 2.0;
    for (const k of m.chord) {
      addNote(events, chordTime1, beatSec * 0.9, k, 62, 25);
      addNote(events, chordTime2, beatSec * 0.85, k, 58, 25);
    }

    // Melodic voice if present
    if (m.mel) {
      const mTime = time + m.mel.beat * beatSec;
      addNote(events, mTime, m.mel.dur, m.mel.key, m.mel.vel, 35);
    }
    if ('mel2' in m && m.mel2) {
      const mTime2 = time + m.mel2.beat * beatSec;
      addNote(events, mTime2, m.mel2.dur, m.mel2.key, m.mel2.vel, 35);
    }

    // Clean pedal catch right before next measure
    addCc(events, time + measureSec - 0.08, 64, 0);

    time += measureSec;
  }

  // Final acoustic tail
  addCc(events, time, 64, 127);
  addNote(events, time + 0.1, 4.0, 38, 70, 20); // D2
  addNote(events, time + 0.1, 4.0, 50, 65, 20); // D3
  addNote(events, time + 0.1, 4.0, 62, 60, 20); // D4
  addNote(events, time + 0.1, 4.0, 66, 62, 20); // F#4
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
