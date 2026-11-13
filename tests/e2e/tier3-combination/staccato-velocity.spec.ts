import { test, expect } from '@playwright/test';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import { computeRMS } from '../helpers/audio-analysis';

test.describe('Tier 3: Combination - Staccato & Expressive Velocity', () => {
  test('T3-CMB-02: Dynamic Velocity Staccato Sequence', async () => {
    const synth = createPianoSynthesizer(44100);
    const testNotes = [
      { midi: 60, vel: 30 },
      { midi: 62, vel: 110 },
      { midi: 64, vel: 45 },
      { midi: 65, vel: 125 },
      { midi: 67, vel: 40 },
    ];

    const noteDur = Math.round(44100 * 0.03); // 30ms noteOn
    const gapDur = Math.round(44100 * 0.03); // 30ms release gap

    const rmsValues: number[] = [];

    for (const { midi, vel } of testNotes) {
      synth.noteOn({ midiNote: midi, velocity: vel });
      const onBuf = new Float32Array(noteDur);
      renderMono(synth, onBuf, noteDur);
      rmsValues.push(computeRMS(onBuf));

      synth.noteOff({ midiNote: midi });
      const offBuf = new Float32Array(gapDur);
      renderMono(synth, offBuf, gapDur);
    }

    // High velocity notes (index 1: vel 110, index 3: vel 125) should have significantly higher RMS
    // than low velocity notes (index 0: vel 30, index 2: vel 45, index 4: vel 40)
    expect(rmsValues[1]).toBeGreaterThan(rmsValues[0] * 2.0);
    expect(rmsValues[3]).toBeGreaterThan(rmsValues[2] * 2.0);
    expect(rmsValues[3]).toBeGreaterThan(rmsValues[4] * 2.0);
  });
});
