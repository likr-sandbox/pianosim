import { test, expect } from '@playwright/test';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import { verifySampleBounds, computeRMS } from '../helpers/audio-analysis';

test.describe('Tier 4: Application Scenario - Beethoven Für Elise', () => {
  test('T4-APP-02: Beethoven Für Elise Opening Motif with Expressive Dynamics', async () => {
    const synth = createPianoSynthesizer(44100);

    // E5 (76), D#5 (75), E5 (76), D#5 (75), E5 (76), B4 (71), D5 (74), C5 (72), A4 (69)
    const phrase = [
      { midi: 76, vel: 85 },
      { midi: 75, vel: 78 },
      { midi: 76, vel: 82 },
      { midi: 75, vel: 76 },
      { midi: 76, vel: 80 },
      { midi: 71, vel: 72 },
      { midi: 74, vel: 75 },
      { midi: 72, vel: 70 },
      { midi: 69, vel: 65 }, // concluding resolved A4
    ];

    const onSamples = Math.round(44100 * 0.1); // 100ms noteOn
    const offSamples = Math.round(44100 * 0.03); // 30ms staccato gap

    const totalSamples = (onSamples + offSamples) * phrase.length + 44100; // phrase + 1s tail
    const fullBuffer = new Float32Array(totalSamples);

    let offset = 0;
    for (const { midi, vel } of phrase) {
      synth.noteOn({ midiNote: midi, velocity: vel });
      const onSlice = fullBuffer.subarray(offset, offset + onSamples);
      renderMono(synth, onSlice, onSamples);
      offset += onSamples;

      synth.noteOff({ midiNote: midi });
      const offSlice = fullBuffer.subarray(offset, offset + offSamples);
      renderMono(synth, offSlice, offSamples);
      offset += offSamples;
    }

    // Render resolution decay
    const tailSamples = totalSamples - offset;
    const tailSlice = fullBuffer.subarray(offset, offset + tailSamples);
    renderMono(synth, tailSlice, tailSamples);

    const bounds = verifySampleBounds(fullBuffer);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
    expect(bounds.absMax).toBeLessThanOrEqual(1.0);
    expect(bounds.zeroClipping).toBe(true);

    // Distinct note attacks: RMS during note attacks should be high
    const firstNoteRMS = computeRMS(fullBuffer, 0, onSamples);
    expect(firstNoteRMS).toBeGreaterThan(0.01);
  });
});
