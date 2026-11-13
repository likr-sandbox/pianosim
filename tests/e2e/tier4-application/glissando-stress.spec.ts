import { test, expect } from '@playwright/test';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import { verifySampleBounds, computeRMS } from '../helpers/audio-analysis';

test.describe('Tier 4: Application Scenario - Chromatic Glissando Stress', () => {
  test('T4-APP-04: Full-Keyboard Chromatic Glissando Burst under Voice-Stealing Cap', async () => {
    const synth = createPianoSynthesizer(44100);

    // Sustain pedal engaged
    synth.setSustainPedal(true);

    const stepSamples = Math.round(44100 * 0.015); // 15ms per chromatic step
    const startNote = 36; // C2
    const endNote = 84; // C6 (48 chromatic notes)
    const noteCount = endNote - startNote + 1;

    const totalSamples = stepSamples * noteCount + 44100; // glissando + 1s tail
    const fullBuffer = new Float32Array(totalSamples);

    let offset = 0;
    for (let midi = startNote; midi <= endNote; midi++) {
      synth.noteOn({ midiNote: midi, velocity: 95 });
      const slice = fullBuffer.subarray(offset, offset + stepSamples);
      renderMono(synth, slice, stepSamples);
      offset += stepSamples;
    }

    // Render residual resonance
    const tailSamples = totalSamples - offset;
    const tailSlice = fullBuffer.subarray(offset, offset + tailSamples);
    renderMono(synth, tailSlice, tailSamples);

    // Verify stability
    const bounds = verifySampleBounds(fullBuffer);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
    expect(bounds.absMax).toBeLessThanOrEqual(1.0);
    expect(bounds.zeroClipping).toBe(true);

    // Audio energy should remain continuous without dropouts
    const midEnergy = computeRMS(fullBuffer, stepSamples * 20, stepSamples * 30);
    expect(midEnergy).toBeGreaterThan(0.01);
  });
});
