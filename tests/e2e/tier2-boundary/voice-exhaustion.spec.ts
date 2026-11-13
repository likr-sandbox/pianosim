import { test, expect } from '@playwright/test';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import { verifySampleBounds, computeRMS } from '../helpers/audio-analysis';

test.describe('Tier 2: Boundary - Voice Exhaustion & Tremolo', () => {
  test('T2-BND-07: Rapid Single-Note Tremolo (20ms)', async () => {
    const synth = createPianoSynthesizer(44100);
    const intervalSamples = Math.round(44100 * 0.02); // 20ms = 882 samples

    const totalBuffer = new Float32Array(intervalSamples * 10);

    for (let strike = 0; strike < 10; strike++) {
      synth.noteOn({ midiNote: 60, velocity: 100 });
      const sliceLeft = totalBuffer.subarray(
        strike * intervalSamples,
        (strike + 1) * intervalSamples
      );
      renderMono(synth, sliceLeft, intervalSamples);
    }

    const bounds = verifySampleBounds(totalBuffer);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
    expect(bounds.absMax).toBeLessThanOrEqual(1.0);
    expect(bounds.zeroClipping).toBe(true);
  });

  test('T2-BND-08: Polyphony Saturation & 24-Voice Stealing', async () => {
    const synth = createPianoSynthesizer(44100);

    // Trigger 32 simultaneous distinct notes (exceeding 24-voice limit)
    for (let note = 36; note < 68; note++) {
      synth.noteOn({ midiNote: note, velocity: 90 });
    }

    const buffer = new Float32Array(8820); // 200ms
    renderMono(synth, buffer, 8820);

    const bounds = verifySampleBounds(buffer);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
    expect(bounds.absMax).toBeLessThanOrEqual(1.0);
    expect(computeRMS(buffer)).toBeGreaterThan(0.01);
  });
});
