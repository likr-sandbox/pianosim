import { test, expect } from '@playwright/test';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import { computeRMS, verifySampleBounds } from '../helpers/audio-analysis';

test.describe('Tier 2: Boundary - Velocity Extremes', () => {
  test('T2-BND-04: Velocity Zero (0) Boundary', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.noteOn({ midiNote: 60, velocity: 0 });

    const buffer = new Float32Array(4096);
    renderMono(synth, buffer, 4096);

    const rms = computeRMS(buffer);
    expect(rms).toBeLessThanOrEqual(1e-6);
  });

  test('T2-BND-05: Velocity One (1, Pianississimo)', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.noteOn({ midiNote: 60, velocity: 1 });

    const buffer = new Float32Array(4096);
    renderMono(synth, buffer, 4096);

    const bounds = verifySampleBounds(buffer);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);

    const rms = computeRMS(buffer);
    expect(rms).toBeGreaterThan(0.0);
    expect(rms).toBeLessThanOrEqual(0.05);
  });

  test('T2-BND-06: Velocity 127 (Fortississimo) Peak Bound', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.noteOn({ midiNote: 60, velocity: 127 });

    const buffer = new Float32Array(8192);
    renderMono(synth, buffer, 8192);

    const bounds = verifySampleBounds(buffer);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
    expect(bounds.absMax).toBeLessThanOrEqual(1.0);
    expect(bounds.zeroClipping).toBe(true);
  });
});
