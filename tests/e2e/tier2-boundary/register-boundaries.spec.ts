import { test, expect } from '@playwright/test';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import { computeRMS, verifySampleBounds } from '../helpers/audio-analysis';

test.describe('Tier 2: Boundary - Register Boundaries', () => {
  test('T2-BND-01: Lowest Note A0 Stability & Delay Line Bounds', async () => {
    const synth = createPianoSynthesizer(44100);
    // MIDI 21 = A0 (27.5 Hz)
    synth.noteOn({ midiNote: 21, velocity: 100 });

    const buffer = new Float32Array(44100);
    renderMono(synth, buffer, 44100);

    const bounds = verifySampleBounds(buffer);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
    expect(bounds.absMax).toBeLessThanOrEqual(1.0);
    expect(computeRMS(buffer)).toBeGreaterThan(0.005);
  });

  test('T2-BND-02: Highest Note C8 Stability & Filter Poles', async () => {
    const synth = createPianoSynthesizer(44100);
    // MIDI 108 = C8 (4186 Hz)
    synth.noteOn({ midiNote: 108, velocity: 100 });

    const buffer = new Float32Array(44100);
    renderMono(synth, buffer, 44100);

    const bounds = verifySampleBounds(buffer);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
    expect(bounds.absMax).toBeLessThanOrEqual(1.0);
    // C8 (4186 Hz) undergoes physical high-frequency string damping and stereo panning.
    // Level is measured on the true (L+R)/2 mixdown of the real engine's stereo output
    // (the old 0.004 threshold was calibrated on an L/R-aliased buffer that doubled the level).
    expect(computeRMS(buffer, 0, 4096)).toBeGreaterThan(0.002);
  });

  test('T2-BND-03: Register Crossover Continuity', async () => {
    const synth = createPianoSynthesizer(44100);
    const crossoverPairs = [
      [21, 22],
      [39, 40],
      [54, 55],
      [88, 89],
    ];

    for (const [m1, m2] of crossoverPairs) {
      synth.allNotesOff();
      synth.noteOn({ midiNote: m1, velocity: 90 });
      synth.noteOn({ midiNote: m2, velocity: 90 });

      const buffer = new Float32Array(4096);
      renderMono(synth, buffer, 4096);

      const bounds = verifySampleBounds(buffer);
      expect(bounds.hasNaN).toBe(false);
      expect(bounds.hasInf).toBe(false);
      expect(bounds.absMax).toBeLessThanOrEqual(1.0);
    }
  });
});
