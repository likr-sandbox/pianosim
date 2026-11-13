import { test, expect } from '@playwright/test';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import {
  computeRMS,
  computeDynamicRangeDB,
  computeSpectralCentroid,
  verifySampleBounds,
} from '../helpers/audio-analysis';

test.describe('Tier 1: Feature Coverage - Velocity Response & Dynamic Range', () => {
  test('T1-VEL-01: Monotonic RMS Amplitude Scaling', async () => {
    const synth = createPianoSynthesizer(44100);

    const getRMSForVelocity = (velocity: number): number => {
      synth.allNotesOff();
      synth.noteOn({ midiNote: 60, velocity });
      const buf = new Float32Array(4096);
      renderMono(synth, buf, 4096);
      return computeRMS(buf, 0, 2048);
    };

    const rms30 = getRMSForVelocity(30);
    const rms70 = getRMSForVelocity(70);
    const rms120 = getRMSForVelocity(120);

    expect(rms70).toBeGreaterThan(rms30);
    expect(rms120).toBeGreaterThan(rms70);
  });

  test('T1-VEL-02: Dynamic Range Span (>12 dB)', async () => {
    const synth = createPianoSynthesizer(44100);

    synth.noteOn({ midiNote: 60, velocity: 20 });
    const bufLow = new Float32Array(4096);
    renderMono(synth, bufLow, 4096);
    const rmsLow = computeRMS(bufLow, 0, 2048);

    synth.allNotesOff();
    synth.noteOn({ midiNote: 60, velocity: 127 });
    const bufHigh = new Float32Array(4096);
    renderMono(synth, bufHigh, 4096);
    const rmsHigh = computeRMS(bufHigh, 0, 2048);

    const dynamicRange = computeDynamicRangeDB(rmsHigh, rmsLow);
    expect(dynamicRange).toBeGreaterThanOrEqual(12.0);
  });

  test('T1-VEL-03: Spectral Centroid Brightness Shift', async () => {
    const synth = createPianoSynthesizer(44100);

    synth.noteOn({ midiNote: 60, velocity: 30 });
    const bufLow = new Float32Array(8192);
    renderMono(synth, bufLow, 8192);
    const centroidLow = computeSpectralCentroid(bufLow, 44100);

    synth.allNotesOff();
    synth.noteOn({ midiNote: 60, velocity: 120 });
    const bufHigh = new Float32Array(8192);
    renderMono(synth, bufHigh, 8192);
    const centroidHigh = computeSpectralCentroid(bufHigh, 44100);

    const ratio = centroidHigh / centroidLow;
    expect(ratio).toBeGreaterThanOrEqual(1.3);
  });

  test('T1-VEL-04: Velocity 0 Silent Handling', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.noteOn({ midiNote: 60, velocity: 0 });

    const buf = new Float32Array(2048);
    renderMono(synth, buf, 2048);

    const rms = computeRMS(buf);
    expect(rms).toBeLessThan(1e-6);
  });

  test('T1-VEL-05: Maximum Velocity Non-Clipping', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.noteOn({ midiNote: 60, velocity: 127 });

    const buf = new Float32Array(8192);
    renderMono(synth, buf, 8192);

    const bounds = verifySampleBounds(buf);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
    expect(bounds.absMax).toBeLessThanOrEqual(1.0);
    expect(bounds.zeroClipping).toBe(true);
  });
});
