import { describe, it, expect } from 'vitest';
import { PianoSynthesizer } from '../../src/audio/PianoSynthesizer';
import {
  computeRMS,
  computeSpectralCentroid,
  renderSynthToMono,
} from './dsp-utils';

describe('Acoustic Suite: Velocity & Timbre Verification (velocity.test.ts)', () => {
  const sampleRate = 44100;

  it('exhibits significant RMS dynamic range (>= 12 dB) between pianissimo (v=25) and fortissimo (v=115)', () => {
    const synthLow = new PianoSynthesizer(sampleRate);
    synthLow.noteOn({ midiNote: 60, velocity: 25 });
    const bufLow = renderSynthToMono(synthLow, 0.8, sampleRate);

    const synthHigh = new PianoSynthesizer(sampleRate);
    synthHigh.noteOn({ midiNote: 60, velocity: 115 });
    const bufHigh = renderSynthToMono(synthHigh, 0.8, sampleRate);

    const rmsLow = computeRMS(bufLow, 0, Math.floor(0.5 * sampleRate));
    const rmsHigh = computeRMS(bufHigh, 0, Math.floor(0.5 * sampleRate));

    expect(rmsLow).toBeGreaterThan(0);
    expect(rmsHigh).toBeGreaterThan(rmsLow);

    const dynamicRangeDb = 20.0 * Math.log10(rmsHigh / rmsLow);
    // Dynamic range must be at least 12 dB
    expect(dynamicRangeDb).toBeGreaterThanOrEqual(12.0);
  });

  it('scales RMS amplitude monotonically across increasing velocity steps [20, 50, 80, 115]', () => {
    const velocities = [20, 50, 80, 115];
    const rmsValues: number[] = [];

    for (const vel of velocities) {
      const synth = new PianoSynthesizer(sampleRate);
      synth.noteOn({ midiNote: 69, velocity: vel });
      const buf = renderSynthToMono(synth, 0.5, sampleRate);
      const rms = computeRMS(buf, 0, Math.floor(0.4 * sampleRate));
      rmsValues.push(rms);
    }

    for (let i = 0; i < rmsValues.length - 1; i++) {
      expect(rmsValues[i + 1]).toBeGreaterThan(rmsValues[i]);
    }
  });

  it('increases spectral centroid by >= 1.25x (brighter harmonics) at high velocity vs low velocity', () => {
    const synthPianissimo = new PianoSynthesizer(sampleRate);
    synthPianissimo.noteOn({ midiNote: 60, velocity: 25 });
    const bufPianissimo = renderSynthToMono(synthPianissimo, 0.3, sampleRate);

    const synthFortissimo = new PianoSynthesizer(sampleRate);
    synthFortissimo.noteOn({ midiNote: 60, velocity: 115 });
    const bufFortissimo = renderSynthToMono(synthFortissimo, 0.3, sampleRate);

    // Initial 0.2s window captures hammer strike felt hardening
    const scLow = computeSpectralCentroid(
      bufPianissimo.subarray(0, Math.floor(0.2 * sampleRate)),
      sampleRate,
      8192
    );
    const scHigh = computeSpectralCentroid(
      bufFortissimo.subarray(0, Math.floor(0.2 * sampleRate)),
      sampleRate,
      8192
    );

    expect(scLow).toBeGreaterThan(0);
    expect(scHigh).toBeGreaterThan(0);

    const centroidRatio = scHigh / scLow;
    // Non-linear hammer model creates richer high harmonics at fortissimo
    expect(centroidRatio).toBeGreaterThanOrEqual(1.25);
  });
});
