import { describe, it, expect } from 'vitest';
import { PianoSynthesizer } from '../../src/audio/PianoSynthesizer';
import {
  detectPeakFrequency,
  midiToFrequency,
  renderSynthToMono,
} from './dsp-utils';

describe('Acoustic Suite: Fundamental Pitch Verification (pitch.test.ts)', () => {
  const sampleRate = 44100;

  it('detects concert pitch A4 (440 Hz) with error < 0.23% (< 1 Hz)', () => {
    const synth = new PianoSynthesizer(sampleRate);
    synth.noteOn({ midiNote: 69, velocity: 90 }); // A4

    const buffer = renderSynthToMono(synth, 1.0, sampleRate);
    // Analyze steady-state window [0.1s, 0.5s]
    const analysisWindow = buffer.subarray(Math.floor(0.1 * sampleRate));
    const targetFreq = 440.0;

    const { detectedFreq } = detectPeakFrequency(
      analysisWindow,
      sampleRate,
      350,
      550,
      16384
    );

    const errorHz = Math.abs(detectedFreq - targetFreq);
    const relativeError = errorHz / targetFreq;

    expect(errorHz).toBeLessThan(1.0);
    expect(relativeError).toBeLessThan(0.01);
  });

  it('detects middle C / C4 (261.63 Hz) with error < 1.0%', () => {
    const synth = new PianoSynthesizer(sampleRate);
    synth.noteOn({ midiNote: 60, velocity: 90 }); // C4

    const buffer = renderSynthToMono(synth, 1.0, sampleRate);
    const analysisWindow = buffer.subarray(Math.floor(0.1 * sampleRate));
    const targetFreq = midiToFrequency(60); // 261.6256 Hz

    const { detectedFreq } = detectPeakFrequency(
      analysisWindow,
      sampleRate,
      200,
      320,
      16384
    );

    const relativeError = Math.abs(detectedFreq - targetFreq) / targetFreq;
    expect(relativeError).toBeLessThan(0.01);
  });

  it('detects treble note C7 (2093.00 Hz) with error < 1.0%', () => {
    const synth = new PianoSynthesizer(sampleRate);
    synth.noteOn({ midiNote: 96, velocity: 90 }); // C7

    const buffer = renderSynthToMono(synth, 1.0, sampleRate);
    const analysisWindow = buffer.subarray(Math.floor(0.05 * sampleRate));
    const targetFreq = midiToFrequency(96); // 2093.0045 Hz

    const { detectedFreq } = detectPeakFrequency(
      analysisWindow,
      sampleRate,
      1800,
      2400,
      16384
    );

    const relativeError = Math.abs(detectedFreq - targetFreq) / targetFreq;
    expect(relativeError).toBeLessThan(0.01);
  });

  it('detects lowest bass key A0 (27.50 Hz) with error < 1.0%', () => {
    const synth = new PianoSynthesizer(sampleRate);
    synth.noteOn({ midiNote: 21, velocity: 100 }); // A0

    const buffer = renderSynthToMono(synth, 1.5, sampleRate);
    const analysisWindow = buffer.subarray(Math.floor(0.1 * sampleRate));
    const targetFreq = 27.5;

    // Use 32768-point FFT for low-frequency resolution (df ≈ 1.35 Hz)
    const { detectedFreq } = detectPeakFrequency(
      analysisWindow,
      sampleRate,
      20,
      38,
      32768
    );

    const relativeError = Math.abs(detectedFreq - targetFreq) / targetFreq;
    expect(relativeError).toBeLessThan(0.01);
  });

  it('maintains accurate pitch tuning at 48000 Hz sample rate', () => {
    const sr48 = 48000;
    const synth = new PianoSynthesizer(sr48);
    synth.noteOn({ midiNote: 69, velocity: 90 }); // A4

    const buffer = renderSynthToMono(synth, 1.0, sr48);
    const analysisWindow = buffer.subarray(Math.floor(0.1 * sr48));

    const { detectedFreq } = detectPeakFrequency(
      analysisWindow,
      sr48,
      350,
      550,
      16384
    );

    const errorHz = Math.abs(detectedFreq - 440.0);
    expect(errorHz).toBeLessThan(1.0);
  });
});
