import { describe, it, expect } from 'vitest';

/**
 * Standard MIDI to frequency calculation formula:
 * f = 440 * 2^((midiNote - 69) / 12)
 */
function midiToFrequency(midiNote: number): number {
  return 440 * Math.pow(2, (midiNote - 69) / 12);
}

describe('Acoustic Suite Sanity & Toolchain Verification', () => {
  it('vitest test runner executes properly in environment', () => {
    expect(true).toBe(true);
  });

  it('verifies standard piano fundamental frequencies (A4 = 440 Hz, A0 = 27.5 Hz, C4 = 261.63 Hz)', () => {
    // A4 (MIDI note 69): standard concert pitch 440.0 Hz
    const a4 = midiToFrequency(69);
    expect(a4).toBeCloseTo(440.0, 4);

    // A0 (MIDI note 21): lowest note on 88-key piano 27.5 Hz
    const a0 = midiToFrequency(21);
    expect(a0).toBeCloseTo(27.5, 4);

    // Middle C / C4 (MIDI note 60): ~261.6256 Hz
    const c4 = midiToFrequency(60);
    expect(c4).toBeCloseTo(261.6256, 3);

    // C8 (MIDI note 108): highest note on 88-key piano ~4186.009 Hz
    const c8 = midiToFrequency(108);
    expect(c8).toBeCloseTo(4186.009, 2);
  });

  it('verifies Float32Array and audio sample buffer numerical operations', () => {
    const bufferSize = 128;
    const buffer = new Float32Array(bufferSize);

    // Synthesize a brief 440 Hz test tone at 44100 Hz sample rate
    const sampleRate = 44100;
    const frequency = 440;
    for (let i = 0; i < bufferSize; i++) {
      buffer[i] = Math.sin((2 * Math.PI * frequency * i) / sampleRate);
    }

    // Verify numerical stability: no NaN, no Inf, bounded within [-1.0, 1.0]
    let hasNaN = false;
    let hasInf = false;
    let peak = 0;

    for (let i = 0; i < bufferSize; i++) {
      const sample = buffer[i];
      if (Number.isNaN(sample)) hasNaN = true;
      if (!Number.isFinite(sample)) hasInf = true;
      const abs = Math.abs(sample);
      if (abs > peak) peak = abs;
    }

    expect(hasNaN).toBe(false);
    expect(hasInf).toBe(false);
    expect(peak).toBeLessThanOrEqual(1.0);
    expect(peak).toBeGreaterThan(0.0);
  });
});
