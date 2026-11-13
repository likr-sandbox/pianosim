/**
 * Binding to the REAL production engine (src/audio/PianoSynthesizer.ts).
 * No mock/stub: every acoustic assertion renders audio through this class.
 */

import { PianoSynthesizer } from '../../../src/audio/PianoSynthesizer';
export { PianoSynthesizer };
export type { NoteOnOptions, NoteOffOptions } from '../../../src/audio/PianoSynthesizer';

export function createPianoSynthesizer(sampleRate = 44100): PianoSynthesizer {
  return new PianoSynthesizer(sampleRate);
}

/**
 * Render `numSamples` through the real synth with distinct L/R buffers and write the
 * mono mixdown (L+R)/2 into `out` (for single-channel DSP analysis). Returns [L, R].
 */
export function renderMono(
  synth: PianoSynthesizer,
  out: Float32Array,
  numSamples: number = out.length
): [Float32Array, Float32Array] {
  const left = new Float32Array(numSamples);
  const right = new Float32Array(numSamples);
  synth.render(left, right, numSamples);
  for (let i = 0; i < numSamples; i++) out[i] = 0.5 * (left[i] + right[i]);
  return [left, right];
}
