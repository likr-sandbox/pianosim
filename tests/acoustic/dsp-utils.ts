/**
 * tests/acoustic/dsp-utils.ts
 * Pure TypeScript DSP utility library for acoustic verification testing.
 * Implements in-place Radix-2 Cooley-Tukey FFT, log-magnitude parabolic peak interpolation,
 * RMS amplitude calculation, spectral centroid tracking, decay rate analysis, and stability scanning.
 */

import { PianoSynthesizer } from '../../src/audio/PianoSynthesizer';

/**
 * Standard MIDI note number to frequency conversion:
 * f = 440 * 2^((midiNote - 69) / 12)
 */
export function midiToFrequency(midiNote: number): number {
  return 440.0 * Math.pow(2.0, (midiNote - 69.0) / 12.0);
}

/**
 * Pure TypeScript in-place Radix-2 Cooley-Tukey Fast Fourier Transform (FFT).
 * Modifies real and imag Float32Arrays in-place. Buffer length N must be a power of 2.
 */
export function fft(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  if ((n & (n - 1)) !== 0) {
    throw new Error(`FFT size must be a power of 2, received ${n}`);
  }

  // Bit reversal permutation
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      const tr = real[i]; real[i] = real[j]; real[j] = tr;
      const ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
    }
    let k = n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  // Cooley-Tukey butterfly combinations
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = (-2.0 * Math.PI) / len;
    const wStepR = Math.cos(angle);
    const wStepI = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wr = 1.0;
      let wi = 0.0;
      for (let m = 0; m < half; m++) {
        const uR = real[i + m];
        const uI = imag[i + m];
        const vR = real[i + m + half] * wr - imag[i + m + half] * wi;
        const vI = real[i + m + half] * wi + imag[i + m + half] * wr;
        real[i + m] = uR + vR;
        imag[i + m] = uI + vI;
        real[i + m + half] = uR - vR;
        imag[i + m + half] = uI - vI;
        const nextWr = wr * wStepR - wi * wStepI;
        wi = wr * wStepI + wi * wStepR;
        wr = nextWr;
      }
    }
  }
}

/**
 * Detects fundamental frequency f0 within [minFreq, maxFreq] using 3-term Blackman windowing,
 * FFT, and 3-point parabolic peak interpolation on log-magnitude spectrum.
 */
export function detectPeakFrequency(
  buffer: Float32Array,
  sampleRate: number,
  minFreq: number,
  maxFreq: number,
  fftSize = 16384
): { detectedFreq: number; peakMagnitude: number; peakBin: number } {
  const n = fftSize;
  const real = new Float32Array(n);
  const imag = new Float32Array(n);

  const copyLen = Math.min(buffer.length, n);
  // Apply 3-term Blackman window to minimize spectral leakage (sidelobes < -58 dB)
  for (let i = 0; i < copyLen; i++) {
    const w =
      0.42 -
      0.5 * Math.cos((2.0 * Math.PI * i) / (n - 1)) +
      0.08 * Math.cos((4.0 * Math.PI * i) / (n - 1));
    real[i] = buffer[i] * w;
  }

  fft(real, imag);

  const numBins = n >> 1;
  const mag = new Float32Array(numBins);
  for (let k = 0; k < numBins; k++) {
    mag[k] = Math.sqrt(real[k] * real[k] + imag[k] * imag[k]);
  }

  const minBin = Math.max(1, Math.floor((minFreq * n) / sampleRate));
  const maxBin = Math.min(numBins - 2, Math.ceil((maxFreq * n) / sampleRate));

  let peakBin = minBin;
  let maxMag = mag[minBin];
  for (let k = minBin; k <= maxBin; k++) {
    if (mag[k] > maxMag) {
      maxMag = mag[k];
      peakBin = k;
    }
  }

  // 3-point parabolic peak interpolation on log-magnitude (dB)
  const eps = 1e-12;
  const alpha = 20.0 * Math.log10(Math.max(mag[peakBin - 1], eps));
  const beta = 20.0 * Math.log10(Math.max(mag[peakBin], eps));
  const gamma = 20.0 * Math.log10(Math.max(mag[peakBin + 1], eps));

  const denom = alpha - 2.0 * beta + gamma;
  const p = Math.abs(denom) > 1e-9 ? 0.5 * ((alpha - gamma) / denom) : 0;

  const detectedFreq = (peakBin + p) * (sampleRate / n);
  return { detectedFreq, peakMagnitude: maxMag, peakBin };
}

/**
 * Computes Root Mean Square (RMS) amplitude of a slice of samples.
 */
export function computeRMS(
  buffer: Float32Array,
  startIndex = 0,
  length = buffer.length - startIndex
): number {
  if (length <= 0) return 0;
  let sumSq = 0;
  const end = Math.min(startIndex + length, buffer.length);
  for (let i = startIndex; i < end; i++) {
    const s = buffer[i];
    sumSq += s * s;
  }
  return Math.sqrt(sumSq / (end - startIndex));
}

/**
 * Computes Spectral Centroid (frequency center of mass) using Hann windowing.
 */
export function computeSpectralCentroid(
  buffer: Float32Array,
  sampleRate: number,
  fftSize = 8192
): number {
  const n = fftSize;
  const real = new Float32Array(n);
  const imag = new Float32Array(n);
  const copyLen = Math.min(buffer.length, n);

  for (let i = 0; i < copyLen; i++) {
    const w = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (n - 1)));
    real[i] = buffer[i] * w;
  }

  fft(real, imag);

  const numBins = n >> 1;
  let weightedSum = 0;
  let totalMagnitude = 0;

  for (let k = 1; k < numBins; k++) {
    const mag = Math.sqrt(real[k] * real[k] + imag[k] * imag[k]);
    const freq = (k * sampleRate) / n;
    weightedSum += freq * mag;
    totalMagnitude += mag;
  }

  return totalMagnitude > 0 ? weightedSum / totalMagnitude : 0;
}

/**
 * Computes decay rate in dB/second between two timestamps t1Sec and t2Sec.
 */
export function computeDecayRateDbPerSec(
  buffer: Float32Array,
  sampleRate: number,
  t1Sec: number,
  t2Sec: number,
  windowSec = 0.05
): number {
  const windowLen = Math.floor(windowSec * sampleRate);
  const idx1 = Math.floor(t1Sec * sampleRate);
  const idx2 = Math.floor(t2Sec * sampleRate);

  const rms1 = Math.max(computeRMS(buffer, idx1, windowLen), 1e-12);
  const rms2 = Math.max(computeRMS(buffer, idx2, windowLen), 1e-12);

  const deltaDb = 20.0 * Math.log10(rms1 / rms2);
  return deltaDb / (t2Sec - t1Sec);
}

/**
 * Evaluates signal numerical stability: checks for NaN, Inf, peak absolute amplitude,
 * DC bias, and active signal presence (RMS).
 */
export function checkSignalStability(buffer: Float32Array): {
  hasNaN: boolean;
  hasInf: boolean;
  peak: number;
  dcOffset: number;
  rms: number;
} {
  let hasNaN = false;
  let hasInf = false;
  let peak = 0;
  let sum = 0;
  let sumSq = 0;

  const n = buffer.length;
  for (let i = 0; i < n; i++) {
    const s = buffer[i];
    if (Number.isNaN(s)) hasNaN = true;
    if (!Number.isFinite(s)) hasInf = true;
    const abs = Math.abs(s);
    if (abs > peak) peak = abs;
    sum += s;
    sumSq += s * s;
  }

  return {
    hasNaN,
    hasInf,
    peak,
    dcOffset: Math.abs(sum / n),
    rms: Math.sqrt(sumSq / n),
  };
}

export interface RenderEvents {
  noteOffTimeSec?: number;
  midiNote?: number;
  sustainPedalEvents?: Array<{ timeSec: number; down: boolean }>;
}

/**
 * Renders PianoSynthesizer in 128-sample blocks into a mono Float32Array offline.
 */
export function renderSynthToMono(
  synth: PianoSynthesizer,
  durationSec: number,
  sampleRate: number,
  events?: RenderEvents
): Float32Array {
  const totalSamples = Math.floor(durationSec * sampleRate);
  const mono = new Float32Array(totalSamples);
  const blockSize = 128;
  const leftBlock = new Float32Array(blockSize);
  const rightBlock = new Float32Array(blockSize);

  let noteOffFired = false;
  let offset = 0;

  while (offset < totalSamples) {
    const chunk = Math.min(blockSize, totalSamples - offset);
    const currentTime = offset / sampleRate;

    // Handle noteOff trigger
    if (
      events?.noteOffTimeSec !== undefined &&
      currentTime >= events.noteOffTimeSec &&
      !noteOffFired
    ) {
      synth.noteOff({ midiNote: events.midiNote ?? 60 });
      noteOffFired = true;
    }

    // Handle sustain pedal triggers
    if (events?.sustainPedalEvents) {
      for (const ev of events.sustainPedalEvents) {
        if (Math.abs(currentTime - ev.timeSec) < (chunk / sampleRate)) {
          synth.setSustainPedal(ev.down);
        }
      }
    }

    synth.render(leftBlock, rightBlock, chunk);

    for (let i = 0; i < chunk; i++) {
      mono[offset + i] = 0.5 * (leftBlock[i] + rightBlock[i]);
    }
    offset += chunk;
  }

  return mono;
}
