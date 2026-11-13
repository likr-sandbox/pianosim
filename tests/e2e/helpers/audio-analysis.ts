/**
 * Audio Analysis DSP Utilities for pianosim E2E Verification
 * Implements windowed Radix-2 FFT, parabolic peak interpolation,
 * RMS energy calculation, spectral centroid, decay rate estimation,
 * and numerical bounds checking.
 */

/**
 * Standard MIDI to frequency formula: f = 440 * 2^((midi - 69) / 12)
 */
export function midiToFrequency(midiNote: number): number {
  return 440 * Math.pow(2, (midiNote - 69) / 12);
}

/**
 * Compute root-mean-square (RMS) of a buffer over [startSample, endSample).
 */
export function computeRMS(
  buffer: Float32Array,
  startSample = 0,
  endSample = buffer.length
): number {
  const start = Math.max(0, startSample);
  const end = Math.min(buffer.length, endSample);
  const count = end - start;
  if (count <= 0) return 0;

  let sumSquares = 0;
  for (let i = start; i < end; i++) {
    const val = buffer[i];
    sumSquares += val * val;
  }
  return Math.sqrt(sumSquares / count);
}

/**
 * Dynamic range in decibels between two RMS values: 20 * log10(rmsHigh / rmsLow)
 */
export function computeDynamicRangeDB(rmsHigh: number, rmsLow: number): number {
  if (rmsLow <= 0) return rmsHigh > 0 ? 100 : 0;
  return 20 * Math.log10(Math.max(rmsHigh, 1e-9) / Math.max(rmsLow, 1e-9));
}

/**
 * Apply Hann window in-place to reduce spectral leakage
 */
export function applyHannWindow(input: Float32Array, output: Float32Array, size: number): void {
  for (let i = 0; i < size; i++) {
    const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
    output[i] = (i < input.length ? input[i] : 0) * w;
  }
}

/**
 * Apply Blackman window in-place for higher dynamic range frequency separation
 */
export function applyBlackmanWindow(input: Float32Array, output: Float32Array, size: number): void {
  const a0 = 0.42;
  const a1 = 0.5;
  const a2 = 0.08;
  for (let i = 0; i < size; i++) {
    const w =
      a0 -
      a1 * Math.cos((2 * Math.PI * i) / (size - 1)) +
      a2 * Math.cos((4 * Math.PI * i) / (size - 1));
    output[i] = (i < input.length ? input[i] : 0) * w;
  }
}

/**
 * In-place Radix-2 Cooley-Tukey FFT.
 * Requires size to be a power of two.
 */
export function fftRadix2(
  real: Float64Array,
  imag: Float64Array,
  size: number
): void {
  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < size - 1; i++) {
    if (i < j) {
      const tempR = real[i];
      real[i] = real[j];
      real[j] = tempR;
      const tempI = imag[i];
      imag[i] = imag[j];
      imag[j] = tempI;
    }
    let k = size >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  // Butterfly computations
  for (let len = 2; len <= size; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wStepR = Math.cos(angle);
    const wStepI = Math.sin(angle);

    for (let i = 0; i < size; i += len) {
      let wR = 1.0;
      let wI = 0.0;
      for (let k = 0; k < halfLen; k++) {
        const idxEven = i + k;
        const idxOdd = idxEven + halfLen;

        const uR = real[idxEven];
        const uI = imag[idxEven];
        const vR = real[idxOdd] * wR - imag[idxOdd] * wI;
        const vI = real[idxOdd] * wI + imag[idxOdd] * wR;

        real[idxEven] = uR + vR;
        imag[idxEven] = uI + vI;
        real[idxOdd] = uR - vR;
        imag[idxOdd] = uI - vI;

        const nextWR = wR * wStepR - wI * wStepI;
        wI = wR * wStepI + wI * wStepR;
        wR = nextWR;
      }
    }
  }
}

/**
 * Calculate FFT magnitude spectrum.
 */
export function computeMagnitudeSpectrum(
  signal: Float32Array,
  fftSize: number
): Float64Array {
  const windowed = new Float32Array(fftSize);
  applyBlackmanWindow(signal, windowed, fftSize);

  const real = new Float64Array(fftSize);
  const imag = new Float64Array(fftSize);
  for (let i = 0; i < fftSize; i++) {
    real[i] = windowed[i];
    imag[i] = 0;
  }

  fftRadix2(real, imag, fftSize);

  const half = (fftSize >> 1) + 1;
  const mag = new Float64Array(half);
  for (let i = 0; i < half; i++) {
    mag[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
  }
  return mag;
}

/**
 * Detect peak fundamental frequency using Parabolic Peak Interpolation on log magnitudes.
 */
export function computeFFTPeak(
  signal: Float32Array,
  sampleRate: number,
  expectedFreq?: number,
  searchMarginFactor = 0.35,
  minFftSize = 16384
): number {
  let fftSize = minFftSize;
  while (fftSize < signal.length && fftSize < 65536) {
    fftSize <<= 1;
  }
  fftSize = Math.min(fftSize, 65536);

  const mag = computeMagnitudeSpectrum(signal, fftSize);
  const half = mag.length;
  const binResolution = sampleRate / fftSize;

  let minBin = 1;
  let maxBin = half - 2;

  if (expectedFreq && expectedFreq > 0) {
    const centerBin = Math.round(expectedFreq / binResolution);
    const margin = Math.max(4, Math.round((expectedFreq * searchMarginFactor) / binResolution));
    minBin = Math.max(1, centerBin - margin);
    maxBin = Math.min(half - 2, centerBin + margin);
  }

  // Find local maximum in search range
  let peakBin = minBin;
  let peakVal = -1;
  for (let i = minBin; i <= maxBin; i++) {
    if (mag[i] > peakVal) {
      peakVal = mag[i];
      peakBin = i;
    }
  }

  if (peakBin <= 0 || peakBin >= half - 1 || peakVal <= 1e-9) {
    return peakBin * binResolution;
  }

  // Parabolic interpolation on log-magnitudes
  const alpha = Math.log(Math.max(mag[peakBin - 1], 1e-12));
  const beta = Math.log(Math.max(mag[peakBin], 1e-12));
  const gamma = Math.log(Math.max(mag[peakBin + 1], 1e-12));

  const denom = alpha - 2 * beta + gamma;
  let delta = 0;
  if (Math.abs(denom) > 1e-12) {
    delta = (0.5 * (alpha - gamma)) / denom;
    // Bound delta to [-1, 1]
    delta = Math.max(-1, Math.min(1, delta));
  }

  return (peakBin + delta) * binResolution;
}

/**
 * Compute Spectral Centroid: sum(f_k * |X_k|) / sum(|X_k|)
 */
export function computeSpectralCentroid(
  buffer: Float32Array,
  sampleRate: number,
  fftSize = 8192
): number {
  const mag = computeMagnitudeSpectrum(buffer, fftSize);
  const half = mag.length;
  const binResolution = sampleRate / fftSize;

  let numerator = 0;
  let denominator = 0;

  for (let k = 1; k < half; k++) {
    const freq = k * binResolution;
    const m = mag[k];
    numerator += freq * m;
    denominator += m;
  }

  if (denominator <= 1e-12) return 0;
  return numerator / denominator;
}

/**
 * Compute decay rate in dB/second between two RMS energy points over deltaSeconds.
 * Decay rate = 20 * log10(rms1 / rms2) / deltaSeconds
 */
export function computeDecayRateDBPerSec(
  rmsStart: number,
  rmsEnd: number,
  deltaSeconds: number
): number {
  if (deltaSeconds <= 0) return 0;
  const safeStart = Math.max(rmsStart, 1e-9);
  const safeEnd = Math.max(rmsEnd, 1e-9);
  const dbDrop = 20 * Math.log10(safeStart / safeEnd);
  return dbDrop / deltaSeconds;
}

/**
 * Verify numerical stability: bounds in [-1.0, 1.0], 0 NaN, 0 Inf.
 */
export function verifySampleBounds(buffer: Float32Array): {
  min: number;
  max: number;
  absMax: number;
  hasNaN: boolean;
  hasInf: boolean;
  zeroClipping: boolean;
} {
  let min = Infinity;
  let max = -Infinity;
  let absMax = 0;
  let hasNaN = false;
  let hasInf = false;
  let zeroClipping = true;

  for (let i = 0; i < buffer.length; i++) {
    const val = buffer[i];
    if (Number.isNaN(val)) {
      hasNaN = true;
      continue;
    }
    if (!Number.isFinite(val)) {
      hasInf = true;
      continue;
    }
    if (val < min) min = val;
    if (val > max) max = val;
    const absVal = Math.abs(val);
    if (absVal > absMax) absMax = absVal;
    if (absVal > 1.0) zeroClipping = false;
  }

  return {
    min: Number.isFinite(min) ? min : 0,
    max: Number.isFinite(max) ? max : 0,
    absMax,
    hasNaN,
    hasInf,
    zeroClipping,
  };
}
