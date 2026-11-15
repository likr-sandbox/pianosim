import { expect, type Page } from '@playwright/test';

export interface AudioSampleStats {
  readonly sampleRate: number;
  readonly rms: number;
  readonly peak: number;
  readonly nonFiniteCount: number;
  readonly peakFreqHz: number;
  readonly spectralCentroidHz: number;
  readonly highFreqRatio: number;
}

/** Samples real destination audio directly from window.__PIANO_DIAG__.analyser tap. */
export async function sampleDestinationAudio(
  page: Page,
  minFreqHz = 0,
  maxFreqHz = Infinity,
): Promise<AudioSampleStats> {
  return page.evaluate(
    ({ minFreq, maxFreq }) => {
      const diag = window.__PIANO_DIAG__;
      if (!diag || !diag.analyser || diag.sampleRate <= 0) {
        throw new Error('AnalyserNode diagnostic tap is not initialized');
      }
      const analyser = diag.analyser;
      const n = analyser.fftSize;
      const timeData = new Float32Array(n);
      analyser.getFloatTimeDomainData(timeData);

      let sumSq = 0;
      let peak = 0;
      let nonFiniteCount = 0;
      for (let i = 0; i < n; i++) {
        const x = timeData[i];
        if (!Number.isFinite(x)) {
          nonFiniteCount += 1;
          continue;
        }
        const ax = Math.abs(x);
        if (ax > peak) peak = ax;
        sumSq += x * x;
      }
      const rms = Math.sqrt(sumSq / n);

      const freqBins = new Float32Array(analyser.frequencyBinCount);
      analyser.getFloatFrequencyData(freqBins);

      const binWidth = diag.sampleRate / analyser.fftSize;
      const minBin = Math.max(1, Math.floor(minFreq / binWidth));
      const maxBinLimit = Math.min(
        freqBins.length - 2,
        Math.ceil(maxFreq / binWidth),
      );
      let maxBin = minBin;
      let maxDb = -Infinity;
      let totalLin = 0;
      let weightedFreqSum = 0;
      let highEnergy = 0;
      let totalEnergy = 0;

      for (let k = minBin; k <= maxBinLimit; k++) {
        const db = freqBins[k];
        if (db > maxDb) {
          maxDb = db;
          maxBin = k;
        }
        if (db > -100) {
          const lin = Math.pow(10, db / 20);
          const freq = k * binWidth;
          weightedFreqSum += freq * lin;
          totalLin += lin;
          const e = lin * lin;
          totalEnergy += e;
          if (freq >= 4000) {
            highEnergy += e;
          }
        }
      }

      // Parabolic interpolation on log-magnitude spectrum for sub-bin accuracy.
      const y0 = freqBins[maxBin - 1];
      const y1 = freqBins[maxBin];
      const y2 = freqBins[maxBin + 1];
      const denom = y0 - 2 * y1 + y2;
      const delta =
        Number.isFinite(denom) && Math.abs(denom) > 1e-12
          ? (0.5 * (y0 - y2)) / denom
          : 0;
      const clampedDelta = Math.max(-0.5, Math.min(0.5, delta));
      const peakFreqHz =
        ((maxBin + clampedDelta) * diag.sampleRate) / analyser.fftSize;
      const spectralCentroidHz =
        totalLin > 1e-9 ? weightedFreqSum / totalLin : 0;
      const highFreqRatio = totalEnergy > 1e-12 ? highEnergy / totalEnergy : 0;

      return {
        sampleRate: diag.sampleRate,
        rms,
        peak,
        nonFiniteCount,
        peakFreqHz,
        spectralCentroidHz,
        highFreqRatio,
      };
    },
    { minFreq: minFreqHz, maxFreq: maxFreqHz },
  );
}

/** Unlocks real Wasm AudioWorklet on user gesture and asserts running state. */
export async function unlockAudioEngine(page: Page): Promise<void> {
  await page.getByTestId('unlock-audio').click();
  await expect
    .poll(async () => {
      return page.evaluate(() => {
        const d = window.__PIANO_DIAG__;
        return d ? `${d.status}:${d.contextState}:${d.telemetryBlocksReceived > 0}` : 'missing';
      });
    })
    .toBe('ready:running:true');
}

/** Asserts that audio decay drops below a factor of initial RMS within timeout. */
export async function expectAudioDecay(
  page: Page,
  initialRms: number,
  factor = 0.25,
  timeoutMs = 3000,
): Promise<void> {
  const targetRms = Math.max(1e-4, initialRms * factor);
  await expect
    .poll(
      async () => {
        const stats = await sampleDestinationAudio(page);
        expect(stats.nonFiniteCount).toBe(0);
        expect(stats.peak).toBeLessThanOrEqual(1.0);
        return stats.rms;
      },
      { timeout: timeoutMs },
    )
    .toBeLessThan(targetRms);
}

/** Attaches console and page error listeners and returns assertion callback. */
export function setupErrorMonitoring(page: Page): () => void {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(`pageerror: ${err.message}`));
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      errors.push(`console.error: ${msg.text()}`);
    }
  });
  return () => {
    expect(errors).toEqual([]);
  };
}

/** Updates an input range slider and dispatches input/change events. */
export async function setSlider(page: Page, testId: string, val: number): Promise<void> {
  const el = page.getByTestId(testId);
  await el.evaluate((input, v) => {
    const range = input as HTMLInputElement;
    range.value = String(v);
    range.dispatchEvent(new Event('input', { bubbles: true }));
    range.dispatchEvent(new Event('change', { bubbles: true }));
  }, val);
}
