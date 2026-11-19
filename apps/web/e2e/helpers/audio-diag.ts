import fs from 'node:fs';
import { expect, type Page } from '@playwright/test';

export interface StrikeAttackStats {
  readonly attackRms: number;
  readonly attackPeak: number;
  readonly spectralCentroidHz: number;
  readonly highFreqRatio: number;
  readonly nonFiniteCount: number;
}

export interface HoldReleaseStats {
  readonly sampleRate: number;
  readonly initRms: number;
  readonly decayedRms: number;
  readonly decayRatio: number;
  readonly peak: number;
  readonly nonFiniteCount: number;
}

interface CapturedWav {
  readonly sampleRate: number;
  readonly mono: Float32Array;
  readonly peak: number;
  readonly nonFiniteCount: number;
  readonly onsetFrame: number;
}

export interface AudioSampleStats {
  readonly sampleRate: number;
  readonly rms: number;
  readonly peak: number;
  readonly nonFiniteCount: number;
  readonly peakFreqHz: number;
  readonly spectralCentroidHz: number;
  readonly highFreqRatio: number;
}

export interface AudioSampleOptions {
  minFreq?: number;
  maxFreq?: number;
}

/** Samples real destination audio directly from window.__PIANO_DIAG__.analyser tap. */
export async function sampleDestinationAudio(
  page: Page,
  minFreqOrOptions: number | AudioSampleOptions = 0,
  maxFreqHz = Infinity,
): Promise<AudioSampleStats> {
  let minFreq = 0;
  let maxFreq = maxFreqHz;
  if (typeof minFreqOrOptions === 'object' && minFreqOrOptions !== null) {
    minFreq = minFreqOrOptions.minFreq ?? 0;
    maxFreq = minFreqOrOptions.maxFreq ?? Infinity;
  } else {
    minFreq = minFreqOrOptions;
  }
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
    { minFreq, maxFreq },
  );
}

/** Unlocks real Wasm AudioWorklet on user gesture and asserts running state. */
export async function unlockAudioEngine(page: Page): Promise<void> {
  await page.getByTestId('unlock-audio').click();
  await expect
    .poll(
      async () => {
        return page.evaluate(() => {
          const d = window.__PIANO_DIAG__;
          return d ? `${d.status}:${d.contextState}:${d.telemetryBlocksReceived > 0}` : 'missing';
        });
      },
      { timeout: 15_000 },
    )
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

/** Polls until telemetry.blockCount advances by at least deltaBlocks (AudioWorklet clock). */
export async function waitAudioBlocks(page: Page, deltaBlocks: number): Promise<number> {
  return page.evaluate((delta) => {
    return new Promise<number>((resolve, reject) => {
      const start = window.__PIANO_DIAG__?.telemetry.blockCount ?? 0;
      const target = start + delta;
      const deadline = performance.now() + 10_000;
      const tick = () => {
        const diag = window.__PIANO_DIAG__;
        if (!diag) {
          reject(new Error('Diagnostic state missing while waiting for audio blocks'));
          return;
        }
        const cur = diag.telemetry.blockCount;
        if (cur >= target) {
          resolve(cur);
          return;
        }
        if (performance.now() > deadline) {
          reject(new Error(`Timed out waiting for ${delta} audio blocks (${cur}/${target})`));
          return;
        }
        setTimeout(tick, 10);
      };
      tick();
    });
  }, deltaBlocks);
}

/** Polls destination RMS until output decays below threshold after panic/release. */
export async function waitForSilence(
  page: Page,
  thresholdRms = 0.0005,
  timeoutMs = 3000,
): Promise<void> {
  await expect
    .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: timeoutMs })
    .toBeLessThan(thresholdRms);
}

/** Polls until Prep Worker space/voicing/perf blobs are committed in the AudioWorklet, then settles blocks. */
export async function waitForPrepIdle(page: Page, settleBlocks = 16): Promise<void> {
  await expect
    .poll(
      async () =>
        page.evaluate(() => {
          const d = window.__PIANO_DIAG__;
          return d
            ? d.appliedSpaceSeq >= d.requestedSpaceSeq &&
                d.appliedVoicingSeq >= d.requestedVoicingSeq &&
                d.appliedPerfSeq >= d.requestedPerfSeq
            : false;
        }),
      { timeout: 10_000, intervals: [15] },
    )
    .toBe(true);
  if (settleBlocks > 0) {
    await waitAudioBlocks(page, settleBlocks);
  }
}

/** Polls destination audio until RMS exceeds threshold with finite amplitude <= 1.0. */
export async function waitForAudioRms(
  page: Page,
  minRms = 0.002,
  timeoutMs = 4000,
): Promise<AudioSampleStats> {
  let last: AudioSampleStats | null = null;
  await expect
    .poll(
      async () => {
        last = await sampleDestinationAudio(page);
        expect(last.nonFiniteCount).toBe(0);
        expect(last.peak).toBeLessThanOrEqual(1.0);
        return last.rms;
      },
      { timeout: timeoutMs, intervals: [20] },
    )
    .toBeGreaterThan(minRms);
  return last!;
}

async function stopAndReadMonoWav(page: Page): Promise<CapturedWav> {
  await page.getByTestId('record-stop').click();
  await expect(page.getByTestId('record-status')).toContainText('Ready', { timeout: 3000 });

  const [dl] = await Promise.all([
    page.waitForEvent('download'),
    page.getByTestId('record-export').click(),
  ]);
  const path = await dl.path();
  if (!path) throw new Error('Missing WAV download path');
  const buf = fs.readFileSync(path);
  expect(buf.subarray(0, 4).toString('ascii')).toBe('RIFF');
  const sampleRate = buf.readUInt32LE(24);
  const frames = buf.readUInt32LE(40) >>> 3;
  const mono = new Float32Array(frames);
  let peak = 0;
  let nonFiniteCount = 0;
  let onsetFrame = -1;
  for (let i = 0; i < frames; i++) {
    const l = buf.readFloatLE(44 + i * 8);
    const r = buf.readFloatLE(48 + i * 8);
    if (!Number.isFinite(l) || !Number.isFinite(r)) {
      nonFiniteCount++;
      continue;
    }
    const maxCh = Math.max(Math.abs(l), Math.abs(r));
    if (maxCh > peak) peak = maxCh;
    mono[i] = 0.5 * (l + r);
    if (onsetFrame < 0 && maxCh > 0.002) {
      onsetFrame = Math.max(0, i - 16);
    }
  }
  expect(onsetFrame).toBeGreaterThanOrEqual(0);
  return { sampleRate, mono, peak, nonFiniteCount, onsetFrame };
}

function windowRms(samples: Float32Array, start: number, len: number): number {
  let sumSq = 0;
  for (let i = start; i < start + len; i++) {
    sumSq += samples[i] * samples[i];
  }
  return Math.sqrt(sumSq / len);
}

/** Captures an onset-aligned 85 ms attack window via the live f32 AudioWorklet WAV recorder. */
export async function captureStrikeAttack(
  page: Page,
  midi: number,
  minFreqHz: number,
  unaCorda: boolean,
): Promise<StrikeAttackStats> {
  await page.getByTestId('panic-btn').click();
  if (unaCorda) {
    await page.getByTestId('pedal-una-corda').click();
    expect(await page.getByTestId('pedal-una-corda').getAttribute('aria-pressed')).toBe('true');
  }
  await waitForSilence(page);

  await page.getByTestId('record-format').selectOption('f32');
  await page.getByTestId('record-start').click();
  await expect(page.getByTestId('record-stop')).toBeEnabled();

  const key = page.locator(`[data-midi="${midi}"]`);
  await key.dispatchEvent('pointerdown', { buttons: 1 });
  await waitAudioBlocks(page, 80);
  await key.dispatchEvent('pointerup', { buttons: 0 });
  const wav = await stopAndReadMonoWav(page);
  const t0 = wav.onsetFrame;
  expect(wav.mono.length - t0).toBeGreaterThanOrEqual(4096);

  let attackPeak = 0;
  for (let i = t0; i < t0 + 4096; i++) {
    attackPeak = Math.max(attackPeak, Math.abs(wav.mono[i]));
  }
  const n = 2048;
  const binW = wav.sampleRate / n;
  let totLin = 0;
  let wFreq = 0;
  let totEnergy = 0;
  let highEnergy = 0;
  const maxBin = Math.min(n / 2 - 1, Math.ceil(12000 / binW));
  for (let k = Math.max(1, Math.floor(minFreqHz / binW)); k <= maxBin; k++) {
    let re = 0;
    let im = 0;
    const step = (2 * Math.PI * k) / n;
    for (let i = 0; i < n; i++) {
      const x = wav.mono[t0 + i] * 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
      re += x * Math.cos(step * i);
      im -= x * Math.sin(step * i);
    }
    const lin = Math.hypot(re, im) / (n * 0.25);
    if (lin > 1e-5) {
      const freq = k * binW;
      wFreq += freq * lin;
      totLin += lin;
      const e = lin * lin;
      totEnergy += e;
      if (freq >= 4000) {
        highEnergy += e;
      }
    }
  }
  return {
    attackRms: windowRms(wav.mono, t0, 4096),
    attackPeak,
    spectralCentroidHz: totLin > 0 ? wFreq / totLin : 0,
    highFreqRatio: totEnergy > 1e-12 ? highEnergy / totEnergy : 0,
    nonFiniteCount: wav.nonFiniteCount,
  };
}

/** Captures sample-exact hold & post-release decay RMS via live f32 AudioWorklet WAV recording. */
export async function captureHoldReleaseWindow(
  page: Page,
  midi: number,
  sustain: number,
  holdBlocks = 112,
  releaseBlocks = 192,
): Promise<HoldReleaseStats> {
  await page.getByTestId('panic-btn').click();
  await setSlider(page, 'sustain-slider', sustain);
  await waitForSilence(page);

  await page.getByTestId('record-format').selectOption('f32');
  await page.getByTestId('record-start').click();
  await expect(page.getByTestId('record-stop')).toBeEnabled();

  const key = page.locator(`[data-midi="${midi}"]`);
  const bDown = await waitAudioBlocks(page, 1);
  await key.dispatchEvent('pointerdown', { buttons: 1 });
  const bUp = await waitAudioBlocks(page, holdBlocks);
  await key.dispatchEvent('pointerup', { buttons: 0 });
  await waitAudioBlocks(page, releaseBlocks);

  const wav = await stopAndReadMonoWav(page);
  const sr = wav.sampleRate;
  const t0 = wav.onsetFrame;
  const holdFrames = (bUp - bDown) * 128;
  const winLen = 2048;
  const initStart = t0 + holdFrames - Math.round(0.04 * sr) - winLen;
  const decayStart = t0 + holdFrames + Math.round(0.35 * sr);

  expect(initStart).toBeGreaterThanOrEqual(t0);
  expect(decayStart + winLen).toBeLessThanOrEqual(wav.mono.length);

  const initRms = windowRms(wav.mono, initStart, winLen);
  const decayedRms = windowRms(wav.mono, decayStart, winLen);
  return {
    sampleRate: sr,
    initRms,
    decayedRms,
    decayRatio: decayedRms / initRms,
    peak: wav.peak,
    nonFiniteCount: wav.nonFiniteCount,
  };
}
