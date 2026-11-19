import fs from 'node:fs';
import { expect, type Page } from '@playwright/test';

export interface OfflineWavStats {
  readonly sampleRate: number;
  readonly frames: number;
  readonly peak: number;
  readonly rms: number;
  readonly nonFiniteCount: number;
  readonly mono: Float32Array | null;
}

/** Polls offline-render-status until rendering finishes ('Playing' or 'Ready'). */
export async function waitForOfflineRenderDone(
  page: Page,
  timeoutMs = 90_000,
): Promise<string> {
  let lastStatus = '';
  await expect
    .poll(
      async () => {
        const txt = ((await page.getByTestId('offline-render-status').textContent()) ?? '').trim();
        lastStatus = txt;
        if (txt === 'Error') {
          throw new Error('Offline render entered Error state');
        }
        return txt === 'Playing' || txt === 'Ready';
      },
      { timeout: timeoutMs, intervals: [50] },
    )
    .toBe(true);
  return lastStatus;
}

/** Downloads rendered WAV via offline-export-wav and validates genuine RIFF/WAVE audio. */
export async function downloadAndVerifyOfflineWav(
  page: Page,
  format: 'pcm24' | 'f32' = 'f32',
): Promise<OfflineWavStats> {
  const formatSel = page.getByTestId('offline-export-format');
  const exportBtn = page.getByTestId('offline-export-wav');
  await expect(exportBtn).toBeEnabled({ timeout: 90_000 });
  await formatSel.selectOption(format);

  const [dl] = await Promise.all([page.waitForEvent('download'), exportBtn.click()]);
  const filePath = await dl.path();
  if (!filePath) throw new Error('Missing offline WAV download path');
  const buf = fs.readFileSync(filePath);

  expect(buf.subarray(0, 4).toString('ascii')).toBe('RIFF');
  expect(buf.subarray(8, 12).toString('ascii')).toBe('WAVE');
  expect(buf.subarray(12, 16).toString('ascii')).toBe('fmt ');
  const audioFmt = buf.readUInt16LE(20);
  const channels = buf.readUInt16LE(22);
  const sampleRate = buf.readUInt32LE(24);
  const blockAlign = buf.readUInt16LE(32);
  const bitsPerSample = buf.readUInt16LE(34);
  expect(channels).toBe(2);
  expect(sampleRate).toBe(48_000);
  expect(buf.subarray(36, 40).toString('ascii')).toBe('data');
  const dataBytes = buf.readUInt32LE(40);
  expect(buf.length).toBe(44 + dataBytes);

  if (format === 'f32') {
    expect(audioFmt).toBe(3);
    expect(blockAlign).toBe(8);
    expect(bitsPerSample).toBe(32);
    const frames = dataBytes >>> 3;
    const mono = new Float32Array(frames);
    let peak = 0;
    let sumSq = 0;
    let nonFiniteCount = 0;
    for (let i = 0; i < frames; i++) {
      const l = buf.readFloatLE(44 + i * 8);
      const r = buf.readFloatLE(48 + i * 8);
      if (!Number.isFinite(l) || !Number.isFinite(r)) {
        nonFiniteCount++;
        continue;
      }
      const m = 0.5 * (l + r);
      mono[i] = m;
      peak = Math.max(peak, Math.abs(l), Math.abs(r));
      sumSq += m * m;
    }
    return {
      sampleRate,
      frames,
      peak,
      rms: frames > 0 ? Math.sqrt(sumSq / frames) : 0,
      nonFiniteCount,
      mono,
    };
  }

  expect(audioFmt).toBe(1);
  expect(blockAlign).toBe(6);
  expect(bitsPerSample).toBe(24);
  const frames = Math.floor(dataBytes / 6);
  let peak = 0;
  let sumSq = 0;
  for (let i = 0; i < frames; i++) {
    const off = 44 + i * 6;
    const rawL = buf[off] | (buf[off + 1] << 8) | (buf[off + 2] << 16);
    const rawR = buf[off + 3] | (buf[off + 4] << 8) | (buf[off + 5] << 16);
    const sL = (rawL & 0x800000 ? rawL | ~0xffffff : rawL) / 8_388_608.0;
    const sR = (rawR & 0x800000 ? rawR | ~0xffffff : rawR) / 8_388_608.0;
    const m = 0.5 * (sL + sR);
    peak = Math.max(peak, Math.abs(sL), Math.abs(sR));
    sumSq += m * m;
  }
  return {
    sampleRate,
    frames,
    peak,
    rms: frames > 0 ? Math.sqrt(sumSq / frames) : 0,
    nonFiniteCount: 0,
    mono: null,
  };
}
