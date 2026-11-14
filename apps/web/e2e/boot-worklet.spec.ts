import { expect, test, type Page } from '@playwright/test';

interface AudioSampleStats {
  readonly sampleRate: number;
  readonly rms: number;
  readonly peak: number;
  readonly nonFiniteCount: number;
  readonly peakFreqHz: number;
}

async function sampleDestinationAudio(page: Page): Promise<AudioSampleStats> {
  return page.evaluate(() => {
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
      if (ax > peak) {
        peak = ax;
      }
      sumSq += x * x;
    }
    const rms = Math.sqrt(sumSq / n);

    const freqBins = new Float32Array(analyser.frequencyBinCount);
    analyser.getFloatFrequencyData(freqBins);

    let maxBin = 1;
    let maxDb = -Infinity;
    for (let k = 1; k < freqBins.length - 1; k++) {
      if (freqBins[k] > maxDb) {
        maxDb = freqBins[k];
        maxBin = k;
      }
    }

    // Parabolic interpolation on log-magnitude spectrum for sub-bin accuracy.
    const y0 = freqBins[maxBin - 1];
    const y1 = freqBins[maxBin];
    const y2 = freqBins[maxBin + 1];
    const denom = y0 - 2 * y1 + y2;
    const delta = Number.isFinite(denom) && Math.abs(denom) > 1e-12 ? (0.5 * (y0 - y2)) / denom : 0;
    const clampedDelta = Math.max(-0.5, Math.min(0.5, delta));
    const peakFreqHz = ((maxBin + clampedDelta) * diag.sampleRate) / analyser.fftSize;

    return {
      sampleRate: diag.sampleRate,
      rms,
      peak,
      nonFiniteCount,
      peakFreqHz,
    };
  });
}

test.describe('M1 Wasm AudioWorklet Vertical Slice', () => {
  test('unlocks real Wasm AudioWorklet, plays A4 at 440Hz, decays on release, handles repeat & sustain', async ({
    page,
  }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(`pageerror: ${err.message}`));
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        errors.push(`console.error: ${msg.text()}`);
      }
    });

    await page.goto('./');

    await expect(page.getByTestId('unlock-audio')).toBeVisible();
    await expect(page.locator('.piano-key')).toHaveCount(13);

    // Unlock audio engine and verify AudioWorklet telemetry starts flowing.
    await page.getByTestId('unlock-audio').click();
    await expect
      .poll(async () => {
        return page.evaluate(() => {
          const d = window.__PIANO_DIAG__;
          return d ? `${d.status}:${d.contextState}:${d.telemetryBlocksReceived > 0}` : 'missing';
        });
      })
      .toBe('ready:running:true');

    await expect(page.getByTestId('hud-status')).toHaveText('Ready');

    const baseline = await sampleDestinationAudio(page);
    expect(baseline.nonFiniteCount).toBe(0);
    expect(baseline.peak).toBeLessThanOrEqual(1.0);
    expect(baseline.rms).toBeLessThan(1e-4);

    // Key-repeat events must be suppressed and not trigger notes.
    await page.evaluate(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyA', key: 'a', repeat: true, bubbles: true }));
    });
    await expect(page.locator('[data-midi="60"]')).not.toHaveClass(/is-active/);
    const afterRepeat = await sampleDestinationAudio(page);
    expect(afterRepeat.rms).toBeLessThan(1e-4);

    // Hold A4 key via pointer down and inspect real AudioContext destination output.
    const a4Key = page.locator('[data-midi="69"]');
    const box = await a4Key.boundingBox();
    expect(box).not.toBeNull();
    await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2);
    await page.mouse.down();

    // Allow AnalyserNode (16384 samples ~341-371 ms) to fill with active A4 audio.
    await page.waitForTimeout(420);

    const held = await sampleDestinationAudio(page);
    expect(held.nonFiniteCount).toBe(0);
    expect(held.peak).toBeLessThanOrEqual(1.0);
    expect(held.rms).toBeGreaterThan(0.002);
    expect(held.peakFreqHz).toBeGreaterThanOrEqual(435);
    expect(held.peakFreqHz).toBeLessThanOrEqual(445);

    const activeVoicesText = await page.getByTestId('hud-voices').textContent();
    expect(Number(activeVoicesText)).toBeGreaterThanOrEqual(1);

    // Release A4 and verify note decays at destination.
    await page.mouse.up();
    await expect
      .poll(
        async () => {
          const stats = await sampleDestinationAudio(page);
          expect(stats.nonFiniteCount).toBe(0);
          expect(stats.peak).toBeLessThanOrEqual(1.0);
          return stats.rms;
        },
        { timeout: 3000 },
      )
      .toBeLessThan(held.rms * 0.25);

    // Toggle Sustain with Space and verify PC keyboard A4 (KeyH) sustains across keyup.
    await page.keyboard.press('Space');
    await expect(page.getByTestId('sustain-toggle')).toHaveAttribute('aria-pressed', 'true');

    await page.keyboard.down('KeyH');
    await page.waitForTimeout(420);
    await page.keyboard.up('KeyH');
    await page.waitForTimeout(250);

    const sustained = await sampleDestinationAudio(page);
    expect(sustained.nonFiniteCount).toBe(0);
    expect(sustained.peak).toBeLessThanOrEqual(1.0);
    expect(sustained.rms).toBeGreaterThan(0.001);

    // Release Sustain with Space and verify decay.
    await page.keyboard.press('Space');
    await expect(page.getByTestId('sustain-toggle')).toHaveAttribute('aria-pressed', 'false');

    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3000 })
      .toBeLessThan(sustained.rms * 0.25);

    // Verify dedicated A4 trigger button produces output cleanly.
    await page.getByTestId('trigger-a4').click();
    await page.waitForTimeout(420);
    const pulse = await sampleDestinationAudio(page);
    expect(pulse.nonFiniteCount).toBe(0);
    expect(pulse.peak).toBeLessThanOrEqual(1.0);
    expect(pulse.rms).toBeGreaterThan(0.001);
    expect(pulse.peakFreqHz).toBeGreaterThanOrEqual(435);
    expect(pulse.peakFreqHz).toBeLessThanOrEqual(445);

    await page.getByTestId('panic-btn').click();
    expect(errors).toEqual([]);
  });
});
