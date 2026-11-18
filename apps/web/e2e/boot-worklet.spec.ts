import { expect, test } from '@playwright/test';
import {
  expectAudioDecay,
  sampleDestinationAudio,
  setupErrorMonitoring,
  unlockAudioEngine,
  waitAudioBlocks,
} from './helpers/audio-diag';

test.describe('Wasm AudioWorklet Boot & Baseline Suite', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('./');
  });

  test('initial state before unlock is locked and uninitialized', async ({ page }) => {
    const assertNoErrors = setupErrorMonitoring(page);

    await expect(page.getByTestId('unlock-audio')).toBeVisible();
    await expect(page.getByTestId('hud-status')).toHaveText('Locked');

    const diag = await page.evaluate(() => window.__PIANO_DIAG__);
    expect(diag?.status).toBe('locked');
    expect(diag?.analyser == null).toBe(true);

    assertNoErrors();
  });

  test('unlocks AudioWorklet on user gesture, streams telemetry, and confirms sample rate', async ({
    page,
  }) => {
    const assertNoErrors = setupErrorMonitoring(page);

    await unlockAudioEngine(page);
    await expect(page.getByTestId('hud-status')).toHaveText('Ready');

    const diag = await page.evaluate(() => window.__PIANO_DIAG__);
    expect([44100, 48000, 96000]).toContain(diag?.sampleRate);
    expect(diag?.telemetryBlocksReceived).toBeGreaterThan(0);
    expect(diag?.telemetry.callbackUs).toBeGreaterThanOrEqual(0);

    const baseline = await sampleDestinationAudio(page);
    expect(baseline.nonFiniteCount).toBe(0);
    expect(baseline.peak).toBeLessThanOrEqual(1.0);
    expect(baseline.rms).toBeLessThan(1e-4);

    assertNoErrors();
  });

  test('A4 acoustic baseline produces 440 Hz fundamental and decays on key release', async ({ page }) => {
    const assertNoErrors = setupErrorMonitoring(page);
    await unlockAudioEngine(page);

    const a4Key = page.locator('[data-midi="69"]');
    const box = await a4Key.boundingBox();
    expect(box).not.toBeNull();

    await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height * 0.8);
    await page.mouse.down();
    await waitAudioBlocks(page, 96);

    const held = await sampleDestinationAudio(page, 400, 500);
    expect(held.nonFiniteCount).toBe(0);
    expect(held.peak).toBeLessThanOrEqual(1.0);
    expect(held.rms).toBeGreaterThan(0.002);
    expect(held.peakFreqHz).toBeGreaterThanOrEqual(435);
    expect(held.peakFreqHz).toBeLessThanOrEqual(445);

    await page.mouse.up();
    await expectAudioDecay(page, held.rms);

    assertNoErrors();
  });

  test('OS key-repeat events are suppressed and do not trigger duplicate notes', async ({ page }) => {
    const assertNoErrors = setupErrorMonitoring(page);
    await unlockAudioEngine(page);

    await page.evaluate(() => {
      window.dispatchEvent(
        new KeyboardEvent('keydown', { code: 'KeyA', key: 'a', repeat: true, bubbles: true }),
      );
    });

    await expect(page.locator('[data-midi="60"]')).not.toHaveClass(/is-active/);
    const audio = await sampleDestinationAudio(page);
    expect(audio.rms).toBeLessThan(1e-4);

    assertNoErrors();
  });

  test('panic button immediately stops sounding notes and silences audio output', async ({ page }) => {
    const assertNoErrors = setupErrorMonitoring(page);
    await unlockAudioEngine(page);

    // Trigger A4 via PC keyboard KeyH to establish sounding tone.
    const a4Key = page.locator('[data-midi="69"]');
    await page.keyboard.down('KeyH');
    await waitAudioBlocks(page, 96);

    const activeAudio = await sampleDestinationAudio(page);
    expect(activeAudio.rms).toBeGreaterThan(0.001);

    await page.getByTestId('panic-btn').click();
    await expect(a4Key).not.toHaveClass(/is-active/);
    await expectAudioDecay(page, activeAudio.rms, 0.15, 2000);

    await page.keyboard.up('KeyH');
    assertNoErrors();
  });

  test('auto-unlocks at 48 kHz on queued A4 trigger and reports honest SPL/Peak/Max HUD', async ({
    page,
  }) => {
    const assertNoErrors = setupErrorMonitoring(page);

    // Click A4 test button directly from locked state without clicking Unlock Audio first.
    await expect(page.getByTestId('hud-status')).toHaveText('Locked');
    await page.getByTestId('trigger-a4').click();

    await expect
      .poll(
        async () => {
          return page.evaluate(() => {
            const d = window.__PIANO_DIAG__;
            return d ? `${d.status}:${d.contextState}` : 'missing';
          });
        },
        { timeout: 15_000 },
      )
      .toBe('ready:running');

    await expect(page.getByTestId('hud-sample-rate')).toHaveText('48000 Hz');

    // Queued A4 trigger fires cleanly once pipeline is ready.
    await expect
      .poll(async () => parseInt((await page.getByTestId('hud-voices').textContent()) ?? '0', 10), {
        timeout: 3000,
      })
      .toBeGreaterThanOrEqual(1);

    await waitAudioBlocks(page, 48);
    const audio = await sampleDestinationAudio(page, 400, 500);
    expect(audio.nonFiniteCount).toBe(0);
    expect(audio.peak).toBeLessThanOrEqual(1.0);
    expect(audio.rms).toBeGreaterThan(0.002);
    expect(audio.peakFreqHz).toBeGreaterThanOrEqual(435);
    expect(audio.peakFreqHz).toBeLessThanOrEqual(445);

    // Verify HUD Peak Hold / RMS / SPL calibration (-18 dBFS == 82.0 dB SPL) and true max/overrun fields.
    await expect
      .poll(async () => {
        return page.evaluate(() => {
          const d = window.__PIANO_DIAG__;
          return d && Number.isFinite(d.splDb) && d.peakHoldDbfs > -60 ? 1 : 0;
        });
      })
      .toBe(1);

    const hudCheck = await page.evaluate(() => {
      const d = window.__PIANO_DIAG__!;
      const fillEl = document.querySelector<HTMLElement>('[data-testid="hud-level-fill"]')!;
      const peakText = document.querySelector('[data-testid="hud-peak-db"]')?.textContent ?? '';
      const rmsText = document.querySelector('[data-testid="hud-rms-db"]')?.textContent ?? '';
      const splText = document.querySelector('[data-testid="hud-spl-db"]')?.textContent ?? '';
      const maxText = document.querySelector('[data-testid="hud-max-us"]')?.textContent ?? '';
      const overrunsText = document.querySelector('[data-testid="hud-overruns"]')?.textContent ?? '';
      const clipClass = document.querySelector('[data-testid="hud-clip"]')?.className ?? '';
      return {
        sampleRate: d.sampleRate,
        peakHoldDbfs: d.peakHoldDbfs,
        splDb: d.splDb,
        fillPct: parseFloat(fillEl.style.width || '0'),
        peakText,
        rmsText,
        splText,
        maxUs: parseFloat(maxText),
        overruns: parseInt(overrunsText, 10),
        isClipping: clipClass.includes('is-clipping'),
      };
    });

    expect(hudCheck.sampleRate).toBe(48000);
    expect(Math.abs(hudCheck.splDb - (hudCheck.peakHoldDbfs + 100.0))).toBeLessThan(1e-5);
    expect(hudCheck.fillPct).toBeGreaterThan(0);
    expect(hudCheck.peakText).toMatch(/^-?\d+\.\d+ dB$/);
    expect(hudCheck.rmsText).toMatch(/^RMS -?\d+\.\d+ dBFS$/);
    expect(hudCheck.splText).toMatch(/^≈ \d+\.\d+ dB SPL$/);
    expect(Number.isFinite(hudCheck.maxUs) && hudCheck.maxUs >= 0).toBe(true);
    expect(Number.isInteger(hudCheck.overruns) && hudCheck.overruns >= 0).toBe(true);
    expect(hudCheck.isClipping).toBe(false);

    // Confirm auto-release after 750ms pulse decays audio cleanly.
    await expectAudioDecay(page, audio.rms, 0.25, 3000);
    assertNoErrors();
  });
});
