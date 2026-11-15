import { expect, test } from '@playwright/test';
import {
  expectAudioDecay,
  sampleDestinationAudio,
  setupErrorMonitoring,
  unlockAudioEngine,
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
    await page.keyboard.down('KeyH');
    await expect(a4Key).toHaveClass(/is-active/);
    await page.waitForTimeout(420);

    const held = await sampleDestinationAudio(page, 400, 500);
    expect(held.nonFiniteCount).toBe(0);
    expect(held.peak).toBeLessThanOrEqual(1.0);
    expect(held.rms).toBeGreaterThan(0.002);
    expect(held.peakFreqHz).toBeGreaterThanOrEqual(435);
    expect(held.peakFreqHz).toBeLessThanOrEqual(445);

    await page.keyboard.up('KeyH');
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
    await page.waitForTimeout(300);

    const activeAudio = await sampleDestinationAudio(page);
    expect(activeAudio.rms).toBeGreaterThan(0.001);

    await page.getByTestId('panic-btn').click();
    await expect(a4Key).not.toHaveClass(/is-active/);
    await expectAudioDecay(page, activeAudio.rms, 0.15, 2000);

    await page.keyboard.up('KeyH');
    assertNoErrors();
  });
});
