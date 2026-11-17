import { expect, test } from '@playwright/test';
import {
  sampleDestinationAudio,
  setSlider,
  setupErrorMonitoring,
  unlockAudioEngine,
} from './helpers/audio-diag';

test.describe('E2E Concert Pitch (A4 = 415–466 Hz) Suite', () => {
  let assertNoErrors: () => void;

  test.beforeEach(async ({ page }) => {
    assertNoErrors = setupErrorMonitoring(page);
    await page.goto('./');
    await unlockAudioEngine(page);
  });

  test.afterEach(async ({ page }) => {
    await page.getByTestId('panic-btn').click();
    assertNoErrors();
  });

  test('Synchronizes range slider, direct numeric input, quick chips, and header A4 label', async ({
    page,
  }) => {
    const slider = page.getByTestId('voicing-a4');
    const numInput = page.getByTestId('voicing-a4-number');
    const headerBtn = page.getByTestId('trigger-a4');

    // Initial nominal default (440.0 Hz)
    expect(Number(await slider.inputValue())).toBeCloseTo(440.0, 1);
    expect(Number(await numInput.inputValue())).toBeCloseTo(440.0, 1);
    await expect(page.getByTestId('voicing-a4-chip-440')).toHaveAttribute('aria-pressed', 'true');
    await expect(headerBtn).toHaveText('A4 = 440.0 Hz');

    // Quick-select preset chips: 415, 432, 442, 440 Hz
    for (const hz of [415, 432, 442, 440]) {
      await page.getByTestId(`voicing-a4-chip-${hz}`).click();
      expect(Number(await slider.inputValue())).toBeCloseTo(hz, 1);
      expect(Number(await numInput.inputValue())).toBeCloseTo(hz, 1);
      await expect(page.getByTestId(`voicing-a4-chip-${hz}`)).toHaveAttribute('aria-pressed', 'true');
      await expect(headerBtn).toHaveText(`A4 = ${hz.toFixed(1)} Hz`);
    }

    // Fractional 0.1 Hz slider step (440.5 Hz -> no preset chip active)
    await setSlider(page, 'voicing-a4', 440.5);
    expect(Number(await numInput.inputValue())).toBeCloseTo(440.5, 1);
    await expect(headerBtn).toHaveText('A4 = 440.5 Hz');
    expect(await page.locator('.btn-a4-chip.active').count()).toBe(0);

    // Direct numeric input in bounds (435.3 Hz)
    await numInput.fill('435.3');
    await numInput.dispatchEvent('change');
    expect(Number(await slider.inputValue())).toBeCloseTo(435.3, 1);
    await expect(headerBtn).toHaveText('A4 = 435.3 Hz');

    // Below minimum clamps to 415.0 Hz on commit
    await numInput.fill('300');
    await numInput.dispatchEvent('change');
    expect(Number(await slider.inputValue())).toBeCloseTo(415.0, 1);
    expect(Number(await numInput.inputValue())).toBeCloseTo(415.0, 1);
    await expect(page.getByTestId('voicing-a4-chip-415')).toHaveAttribute('aria-pressed', 'true');
    await expect(headerBtn).toHaveText('A4 = 415.0 Hz');

    // Above maximum clamps to 466.0 Hz on commit
    await numInput.fill('520');
    await numInput.dispatchEvent('change');
    expect(Number(await slider.inputValue())).toBeCloseTo(466.0, 1);
    expect(Number(await numInput.inputValue())).toBeCloseTo(466.0, 1);
    await expect(headerBtn).toHaveText('A4 = 466.0 Hz');
  });

  test('Acoustic retuning shifts real Wasm AudioWorklet A4 fundamental across 415, 440, and 466 Hz', async ({
    page,
  }) => {
    const a4Key = page.locator('[data-midi="69"]');

    // 1. Baroque 415.0 Hz
    await page.getByTestId('voicing-a4-chip-415').click();
    await page.waitForTimeout(220);

    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(380);
    const bar415 = await sampleDestinationAudio(page, 380, 500);
    await a4Key.dispatchEvent('pointerup', { buttons: 0 });

    expect(bar415.nonFiniteCount).toBe(0);
    expect(bar415.rms).toBeGreaterThan(0.002);
    expect(bar415.peakFreqHz).toBeGreaterThanOrEqual(409);
    expect(bar415.peakFreqHz).toBeLessThanOrEqual(421);

    await page.getByTestId('panic-btn').click();
    await page.waitForTimeout(200);

    // 2. Modern Standard 440.0 Hz
    await page.getByTestId('voicing-a4-chip-440').click();
    await page.waitForTimeout(220);

    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(380);
    const std440 = await sampleDestinationAudio(page, 380, 500);
    await a4Key.dispatchEvent('pointerup', { buttons: 0 });

    expect(std440.nonFiniteCount).toBe(0);
    expect(std440.rms).toBeGreaterThan(0.002);
    expect(std440.peakFreqHz).toBeGreaterThanOrEqual(434);
    expect(std440.peakFreqHz).toBeLessThanOrEqual(446);

    await page.getByTestId('panic-btn').click();
    await page.waitForTimeout(200);

    // 3. Chorton 466.0 Hz
    await setSlider(page, 'voicing-a4', 466.0);
    await page.waitForTimeout(220);

    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(380);
    const high466 = await sampleDestinationAudio(page, 380, 500);
    await a4Key.dispatchEvent('pointerup', { buttons: 0 });

    expect(high466.nonFiniteCount).toBe(0);
    expect(high466.rms).toBeGreaterThan(0.002);
    expect(high466.peakFreqHz).toBeGreaterThanOrEqual(460);
    expect(high466.peakFreqHz).toBeLessThanOrEqual(472);

    expect(bar415.peakFreqHz).toBeLessThan(std440.peakFreqHz - 18);
    expect(std440.peakFreqHz).toBeLessThan(high466.peakFreqHz - 18);
  });

  test('Live Concert Pitch sweep during sustained chord remains click-free and finite', async ({
    page,
  }) => {
    await setSlider(page, 'sustain-slider', 1.0);
    for (const midi of [60, 64, 67]) {
      await page.locator(`[data-midi="${midi}"]`).dispatchEvent('pointerdown', { buttons: 1 });
    }
    await page.waitForTimeout(250);

    for (const targetHz of [415.0, 466.0, 432.0, 440.0]) {
      await setSlider(page, 'voicing-a4', targetHz);
      await page.waitForTimeout(160);
      const stats = await sampleDestinationAudio(page);
      expect(stats.nonFiniteCount).toBe(0);
      expect(stats.peak).toBeLessThanOrEqual(1.0);
      expect(stats.rms).toBeGreaterThan(0.001);
    }

    for (const midi of [60, 64, 67]) {
      await page.locator(`[data-midi="${midi}"]`).dispatchEvent('pointerup', { buttons: 0 });
    }
    await setSlider(page, 'sustain-slider', 0.0);
  });
});
