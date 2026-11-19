import { expect, test } from '@playwright/test';
import {
  sampleDestinationAudio,
  setSlider,
  setupErrorMonitoring,
  unlockAudioEngine,
  waitAudioBlocks,
  waitForAudioRms,
  waitForPrepIdle,
  waitForSilence,
} from './helpers/audio-diag';

test.describe('E2E Parameter Presets Suite', () => {
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

  test('Preset selection synchronizes Voicing and Space DOM controls', async ({ page }) => {
    // 1. Concert Grand (440.0 Hz)
    await page.getByTestId('preset-concert-grand').click();
    expect(Number(await page.getByTestId('voicing-hammer').inputValue())).toBeCloseTo(0.5, 2);
    expect(Number(await page.getByTestId('voicing-unison').inputValue())).toBeCloseTo(1.0, 2);
    expect(Number(await page.getByTestId('voicing-stretch').inputValue())).toBeCloseTo(1.0, 2);
    expect(Number(await page.getByTestId('voicing-length').inputValue())).toBeCloseTo(1.0, 2);
    expect(Number(await page.getByTestId('voicing-a4').inputValue())).toBeCloseTo(440.0, 1);
    expect(Number(await page.getByTestId('voicing-a4-number').inputValue())).toBeCloseTo(440.0, 1);
    await expect(page.getByTestId('voicing-a4-chip-440')).toHaveClass(/active/);
    await expect(page.getByTestId('trigger-a4')).toHaveText('A4 = 440.0 Hz');
    await expect(page.getByTestId('body-concert-grand')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('lid-open')).toHaveClass(/active/);
    await expect(page.getByTestId('perspective-player')).toHaveClass(/active/);
    expect(Number(await page.getByTestId('reverb-decay-slider').inputValue())).toBeCloseTo(1.6, 2);
    expect(Number(await page.getByTestId('reverb-mix-slider').inputValue())).toBe(18);
    await expect(page.getByTestId('preset-description')).toContainText('D-274');

    // 2. Intimate Upright (440.0 Hz)
    await page.getByTestId('preset-intimate-upright').click();
    expect(Number(await page.getByTestId('voicing-hammer').inputValue())).toBeCloseTo(0.25, 2);
    expect(Number(await page.getByTestId('voicing-unison').inputValue())).toBeCloseTo(0.6, 2);
    expect(Number(await page.getByTestId('voicing-stretch').inputValue())).toBeCloseTo(0.7, 2);
    expect(Number(await page.getByTestId('voicing-length').inputValue())).toBeCloseTo(0.75, 2);
    expect(Number(await page.getByTestId('voicing-a4').inputValue())).toBeCloseTo(440.0, 1);
    await expect(page.getByTestId('voicing-a4-chip-440')).toHaveClass(/active/);
    await expect(page.getByTestId('trigger-a4')).toHaveText('A4 = 440.0 Hz');
    await expect(page.getByTestId('body-upright-studio')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('lid-closed')).toHaveClass(/active/);
    await expect(page.getByTestId('perspective-close')).toHaveClass(/active/);
    expect(Number(await page.getByTestId('reverb-decay-slider').inputValue())).toBeCloseTo(0.85, 2);
    expect(Number(await page.getByTestId('reverb-mix-slider').inputValue())).toBe(12);
    await expect(page.getByTestId('preset-description')).toContainText('chamber');

    // 3. Bright Pop (442.0 Hz)
    await page.getByTestId('preset-bright-pop').click();
    expect(Number(await page.getByTestId('voicing-hammer').inputValue())).toBeCloseTo(0.8, 2);
    expect(Number(await page.getByTestId('voicing-unison').inputValue())).toBeCloseTo(1.4, 2);
    expect(Number(await page.getByTestId('voicing-stretch').inputValue())).toBeCloseTo(1.3, 2);
    expect(Number(await page.getByTestId('voicing-length').inputValue())).toBeCloseTo(1.15, 2);
    expect(Number(await page.getByTestId('voicing-a4').inputValue())).toBeCloseTo(442.0, 1);
    expect(Number(await page.getByTestId('voicing-a4-number').inputValue())).toBeCloseTo(442.0, 1);
    await expect(page.getByTestId('voicing-a4-chip-442')).toHaveClass(/active/);
    await expect(page.getByTestId('trigger-a4')).toHaveText('A4 = 442.0 Hz');
    await expect(page.getByTestId('body-concert-grand')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('lid-open')).toHaveClass(/active/);
    await expect(page.getByTestId('perspective-close')).toHaveClass(/active/);
    expect(Number(await page.getByTestId('reverb-decay-slider').inputValue())).toBeCloseTo(1.25, 2);
    expect(Number(await page.getByTestId('reverb-mix-slider').inputValue())).toBe(22);
    await expect(page.getByTestId('preset-description')).toContainText('modern');

    // 4. Vintage Saloon (435.0 Hz)
    await page.getByTestId('preset-vintage-saloon').click();
    expect(Number(await page.getByTestId('voicing-hammer').inputValue())).toBeCloseTo(0.7, 2);
    expect(Number(await page.getByTestId('voicing-unison').inputValue())).toBeCloseTo(2.5, 2);
    expect(Number(await page.getByTestId('voicing-stretch').inputValue())).toBeCloseTo(0.2, 2);
    expect(Number(await page.getByTestId('voicing-length').inputValue())).toBeCloseTo(0.65, 2);
    expect(Number(await page.getByTestId('voicing-a4').inputValue())).toBeCloseTo(435.0, 1);
    expect(Number(await page.getByTestId('voicing-a4-number').inputValue())).toBeCloseTo(435.0, 1);
    expect(await page.locator('.btn-a4-chip.active').count()).toBe(0);
    await expect(page.getByTestId('trigger-a4')).toHaveText('A4 = 435.0 Hz');
    await expect(page.getByTestId('body-vintage-saloon')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('lid-half')).toHaveClass(/active/);
    await expect(page.getByTestId('perspective-player')).toHaveClass(/active/);
    expect(Number(await page.getByTestId('reverb-decay-slider').inputValue())).toBeCloseTo(0.55, 2);
    expect(Number(await page.getByTestId('reverb-mix-slider').inputValue())).toBe(8);
    await expect(page.getByTestId('preset-description')).toContainText('saloon');
  });

  test('Preset dropdown selector synchronizes with segmented buttons', async ({ page }) => {
    const select = page.getByTestId('preset-select');
    await select.selectOption('bright-pop');

    await expect(page.getByTestId('preset-bright-pop')).toHaveClass(/active/);
    expect(Number(await page.getByTestId('voicing-hammer').inputValue())).toBeCloseTo(0.8, 2);

    await select.selectOption('intimate-upright');
    await expect(page.getByTestId('preset-intimate-upright')).toHaveClass(/active/);
    expect(Number(await page.getByTestId('voicing-hammer').inputValue())).toBeCloseTo(0.25, 2);
  });

  test('Preset switching during a sounding chord causes zero NaNs, clicks, or dropouts', async ({
    page,
  }) => {
    // Hold sustain pedal down and use forte strike velocity
    await setSlider(page, 'velocity-slider', 112);
    await setSlider(page, 'sustain-slider', 1.0);

    // Strike/re-strike [48, 60, 64, 67] right before each preset swap so each swap
    // measures one PrepWorker transition without cumulative decay across 4 presets.
    const presetIds = ['intimate-upright', 'bright-pop', 'vintage-saloon', 'concert-grand'];
    for (const pid of presetIds) {
      for (const key of [48, 60, 64, 67]) {
        const keyEl = page.locator(`[data-midi="${key}"]`);
        await keyEl.dispatchEvent('pointerdown', { buttons: 1 });
        await keyEl.dispatchEvent('pointerup', { buttons: 0 });
      }
      await waitAudioBlocks(page, 16);
      await waitForAudioRms(page, 0.003);

      await page.getByTestId(`preset-${pid}`).click();
      await waitForPrepIdle(page);

      const stats = await sampleDestinationAudio(page);
      expect(stats.nonFiniteCount).toBe(0);
      expect(stats.peak).toBeLessThanOrEqual(1.0);
      expect(stats.rms).toBeGreaterThan(0.001);
    }
  });

  test('Acoustic spectral difference: bright-pop produces higher centroid and high-freq ratio than intimate-upright', async ({
    page,
  }) => {
    // 1. Measure Intimate Upright
    await page.getByTestId('preset-intimate-upright').click();
    await waitForPrepIdle(page);

    const a4Key = page.locator('[data-midi="69"]');
    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await waitAudioBlocks(page, 64);
    const intimateStats = await sampleDestinationAudio(page);
    await a4Key.dispatchEvent('pointerup', { buttons: 0 });

    expect(intimateStats.nonFiniteCount).toBe(0);
    expect(intimateStats.rms).toBeGreaterThan(0.003);

    // Reset silence
    await page.getByTestId('panic-btn').click();
    await waitForSilence(page);

    // 2. Measure Bright Pop
    await page.getByTestId('preset-bright-pop').click();
    await waitForPrepIdle(page);

    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await waitAudioBlocks(page, 64);
    const brightStats = await sampleDestinationAudio(page);
    await a4Key.dispatchEvent('pointerup', { buttons: 0 });

    expect(brightStats.nonFiniteCount).toBe(0);
    expect(brightStats.rms).toBeGreaterThan(0.003);

    // Bright pop has hardened hammers (0.80 vs 0.25) and open lid (vs closed)
    // yielding higher spectral centroid and high-frequency ratio
    expect(brightStats.spectralCentroidHz).toBeGreaterThan(intimateStats.spectralCentroidHz);
    expect(brightStats.highFreqRatio).toBeGreaterThan(intimateStats.highFreqRatio);
  });

  test('Modifying sliders or Concert Pitch transitions preset indicator to custom mode', async ({
    page,
  }) => {
    await page.getByTestId('preset-concert-grand').click();
    await expect(page.getByTestId('preset-custom-badge')).toBeHidden();

    // Modify hammer hardness slider
    await setSlider(page, 'voicing-hammer', 0.95);

    // Custom badge must now be visible
    await expect(page.getByTestId('preset-custom-badge')).toBeVisible();

    // Segmented buttons should no longer be active
    const activeButtons = await page.locator('.preset-segmented .btn-segmented.active').count();
    expect(activeButtons).toBe(0);

    // Description reflects custom mode
    await expect(page.getByTestId('preset-description')).toContainText('Custom');

    // Selecting a preset resets custom mode and updates A4
    await page.getByTestId('preset-bright-pop').click();
    await expect(page.getByTestId('preset-custom-badge')).toBeHidden();
    await expect(page.getByTestId('preset-bright-pop')).toHaveClass(/active/);
    await expect(page.getByTestId('trigger-a4')).toHaveText('A4 = 442.0 Hz');

    // Changing Concert Pitch via quick chip also transitions to Custom
    await page.getByTestId('voicing-a4-chip-432').click();
    await expect(page.getByTestId('preset-custom-badge')).toBeVisible();
    await expect(page.getByTestId('trigger-a4')).toHaveText('A4 = 432.0 Hz');
  });

  test('I-1: rapid concurrent preset switching applies voicing detune/stretch without staging corruption', async ({
    page,
  }) => {
    const e7Key = page.locator('[data-midi="100"]');

    // 1. Baseline E7 under Concert Grand (A4=440, stretch=1.0, lengthScale=1.0)
    await page.getByTestId('preset-concert-grand').click();
    await waitForPrepIdle(page);
    await e7Key.dispatchEvent('pointerdown', { buttons: 1 });
    await waitAudioBlocks(page, 80);
    const concertStats = await sampleDestinationAudio(page, { minFreq: 2400, maxFreq: 3100 });
    await e7Key.dispatchEvent('pointerup', { buttons: 0 });
    expect(concertStats.nonFiniteCount).toBe(0);
    expect(concertStats.rms).toBeGreaterThan(0.0005);

    await page.getByTestId('panic-btn').click();
    await waitForSilence(page);

    // 2. Rapidly switch presets concurrently ending on Vintage Saloon (A4=435, stretch=0.2, lengthScale=0.65, detune=2.5)
    // so spaceBlob and voicingBlob contend for WasmContext::staging.
    await page.getByTestId('preset-bright-pop').click();
    await page.getByTestId('preset-intimate-upright').click();
    await page.getByTestId('preset-vintage-saloon').click();
    await waitForPrepIdle(page);

    await e7Key.dispatchEvent('pointerdown', { buttons: 1 });
    await waitAudioBlocks(page, 80);
    const saloonStats = await sampleDestinationAudio(page, { minFreq: 2400, maxFreq: 3100 });
    await e7Key.dispatchEvent('pointerup', { buttons: 0 });
    expect(saloonStats.nonFiniteCount).toBe(0);
    expect(saloonStats.peak).toBeLessThanOrEqual(1.0);
    expect(saloonStats.rms).toBeGreaterThan(0.0005);
    // Vintage Saloon's voicingBlob (short scale 0.65 / stretch 0.20) shifts E7 > 45 Hz above Concert Grand
    expect(saloonStats.peakFreqHz - concertStats.peakFreqHz).toBeGreaterThan(45);

    await page.getByTestId('panic-btn').click();
    await waitForSilence(page);

    // 3. Rapidly switch back concurrently ending on Bright Pop (A4=442, stretch=1.3, lengthScale=1.15)
    await page.getByTestId('preset-intimate-upright').click();
    await page.getByTestId('preset-concert-grand').click();
    await page.getByTestId('preset-bright-pop').click();
    await waitForPrepIdle(page);

    await e7Key.dispatchEvent('pointerdown', { buttons: 1 });
    await waitAudioBlocks(page, 80);
    const brightStats = await sampleDestinationAudio(page, { minFreq: 2400, maxFreq: 3100 });
    await e7Key.dispatchEvent('pointerup', { buttons: 0 });
    expect(brightStats.nonFiniteCount).toBe(0);
    expect(brightStats.peak).toBeLessThanOrEqual(1.0);
    expect(brightStats.rms).toBeGreaterThan(0.0005);
    expect(saloonStats.peakFreqHz - brightStats.peakFreqHz).toBeGreaterThan(45);
  });
});
