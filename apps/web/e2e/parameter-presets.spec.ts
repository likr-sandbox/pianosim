import { expect, test } from '@playwright/test';
import {
  sampleDestinationAudio,
  setSlider,
  setupErrorMonitoring,
  unlockAudioEngine,
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
    await expect(page.getByTestId('lid-open')).toHaveClass(/active/);
    await expect(page.getByTestId('perspective-player')).toHaveClass(/active/);
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
    await expect(page.getByTestId('lid-closed')).toHaveClass(/active/);
    await expect(page.getByTestId('perspective-close')).toHaveClass(/active/);
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
    await expect(page.getByTestId('lid-open')).toHaveClass(/active/);
    await expect(page.getByTestId('perspective-close')).toHaveClass(/active/);
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
    await expect(page.getByTestId('lid-half')).toHaveClass(/active/);
    await expect(page.getByTestId('perspective-player')).toHaveClass(/active/);
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
    // Hold sustain pedal down
    await setSlider(page, 'sustain-slider', 1.0);

    // Strike C major triad (C4=60, E4=64, G4=67)
    for (const key of [60, 64, 67]) {
      const keyEl = page.locator(`[data-midi="${key}"]`);
      await keyEl.dispatchEvent('pointerdown', { buttons: 1 });
      await page.waitForTimeout(50);
      await keyEl.dispatchEvent('pointerup', { buttons: 0 });
    }

    await page.waitForTimeout(150);
    const initialStats = await sampleDestinationAudio(page);
    expect(initialStats.rms).toBeGreaterThan(0.003);

    // Rapidly cycle through presets while chord rings out
    const presetIds = ['intimate-upright', 'bright-pop', 'vintage-saloon', 'concert-grand'];
    for (const pid of presetIds) {
      await page.getByTestId(`preset-${pid}`).click();
      await page.waitForTimeout(150);

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
    await page.waitForTimeout(200); // Allow prep worker and crossfade to settle

    const a4Key = page.locator('[data-midi="69"]');
    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(300);
    const intimateStats = await sampleDestinationAudio(page);
    await a4Key.dispatchEvent('pointerup', { buttons: 0 });

    expect(intimateStats.nonFiniteCount).toBe(0);
    expect(intimateStats.rms).toBeGreaterThan(0.003);

    // Reset silence
    await page.getByTestId('panic-btn').click();
    await page.waitForTimeout(200);

    // 2. Measure Bright Pop
    await page.getByTestId('preset-bright-pop').click();
    await page.waitForTimeout(200);

    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(300);
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
});
