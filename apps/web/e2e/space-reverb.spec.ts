import { expect, test } from '@playwright/test';
import {
  expectAudioDecay,
  sampleDestinationAudio,
  setSlider,
  setupErrorMonitoring,
  unlockAudioEngine,
  waitAudioBlocks,
  waitForPrepIdle,
  waitForSilence,
} from './helpers/audio-diag';

test.describe('E2E Multi-Body Space & Hall Reverb Suite', () => {
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

  test('III-2: Piano body switching via Prep Worker produces clean audio and distinct RMS/brightness profiles', async ({
    page,
  }) => {
    // Ensure dry reverb so body acoustic differences are measured directly
    await setSlider(page, 'reverb-mix-slider', 0);

    const bodies = [
      { testId: 'body-concert-grand', name: 'Concert Grand' },
      { testId: 'body-upright-studio', name: 'Upright Studio' },
      { testId: 'body-vintage-saloon', name: 'Vintage Saloon' },
    ] as const;

    const profiles: Array<{
      name: string;
      rms: number;
      centroidHz: number;
      highFreqRatio: number;
    }> = [];

    const a4Key = page.locator('[data-midi="69"]');

    for (const body of bodies) {
      await page.getByTestId(body.testId).click();
      await expect(page.getByTestId(body.testId)).toHaveAttribute('aria-pressed', 'true');
      await expect(page.getByTestId(body.testId)).toHaveClass(/active/);
      await waitForPrepIdle(page);

      await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
      await waitAudioBlocks(page, 80);
      const stats = await sampleDestinationAudio(page);
      await a4Key.dispatchEvent('pointerup', { buttons: 0 });

      expect(stats.nonFiniteCount).toBe(0);
      expect(stats.peak).toBeLessThanOrEqual(1.0);
      expect(stats.rms).toBeGreaterThan(0.002);

      profiles.push({
        name: body.name,
        rms: stats.rms,
        centroidHz: stats.spectralCentroidHz,
        highFreqRatio: stats.highFreqRatio,
      });

      await page.getByTestId('panic-btn').click();
      await waitForSilence(page);
    }

    // Verify all 3 physical body archetypes yield distinct RMS/brightness acoustic profiles
    for (let i = 0; i < profiles.length; i++) {
      for (let j = i + 1; j < profiles.length; j++) {
        const a = profiles[i];
        const b = profiles[j];
        const rmsDiffRel = Math.abs(a.rms - b.rms) / Math.max(a.rms, b.rms);
        const centroidDiffHz = Math.abs(a.centroidHz - b.centroidHz);
        const hfDiff = Math.abs(a.highFreqRatio - b.highFreqRatio);
        expect(rmsDiffRel > 0.02 || centroidDiffHz > 15 || hfDiff > 0.002).toBe(true);
      }
    }

    // Switching body transitions preset indicator to Custom mode
    await expect(page.getByTestId('preset-custom-badge')).toBeVisible();
    expect(await page.locator('.preset-segmented .btn-segmented.active').count()).toBe(0);
  });

  test('III-2: Live body switching under sustained chord executes clickless crossfade', async ({
    page,
  }) => {
    await setSlider(page, 'sustain-slider', 1.0);
    for (const midi of [60, 64, 67]) {
      await page.locator(`[data-midi="${midi}"]`).dispatchEvent('pointerdown', { buttons: 1 });
    }
    await waitAudioBlocks(page, 96);

    for (const bodyId of ['body-upright-studio', 'body-vintage-saloon', 'body-concert-grand']) {
      await page.getByTestId(bodyId).click();
      await waitForPrepIdle(page);
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

  test('III-1: Reverb Decay and Reverb Mix sliders extend post-release decay tail and clear active preset', async ({
    page,
  }) => {
    await page.getByTestId('preset-concert-grand').click();
    await expect(page.getByTestId('preset-custom-badge')).toBeHidden();
    await expect(page.getByTestId('reverb-decay-value')).toHaveText('1.60 s');
    await expect(page.getByTestId('reverb-mix-value')).toHaveText('18%');

    await setSlider(page, 'velocity-slider', 127);

    const strikeAndSampleTail = async () => {
      for (const midi of [60, 64, 67]) {
        await page.locator(`[data-midi="${midi}"]`).dispatchEvent('pointerdown', { buttons: 1 });
      }
      await waitAudioBlocks(page, 42);
      for (const midi of [60, 64, 67]) {
        await page.locator(`[data-midi="${midi}"]`).dispatchEvent('pointerup', { buttons: 0 });
      }
      await waitAudioBlocks(page, 160);
      return sampleDestinationAudio(page);
    };

    // 1. Dry baseline: Decay 0.0 s, Mix 0%
    await setSlider(page, 'reverb-decay-slider', 0.0);
    await setSlider(page, 'reverb-mix-slider', 0);
    await expect(page.getByTestId('reverb-decay-value')).toHaveText('0.00 s');
    await expect(page.getByTestId('reverb-mix-value')).toHaveText('0%');
    await expect(page.getByTestId('preset-custom-badge')).toBeVisible();
    expect(await page.locator('.preset-segmented .btn-segmented.active').count()).toBe(0);

    const dryTail = await strikeAndSampleTail();
    expect(dryTail.nonFiniteCount).toBe(0);
    expect(dryTail.peak).toBeLessThanOrEqual(1.0);

    await page.getByTestId('panic-btn').click();
    await waitForSilence(page);

    // 2. Zero-decay wet mix check: Decay 0.0 s, Mix 65%
    await setSlider(page, 'reverb-decay-slider', 0.0);
    await setSlider(page, 'reverb-mix-slider', 65);
    await expect(page.getByTestId('reverb-mix-value')).toHaveText('65%');

    const zeroDecayTail = await strikeAndSampleTail();
    expect(zeroDecayTail.nonFiniteCount).toBe(0);
    expect(zeroDecayTail.peak).toBeLessThanOrEqual(1.0);

    await page.getByTestId('panic-btn').click();
    await waitForSilence(page);

    // 3. Long hall reverb: Decay 3.0 s, Mix 65%
    await setSlider(page, 'reverb-decay-slider', 3.0);
    await setSlider(page, 'reverb-mix-slider', 65);
    await expect(page.getByTestId('reverb-decay-value')).toHaveText('3.00 s');
    await expect(page.getByTestId('reverb-mix-value')).toHaveText('65%');

    const wetTail = await strikeAndSampleTail();

    expect(wetTail.nonFiniteCount).toBe(0);
    expect(wetTail.peak).toBeLessThanOrEqual(1.0);
    expect(wetTail.rms).toBeGreaterThan(0.0005);
    expect(wetTail.rms).toBeGreaterThan(dryTail.rms * 2.5);
    expect(wetTail.rms).toBeGreaterThan(zeroDecayTail.rms * 2.0);

    // Panic button clears the active reverb tail cleanly
    await page.getByTestId('panic-btn').click();
    await expectAudioDecay(page, wetTail.rms, 0.15, 2000);
  });
});
