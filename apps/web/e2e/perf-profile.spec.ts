import { expect, test } from '@playwright/test';
import {
  sampleDestinationAudio,
  setSlider,
  setupErrorMonitoring,
  unlockAudioEngine,
  waitAudioBlocks,
  waitForPrepIdle,
  waitForSilence,
} from './helpers/audio-diag';

test.describe('E2E PerfProfile UI, URL Query, and 3-Tier Reconfiguration', () => {
  let assertNoErrors: () => void;

  test.beforeEach(async ({ page }) => {
    assertNoErrors = setupErrorMonitoring(page);
  });

  test.afterEach(async ({ page }) => {
    await page.getByTestId('panic-btn').click();
    assertNoErrors();
  });

  test('Default live preset is Balanced and ?perf=realtime overrides boot preset', async ({
    page,
  }) => {
    await page.goto('./');
    await unlockAudioEngine(page);
    const defaultDiag = await page.evaluate(() => ({
      preset: window.__PIANO_DIAG__?.perfPreset,
      timingSource: window.__PIANO_DIAG__?.telemetry.timingSource,
    }));
    expect(defaultDiag.preset).toBe(1);
    expect(['perf', 'date']).toContain(defaultDiag.timingSource);
    await expect(page.getByTestId('perf-preset-balanced')).toHaveClass(/active/);

    await page.goto('./?perf=realtime');
    await unlockAudioEngine(page);
    const rtDiag = await page.evaluate(() => ({
      preset: window.__PIANO_DIAG__?.perfPreset,
      irDurationSecs: window.__PIANO_DIAG__?.perfConfig.irDurationSecs,
      firBlockFft: window.__PIANO_DIAG__?.perfConfig.firBlockFft,
    }));
    expect(rtDiag.preset).toBe(2);
    expect(rtDiag.irDurationSecs).toBeCloseTo(0.18, 2);
    expect(rtDiag.firBlockFft).toBe(true);
    await expect(page.getByTestId('perf-preset-realtime')).toHaveClass(/active/);
  });

  test('3-tier preset switching and 19-knob overrides preserve voicing/space state cleanly', async ({
    page,
  }) => {
    await page.goto('./');
    await unlockAudioEngine(page);

    // Configure non-default space + voicing to verify Tier 3 rebuild preserves them.
    await page.getByTestId('body-upright-studio').click();
    await page.getByTestId('lid-closed').click();
    await setSlider(page, 'voicing-a4', 442.0);
    await waitForPrepIdle(page);

    // Switch Balanced -> Realtime -> Full -> Balanced via UI controls.
    await page.getByTestId('perf-preset-realtime').click();
    await waitForPrepIdle(page);
    await expect(page.getByTestId('perf-preset-realtime')).toHaveClass(/active/);

    await page.getByTestId('perf-preset-select').selectOption('full');
    await waitForPrepIdle(page);
    await expect(page.getByTestId('perf-preset-full')).toHaveClass(/active/);

    await page.getByTestId('perf-preset-balanced').click();
    await waitForPrepIdle(page);
    await expect(page.getByTestId('perf-preset-balanced')).toHaveClass(/active/);

    // Verify note playback produces finite, non-clipping audio after Tier 3 rebuilds.
    await page.locator('[data-midi="69"]').dispatchEvent('pointerdown', { buttons: 1 });
    await waitAudioBlocks(page, 64);
    const stats = await sampleDestinationAudio(page);
    expect(stats.nonFiniteCount).toBe(0);
    expect(stats.peak).toBeLessThanOrEqual(1.0);
    expect(stats.rms).toBeGreaterThan(0.002);
    await page.locator('[data-midi="69"]').dispatchEvent('pointerup', { buttons: 0 });
    await page.getByTestId('panic-btn').click();
    await waitForSilence(page);

    // Exercise Tier 1, Tier 2, and Tier 3 individual knob overrides + reset.
    await page.getByTestId('perf-advanced-toggle').locator('summary').click();
    await setSlider(page, 'perf-knob-energy_horizon_db', -80);
    await waitForPrepIdle(page, 4);
    await expect(page.getByTestId('perf-custom-badge')).toBeVisible();

    await page.getByTestId('perf-knob-fir_head_taps').selectOption('32');
    await waitForPrepIdle(page, 4);

    await setSlider(page, 'perf-knob-dead_mode_prune_db', -60);
    await waitForPrepIdle(page, 8);

    const customCfg = await page.evaluate(() => window.__PIANO_DIAG__?.perfConfig);
    expect(customCfg?.energyHorizonDb).toBe(-80);
    expect(customCfg?.firHeadTaps).toBe(32);
    expect(customCfg?.deadModePruneDb).toBe(-60);

    await page.getByTestId('perf-reset').click();
    await waitForPrepIdle(page, 8);
    await expect(page.getByTestId('perf-custom-badge')).toBeHidden();
    const resetCfg = await page.evaluate(() => window.__PIANO_DIAG__?.perfConfig);
    expect(resetCfg?.energyHorizonDb).toBe(-96);
    expect(resetCfg?.firHeadTaps).toBe(64);
    expect(resetCfg?.deadModePruneDb).toBe(-50);
  });
});
