import { expect, test } from '@playwright/test';
import {
  sampleDestinationAudio,
  setupErrorMonitoring,
  unlockAudioEngine,
  waitForAudioRms,
} from './helpers/audio-diag';
import { buildMidiFile } from './helpers/midi-helpers';
import { waitForOfflineRenderDone } from './helpers/offline-helpers';

test.describe('E2E Enriched Demo Scores Suite (Offline Worker)', () => {
  let assertNoErrors: () => void;

  test.beforeEach(async ({ page }) => {
    assertNoErrors = setupErrorMonitoring(page);
    await page.goto('./');
    await unlockAudioEngine(page);
  });

  test.afterEach(async ({ page }) => {
    await page.getByTestId('transport-stop').click().catch(() => {});
    await page.getByTestId('panic-btn').click();
    assertNoErrors();
  });

  test('All 6 classical demo scores are present in selector', async ({ page }) => {
    const demoSelect = page.getByTestId('demo-select');
    const options = await demoSelect.locator('option').all();
    expect(options.length).toBe(6);

    const values = await Promise.all(options.map((opt) => opt.getAttribute('value')));
    expect(values).toEqual([
      'bach-prelude',
      'chopin-nocturne',
      'beethoven-moonlight',
      'debussy-claire',
      'satie-gymnopedie',
      'rachmaninoff-prelude',
    ]);
  });

  const DEMO_CASES = [
    { id: 'bach-prelude', name: 'Bach Prelude' },
    { id: 'chopin-nocturne', name: 'Chopin Nocturne' },
    { id: 'beethoven-moonlight', name: 'Beethoven Moonlight' },
    { id: 'debussy-claire', name: 'Debussy Clair de Lune' },
    { id: 'satie-gymnopedie', name: 'Satie Gymnopedie' },
    { id: 'rachmaninoff-prelude', name: 'Rachmaninoff Prelude' },
  ];

  for (const demo of DEMO_CASES) {
    test(`Demo score ${demo.name} (${demo.id}) renders offline, plays, and stops cleanly`, async ({
      page,
    }) => {
      test.setTimeout(90_000);
      await page.getByTestId('demo-select').selectOption(demo.id);
      await page.getByTestId('transport-play').click();

      await waitForOfflineRenderDone(page);
      const playing = await waitForAudioRms(page, 0.003);
      await expect(page.getByTestId('hud-voices')).toHaveText('0');

      await page.getByTestId('transport-stop').click();
      await expect(page.getByTestId('offline-render-status')).toHaveText('Ready', {
        timeout: 3_000,
      });
      await expect(page.getByTestId('hud-voices')).toHaveText('0');
      await expect
        .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3_500 })
        .toBeLessThan(Math.max(5e-4, playing.rms * 0.1));
    });
  }

  test('Mid-playback score switching and mid-render cancel execute cleanly', async ({ page }) => {
    test.setTimeout(90_000);
    // Start Beethoven Moonlight and cancel mid-render
    await page.getByTestId('demo-select').selectOption('beethoven-moonlight');
    await page.getByTestId('transport-play').click();
    await expect(page.getByTestId('offline-cancel')).toBeEnabled({ timeout: 5_000 });
    await page.getByTestId('offline-cancel').click();
    await expect(page.getByTestId('offline-render-status')).toHaveText('Idle', { timeout: 5_000 });

    // Switch to Rachmaninoff Prelude, render, play, and verify clean stop
    await page.getByTestId('demo-select').selectOption('rachmaninoff-prelude');
    await expect(page.getByTestId('hud-voices')).toHaveText('0');
    await page.getByTestId('transport-play').click();
    await waitForOfflineRenderDone(page);
    await waitForAudioRms(page, 0.003);

    // Mid-playback switch back to Bach Prelude resets to Idle cleanly
    await page.getByTestId('demo-select').selectOption('bach-prelude');
    await expect(page.getByTestId('offline-render-status')).toHaveText('Idle', { timeout: 3_000 });
    await expect(page.getByTestId('hud-voices')).toHaveText('0');
  });

  test('Interactive feedback: keys animate visually during offline playback without waking AudioWorklet', async ({
    page,
  }) => {
    test.setTimeout(90_000);
    await page.getByTestId('demo-select').selectOption('rachmaninoff-prelude');
    await page.getByTestId('transport-play').click();
    await waitForOfflineRenderDone(page);

    await expect
      .poll(
        async () => page.locator('.piano-key.active, .piano-key.is-active').count(),
        { timeout: 3_000 },
      )
      .toBeGreaterThan(0);
    await expect(page.getByTestId('hud-voices')).toHaveText('0');

    await page.getByTestId('transport-stop').click();
    await expect
      .poll(async () => page.locator('.piano-key.active, .piano-key.is-active').count())
      .toBe(0);
  });

  test('Short custom .mid completes playback naturally (Playing -> Ready) and replays immediately from t=0', async ({
    page,
  }) => {
    test.setTimeout(30_000);
    const smf = buildMidiFile([
      { delta: 0, data: [0x90, 60, 108] },
      { delta: 384, data: [0x80, 60, 64] },
    ]);
    await page.getByTestId('midi-file-input').setInputFiles({
      name: 'short-end.mid',
      mimeType: 'audio/midi',
      buffer: Buffer.from(smf),
    });

    await page.getByTestId('transport-play').click();
    await waitForOfflineRenderDone(page);
    await waitForAudioRms(page, 0.003);

    await expect(page.getByTestId('offline-render-status')).toHaveText('Ready', {
      timeout: 15_000,
    });
    await expect(page.getByTestId('hud-voices')).toHaveText('0');
    expect(await page.locator('.piano-key.is-active').count()).toBe(0);
    expect(Number(await page.getByTestId('transport-seek').inputValue())).toBeCloseTo(0, 1);
    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3_000 })
      .toBeLessThan(0.0005);

    // Immediate replay from t = 0 using cached render.
    await page.getByTestId('transport-play').click();
    await expect(page.getByTestId('offline-render-status')).toHaveText('Playing');
    await waitForAudioRms(page, 0.003);
    await page.getByTestId('transport-stop').click();
  });
});
