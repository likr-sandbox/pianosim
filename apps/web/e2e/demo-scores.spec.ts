import { expect, test } from '@playwright/test';
import {
  sampleDestinationAudio,
  setupErrorMonitoring,
  unlockAudioEngine,
  waitForAudioRms,
} from './helpers/audio-diag';

test.describe('E2E Enriched Demo Scores Suite', () => {
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
    test(`Demo score ${demo.name} (${demo.id}) plays audio and stops cleanly`, async ({ page }) => {
      await page.getByTestId('demo-select').selectOption(demo.id);
      await page.getByTestId('transport-play').click();

      const playing = await waitForAudioRms(page, 0.003);

      await expect
        .poll(async () => Number(await page.getByTestId('hud-voices').textContent()), {
          timeout: 2000,
        })
        .toBeGreaterThan(0);

      // Verify stop clears voices and audio decays to silence
      await page.getByTestId('transport-stop').click();
      await expect(page.getByTestId('hud-voices')).toHaveText('0', { timeout: 3500 });
      await expect
        .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3500 })
        .toBeLessThan(Math.max(5e-4, playing.rms * 0.1));
    });
  }

  test('Mid-playback score switching executes without errors or hung voices', async ({ page }) => {
    // Start Beethoven Moonlight
    await page.getByTestId('demo-select').selectOption('beethoven-moonlight');
    await page.getByTestId('transport-play').click();
    await waitForAudioRms(page, 0.003);

    // Switch mid-playback to Rachmaninoff Prelude
    await page.getByTestId('demo-select').selectOption('rachmaninoff-prelude');
    // Selecting loads the new score into MidiPlayer and stops previous playback
    await expect(page.getByTestId('hud-voices')).toHaveText('0', { timeout: 3500 });

    // Play the new score
    await page.getByTestId('transport-play').click();
    await waitForAudioRms(page, 0.003);

    // Clean stop
    await page.getByTestId('transport-stop').click();
    await expect(page.getByTestId('hud-voices')).toHaveText('0', { timeout: 3500 });
  });

  test('Interactive feedback: keys animate on keyboard during demo playback', async ({ page }) => {
    await page.getByTestId('demo-select').selectOption('rachmaninoff-prelude');
    await page.getByTestId('transport-play').click();

    // In Rachmaninoff Prelude, notes are struck immediately on beat 0
    await expect
      .poll(
        async () => {
          const activeKeys = await page.locator('.piano-key.active, .piano-key.is-active').count();
          return activeKeys;
        },
        { timeout: 2000 },
      )
      .toBeGreaterThan(0);

    // When stopped, all keys must deactivate
    await page.getByTestId('transport-stop').click();
    await expect
      .poll(async () => {
        return await page.locator('.piano-key.active, .piano-key.is-active').count();
      })
      .toBe(0);
  });
});
