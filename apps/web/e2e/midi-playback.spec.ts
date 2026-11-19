import { expect, test } from '@playwright/test';
import {
  sampleDestinationAudio,
  setSlider,
  setupErrorMonitoring,
  unlockAudioEngine,
  waitAudioBlocks,
  waitForAudioRms,
} from './helpers/audio-diag';
import { buildMidiFile, dispatchMockMidi, injectMockMidi } from './helpers/midi-helpers';
import { downloadAndVerifyOfflineWav, waitForOfflineRenderDone } from './helpers/offline-helpers';

test.describe('E2E MIDI & SMF Playback Suite (Offline Render + Live Web MIDI)', () => {
  let assertNoErrors: () => void;

  test.beforeEach(async ({ page }) => {
    assertNoErrors = setupErrorMonitoring(page);
    await page.goto('./');
    await injectMockMidi(page);
    await unlockAudioEngine(page);
  });

  test.afterEach(async ({ page }) => {
    await page.getByTestId('panic-btn').click();
    assertNoErrors();
  });

  // =========================================================================
  // Tier 1: Built-in Classical Demo Scores & Offline Transport Controls
  // =========================================================================
  test('Tier 1: Bach Prelude demo offline render triggers sound, keeps Worklet idle, and decays on Stop', async ({
    page,
  }) => {
    test.setTimeout(90_000);
    await page.getByTestId('demo-select').selectOption('bach-prelude');
    await page.getByTestId('transport-play').click();

    await waitForOfflineRenderDone(page);
    await waitForAudioRms(page, 0.005);
    await expect(page.getByTestId('hud-voices')).toHaveText('0');

    await page.getByTestId('transport-stop').click();
    await expect(page.getByTestId('offline-render-status')).toHaveText('Ready');
    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3000, intervals: [25] })
      .toBeLessThan(1e-4);
    await expect(page.getByTestId('hud-voices')).toHaveText('0');
  });

  test('Tier 1: Chopin Nocturne demo plays and pauses/resumes cleanly', async ({ page }) => {
    test.setTimeout(90_000);
    await page.getByTestId('demo-select').selectOption('chopin-nocturne');
    await page.getByTestId('transport-play').click();
    await waitForOfflineRenderDone(page);
    const initial = await waitForAudioRms(page, 0.005);

    await page.getByTestId('transport-pause').click();
    await expect(page.getByTestId('offline-render-status')).toHaveText('Paused');
    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3000, intervals: [25] })
      .toBeLessThan(initial.rms * 0.25);

    // Cached instant resume without re-rendering
    await page.getByTestId('transport-play').click();
    await expect(page.getByTestId('offline-render-status')).toHaveText('Playing');
    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3000, intervals: [25] })
      .toBeGreaterThan(0.005);

    await page.getByTestId('transport-stop').click();
  });

  test('Tier 1: Panic clears active offline playback instantly', async ({ page }) => {
    const smf = buildMidiFile([
      { delta: 0, data: [0xb0, 64, 127] },
      { delta: 0, data: [0x90, 60, 108] },
      { delta: 120, data: [0x90, 64, 108] },
      { delta: 120, data: [0x90, 67, 108] },
      { delta: 3840, data: [0x80, 60, 64] },
      { delta: 0, data: [0x80, 64, 64] },
      { delta: 0, data: [0x80, 67, 64] },
    ]);
    await page.getByTestId('midi-file-input').setInputFiles({
      name: 'panic-sustain.mid',
      mimeType: 'audio/midi',
      buffer: Buffer.from(smf),
    });
    await page.getByTestId('transport-play').click();
    await waitForOfflineRenderDone(page);
    await waitForAudioRms(page, 0.003);

    await page.getByTestId('panic-btn').click();
    await expect(page.getByTestId('hud-voices')).toHaveText('0');
    await expect(page.getByTestId('offline-render-status')).toHaveText('Ready');
    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3000, intervals: [25] })
      .toBeLessThan(1e-4);
    const stats = await sampleDestinationAudio(page);
    expect(stats.nonFiniteCount).toBe(0);
  });

  // =========================================================================
  // Tier 2: Boundary & Corner Cases (SMF Upload & Robustness)
  // =========================================================================
  test('Tier 2: Upload valid SMF binary via file input and verify offline playback', async ({
    page,
  }) => {
    const smf = buildMidiFile([
      { delta: 0, data: [0x90, 60, 100] },
      { delta: 480, data: [0x80, 60, 64] },
    ]);
    await page.getByTestId('midi-file-input').setInputFiles({
      name: 'valid-test.mid',
      mimeType: 'audio/midi',
      buffer: Buffer.from(smf),
    });

    await page.getByTestId('transport-play').click();
    await waitForOfflineRenderDone(page);
    await waitForAudioRms(page, 0.003);
    await page.getByTestId('transport-stop').click();
  });

  test('Tier 2: Corrupt SMF header shows error banner without crashing', async ({ page }) => {
    const corruptBytes = new Uint8Array([0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00]); // 'RIFF...'
    await page.getByTestId('midi-file-input').setInputFiles({
      name: 'corrupt.mid',
      mimeType: 'audio/midi',
      buffer: Buffer.from(corruptBytes),
    });

    const banner = page.getByTestId('error-banner');
    await expect(banner).toBeVisible();
    await expect(banner).toContainText(/invalid|error/i);

    await page.getByTestId('trigger-a4').click();
    await waitForAudioRms(page, 0.001);
  });

  test('Tier 2: Running status across delta-times renders multi-note audio and exports valid WAV', async ({
    page,
  }) => {
    const smf = buildMidiFile([
      { delta: 0, data: [0x90, 60, 100] },
      { delta: 50, data: [64, 100] },
      { delta: 50, data: [67, 100] },
      { delta: 480, data: [60, 0] },
      { delta: 0, data: [64, 0] },
      { delta: 0, data: [67, 0] },
    ]);
    await page.getByTestId('midi-file-input').setInputFiles({
      name: 'running-status.mid',
      mimeType: 'audio/midi',
      buffer: Buffer.from(smf),
    });
    await page.getByTestId('transport-play').click();
    await waitForOfflineRenderDone(page);
    await waitForAudioRms(page, 0.002);

    const wav = await downloadAndVerifyOfflineWav(page, 'f32');
    expect(wav.nonFiniteCount).toBe(0);
    expect(wav.peak).toBeGreaterThan(0.005);
    expect(wav.peak).toBeLessThanOrEqual(1.0);
    await page.getByTestId('transport-stop').click();
  });

  test('Tier 2: Zero-velocity NoteOn is treated strictly as NoteOff', async ({ page }) => {
    // NoteOn 60 vel 100, then NoteOn 60 vel 0 after ~104 ms (MUST act as NoteOff)
    const smf = buildMidiFile([
      { delta: 0, data: [0x90, 60, 100] },
      { delta: 100, data: [0x90, 60, 0] },
    ]);
    await page.getByTestId('midi-file-input').setInputFiles({
      name: 'zero-vel.mid',
      mimeType: 'audio/midi',
      buffer: Buffer.from(smf),
    });
    await page.getByTestId('transport-play').click();
    await waitForOfflineRenderDone(page);

    const wav = await downloadAndVerifyOfflineWav(page, 'f32');
    expect(wav.nonFiniteCount).toBe(0);
    expect(wav.mono).not.toBeNull();
    const mono = wav.mono!;
    const rmsSlice = (startSec: number, lenSec: number) => {
      const s0 = Math.floor(startSec * wav.sampleRate);
      const n = Math.floor(lenSec * wav.sampleRate);
      let sq = 0;
      for (let i = s0; i < s0 + n && i < mono.length; i++) sq += mono[i] * mono[i];
      return Math.sqrt(sq / Math.max(1, n));
    };
    const onsetRms = rmsSlice(0.01, 0.08);
    const dampedRms = rmsSlice(0.55, 0.15);
    expect(onsetRms).toBeGreaterThan(0.01);
    expect(dampedRms).toBeLessThan(onsetRms * 0.15);

    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3000, intervals: [25] })
      .toBeLessThan(0.001);
    await page.getByTestId('transport-stop').click();
  });

  // =========================================================================
  // Tier 3: Cross-Feature Combinations (Transport Scrubbing & Tempo Scaling)
  // =========================================================================
  test('Tier 3: Transport seek during playback jumps smoothly', async ({ page }) => {
    test.setTimeout(90_000);
    await page.getByTestId('demo-select').selectOption('bach-prelude');
    await page.getByTestId('transport-play').click();
    await waitForOfflineRenderDone(page);
    await waitForAudioRms(page, 0.003);

    await setSlider(page, 'transport-seek', 15);
    await waitForOfflineRenderDone(page, 30_000);
    await waitAudioBlocks(page, 24);
    await waitForAudioRms(page, 0.003);

    await page.getByTestId('transport-stop').click();
  });

  test('Tier 3: Tempo slider adjustments (0.5x to 2.0x) play smoothly without audio glitches', async ({
    page,
  }) => {
    const smf = buildMidiFile([
      { delta: 0, data: [0xb0, 64, 127] },
      { delta: 0, data: [0x90, 48, 108] },
      { delta: 240, data: [0x90, 60, 104] },
      { delta: 240, data: [0x90, 64, 104] },
      { delta: 240, data: [0x90, 67, 104] },
      { delta: 4800, data: [0x80, 48, 64] },
      { delta: 0, data: [0x80, 60, 64] },
      { delta: 0, data: [0x80, 64, 64] },
      { delta: 0, data: [0x80, 67, 64] },
    ]);
    await page.getByTestId('midi-file-input').setInputFiles({
      name: 'tempo-test.mid',
      mimeType: 'audio/midi',
      buffer: Buffer.from(smf),
    });
    await page.getByTestId('transport-play').click();
    await waitForOfflineRenderDone(page);

    await setSlider(page, 'tempo-slider', 0.5);
    await waitAudioBlocks(page, 48);
    await waitForAudioRms(page, 0.003);

    await setSlider(page, 'tempo-slider', 2.0);
    await waitAudioBlocks(page, 48);
    await waitForAudioRms(page, 0.003);

    await page.getByTestId('transport-stop').click();
  });

  // =========================================================================
  // Tier 4: Real-World Workloads (Web MIDI RT & Full Offline WAV Export)
  // =========================================================================
  test('Tier 4: Live Web MIDI stream with 14-bit high-res velocity and CC64 damper', async ({
    page,
  }) => {
    await dispatchMockMidi(page, [0xb0, 88, 64]);
    await dispatchMockMidi(page, [0x90, 60, 100]); // C4
    await waitAudioBlocks(page, 80);

    let stats = await sampleDestinationAudio(page);
    expect(stats.rms).toBeGreaterThan(0.002);

    await dispatchMockMidi(page, [0xb0, 64, 64]); // Half pedal
    await dispatchMockMidi(page, [0xb0, 64, 127]); // Full sustain
    await dispatchMockMidi(page, [0x80, 60, 64]); // Release key

    await waitAudioBlocks(page, 96);
    stats = await sampleDestinationAudio(page);
    expect(stats.rms).toBeGreaterThan(0.001);

    await dispatchMockMidi(page, [0xb0, 64, 0]);
    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3000, intervals: [25] })
      .toBeLessThan(0.0005);
  });

  test('Tier 4: Full Bach Prelude offline render playback and 24-bit/32-bit Full WAV export', async ({
    page,
  }) => {
    test.setTimeout(90_000);
    await page.getByTestId('demo-select').selectOption('bach-prelude');
    await page.getByTestId('transport-play').click();
    await waitForOfflineRenderDone(page);
    await expect(page.getByTestId('offline-export-wav')).toBeDisabled();
    await waitForAudioRms(page, 0.003);

    for (let step = 0; step < 6; step++) {
      await waitAudioBlocks(page, 128);
      const audio = await sampleDestinationAudio(page);
      expect(audio.nonFiniteCount).toBe(0);
      expect(audio.peak).toBeLessThanOrEqual(1.0);
      expect(audio.rms).toBeGreaterThan(0.003);
    }
    await expect(page.getByTestId('hud-voices')).toHaveText('0');

    const wav24 = await downloadAndVerifyOfflineWav(page, 'pcm24');
    expect(wav24.frames).toBeGreaterThan(48_000 * 30);
    expect(wav24.peak).toBeGreaterThan(0.005);
    expect(wav24.peak).toBeLessThanOrEqual(1.0);

    const wav32 = await downloadAndVerifyOfflineWav(page, 'f32');
    expect(wav32.nonFiniteCount).toBe(0);
    expect(wav32.peak).toBeGreaterThan(0.005);
    expect(wav32.peak).toBeLessThanOrEqual(1.0);

    await page.getByTestId('transport-stop').click();
  });

  test('Tier 3: Mid-render invalidateIfStale resets playback cleanly to Idle without buffering hang', async ({
    page,
  }) => {
    test.setTimeout(90_000);
    await page.getByTestId('demo-select').selectOption('bach-prelude');
    await page.getByTestId('transport-play').click();

    await expect
      .poll(
        async () => ((await page.getByTestId('offline-render-status').textContent()) ?? '').trim(),
        { timeout: 30_000, intervals: [50] },
      )
      .toBe('Playing');
    await expect(page.getByTestId('offline-cancel')).toBeEnabled();

    await setSlider(page, 'voicing-hammer', 0.85);
    await expect(page.getByTestId('offline-render-status')).toHaveText('Idle');
    await expect(page.getByTestId('transport-play')).toBeEnabled();
    await expect(page.getByTestId('offline-cancel')).toBeDisabled();
    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3000, intervals: [25] })
      .toBeLessThan(1e-4);

    // Confirm clean post-invalidation recovery on subsequent Play.
    await page.getByTestId('transport-play').click();
    await waitForOfflineRenderDone(page);
    await waitForAudioRms(page, 0.002);
    await page.getByTestId('transport-stop').click();
  });
});
