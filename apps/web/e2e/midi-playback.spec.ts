import { expect, test, type Page } from '@playwright/test';
import {
  sampleDestinationAudio,
  setSlider,
  setupErrorMonitoring,
  unlockAudioEngine,
} from './helpers/audio-diag';

/** Encodes an unsigned integer as a MIDI Variable-Length Quantity (VLQ). */
function encodeVlq(val: number): number[] {
  let v = val & 0x7f;
  const bytes = [v];
  while ((val >>= 7) > 0) {
    bytes.unshift((val & 0x7f) | 0x80);
  }
  return bytes;
}

/** Builds an in-memory Format 0 SMF binary buffer from raw events. */
function buildMidiFile(events: Array<{ delta: number; data: number[] }>): Uint8Array {
  const trackBytes: number[] = [];
  // Set tempo 120 BPM: delta=0, 0xFF 0x51 0x03 0x07 0xA1 0x20
  trackBytes.push(0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20);
  for (const ev of events) {
    trackBytes.push(...encodeVlq(ev.delta), ...ev.data);
  }
  // End of track: delta=0, 0xFF 0x2F 0x00
  trackBytes.push(0x00, 0xff, 0x2f, 0x00);

  const header = [
    0x4d, 0x54, 0x68, 0x64, // 'MThd'
    0x00, 0x00, 0x00, 0x06, // length = 6
    0x00, 0x00,             // format = 0
    0x00, 0x01,             // 1 track
    0x01, 0xe0,             // 480 ticks/quarter
    0x4d, 0x54, 0x72, 0x6b, // 'MTrk'
    (trackBytes.length >> 24) & 0xff,
    (trackBytes.length >> 16) & 0xff,
    (trackBytes.length >> 8) & 0xff,
    trackBytes.length & 0xff,
    ...trackBytes,
  ];
  return new Uint8Array(header);
}

/** Injects a mock Web MIDI input device into navigator.requestMIDIAccess. */
async function injectMockMidi(page: Page): Promise<void> {
  await page.evaluate(() => {
    class MockMIDIInput extends EventTarget {
      readonly id = 'mock-piano-in';
      readonly name = 'Mock Piano MIDI Controller';
      readonly state = 'connected';
      readonly type = 'input';
      onmidimessage: ((e: unknown) => void) | null = null;
      send(data: number[]) {
        const ev = new Event('midimessage') as Event & { data: Uint8Array; receivedTime: number };
        ev.data = new Uint8Array(data);
        ev.receivedTime = performance.now();
        this.dispatchEvent(ev);
        if (this.onmidimessage) this.onmidimessage(ev);
      }
    }
    const mock = new MockMIDIInput();
    const map = new Map<string, unknown>([[mock.id, mock]]);
    (window as unknown as { __MOCK_MIDI__: MockMIDIInput }).__MOCK_MIDI__ = mock;
    (navigator as unknown as { requestMIDIAccess: () => Promise<unknown> }).requestMIDIAccess =
      async () => ({
        inputs: map,
        outputs: new Map(),
        sysexEnabled: false,
      });
  });
}

/** Dispatches raw MIDI bytes via the mock Web MIDI subsystem. */
async function dispatchMockMidi(page: Page, bytes: number[]): Promise<void> {
  await page.evaluate((b) => {
    const mock = (window as unknown as { __MOCK_MIDI__?: { send: (data: number[]) => void } })
      .__MOCK_MIDI__;
    mock?.send(b);
  }, bytes);
}

test.describe('E2E MIDI & SMF Playback Suite', () => {
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
  // Tier 1: Built-in Classical Demo Scores & Transport Controls
  // =========================================================================
  test('Tier 1: Bach Prelude demo playback triggers sound, voices, and decays on Stop', async ({
    page,
  }) => {
    await page.getByTestId('demo-select').selectOption('bach-prelude');
    await page.getByTestId('transport-play').click();

    await page.waitForTimeout(450);
    const playing = await sampleDestinationAudio(page);
    expect(playing.nonFiniteCount).toBe(0);
    expect(playing.peak).toBeLessThanOrEqual(1.0);
    expect(playing.rms).toBeGreaterThan(0.005);

    const voices = Number(await page.getByTestId('hud-voices').textContent());
    expect(voices).toBeGreaterThan(0);

    // Stop playback and verify voices decay to 0 and RMS silences.
    await page.getByTestId('transport-stop').click();
    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3000 })
      .toBeLessThan(1e-4);
    await expect(page.getByTestId('hud-voices')).toHaveText('0');
  });

  test('Tier 1: Chopin Nocturne demo plays and pauses/resumes cleanly', async ({ page }) => {
    await page.getByTestId('demo-select').selectOption('chopin-nocturne');
    await page.getByTestId('transport-play').click();
    await page.waitForTimeout(450);

    const initial = await sampleDestinationAudio(page);
    expect(initial.rms).toBeGreaterThan(0.005);

    // Pause playback: sound should decay.
    await page.getByTestId('transport-pause').click();
    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3000 })
      .toBeLessThan(initial.rms * 0.25);

    // Resume playback: sound should return.
    await page.getByTestId('transport-play').click();
    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3000 })
      .toBeGreaterThan(0.005);

    await page.getByTestId('transport-stop').click();
  });

  test('Tier 1: Panic clears active demo playback instantly', async ({ page }) => {
    await page.getByTestId('demo-select').selectOption('bach-prelude');
    await page.getByTestId('transport-play').click();
    await page.waitForTimeout(400);

    await page.getByTestId('panic-btn').click();
    await expect(page.getByTestId('hud-voices')).toHaveText('0');
    const stats = await sampleDestinationAudio(page);
    expect(stats.nonFiniteCount).toBe(0);
  });

  // =========================================================================
  // Tier 2: Boundary & Corner Cases (SMF Upload & Robustness)
  // =========================================================================
  test('Tier 2: Upload valid SMF binary via file input and verify playback', async ({ page }) => {
    // NoteOn C4 vel 100, delta 480, NoteOff C4
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
    await page.waitForTimeout(300);
    const stats = await sampleDestinationAudio(page);
    expect(stats.rms).toBeGreaterThan(0.003);
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

    // Verify application remains responsive: normal A4 button works.
    await page.getByTestId('trigger-a4').click();
    await page.waitForTimeout(400);
    const audio = await sampleDestinationAudio(page);
    expect(audio.rms).toBeGreaterThan(0.001);
  });

  test('Tier 2: Running status across delta-times triggers multiple notes', async ({ page }) => {
    // NoteOn 60 (C4), then running status 64 (E4), 67 (G4)
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

    await expect
      .poll(async () => Number(await page.getByTestId('hud-voices').textContent()), { timeout: 2000 })
      .toBeGreaterThanOrEqual(3);

    await page.getByTestId('transport-stop').click();
  });

  test('Tier 2: Zero-velocity NoteOn is treated strictly as NoteOff', async ({ page }) => {
    // NoteOn 60 vel 100, then NoteOn 60 vel 0 (MUST act as NoteOff)
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
    await page.waitForTimeout(200);

    // After NoteOn vel 0, voice must decay cleanly.
    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3000 })
      .toBeLessThan(0.001);
    await page.getByTestId('transport-stop').click();
  });

  // =========================================================================
  // Tier 3: Cross-Feature Combinations (Transport Scrubbing & Tempo Scaling)
  // =========================================================================
  test('Tier 3: Transport seek during playback jumps smoothly and flushes obsolete notes', async ({
    page,
  }) => {
    await page.getByTestId('demo-select').selectOption('bach-prelude');
    await page.getByTestId('transport-play').click();
    await page.waitForTimeout(350);

    // Seek to 15 seconds into the score.
    await setSlider(page, 'transport-seek', 15);
    await page.waitForTimeout(350);

    const stats = await sampleDestinationAudio(page);
    expect(stats.nonFiniteCount).toBe(0);
    expect(stats.peak).toBeLessThanOrEqual(1.0);
    expect(stats.rms).toBeGreaterThan(0.003);

    await page.getByTestId('transport-stop').click();
  });

  test('Tier 3: Tempo slider adjustments (0.5x to 2.0x) play smoothly without audio glitches', async ({
    page,
  }) => {
    await page.getByTestId('demo-select').selectOption('bach-prelude');
    await page.getByTestId('transport-play').click();

    // Scale tempo to 0.5x (slow)
    await setSlider(page, 'tempo-slider', 0.5);
    await page.waitForTimeout(300);
    let stats = await sampleDestinationAudio(page);
    expect(stats.nonFiniteCount).toBe(0);
    expect(stats.rms).toBeGreaterThan(0.003);

    // Scale tempo to 2.0x (fast)
    await setSlider(page, 'tempo-slider', 2.0);
    await page.waitForTimeout(300);
    stats = await sampleDestinationAudio(page);
    expect(stats.nonFiniteCount).toBe(0);
    expect(stats.rms).toBeGreaterThan(0.003);

    await page.getByTestId('transport-stop').click();
  });

  // =========================================================================
  // Tier 4: Real-World Workloads (Web MIDI Simulation & Full Demo Section)
  // =========================================================================
  test('Tier 4: Live Web MIDI stream with 14-bit high-res velocity and CC64 damper', async ({
    page,
  }) => {
    // 1. Dispatch 14-bit high-res velocity NoteOn: CC 88 prefix (LSB 64) followed by NoteOn (MSB 100)
    await dispatchMockMidi(page, [0xb0, 88, 64]);
    await dispatchMockMidi(page, [0x90, 60, 100]); // C4
    await page.waitForTimeout(200);

    let stats = await sampleDestinationAudio(page);
    expect(stats.rms).toBeGreaterThan(0.002);

    // 2. Continuous Damper Pedal CC64 modulation (half-pedal -> full pedal)
    await dispatchMockMidi(page, [0xb0, 64, 64]); // Half pedal
    await dispatchMockMidi(page, [0xb0, 64, 127]); // Full sustain
    await dispatchMockMidi(page, [0x80, 60, 64]); // Release key

    await page.waitForTimeout(300);
    // Because pedal is sustained, sound should still ring.
    stats = await sampleDestinationAudio(page);
    expect(stats.rms).toBeGreaterThan(0.001);

    // 3. Release pedal: sound should decay cleanly.
    await dispatchMockMidi(page, [0xb0, 64, 0]);
    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3000 })
      .toBeLessThan(0.0005);
  });

  test('Tier 4: Full Bach Prelude playback section with D1 telemetry and zero errors', async ({
    page,
  }) => {
    await page.getByTestId('demo-select').selectOption('bach-prelude');
    await page.getByTestId('transport-play').click();

    // Monitor for 6 continuous seconds of active classical performance.
    for (let sec = 0; sec < 12; sec++) {
      await page.waitForTimeout(500);
      const audio = await sampleDestinationAudio(page);
      expect(audio.nonFiniteCount).toBe(0);
      expect(audio.peak).toBeLessThanOrEqual(1.0);
      expect(audio.rms).toBeGreaterThan(0.003);
    }

    // Verify telemetry HUD p99 callback is finite and non-negative
    const hudP99 = await page.evaluate(() => {
      const text = document.querySelector('[data-testid="hud-p99"]')?.textContent;
      return text ? parseFloat(text) : NaN;
    });
    expect(Number.isFinite(hudP99) && hudP99 >= 0).toBe(true);

    await page.getByTestId('transport-stop').click();
  });
});
