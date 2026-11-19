import { expect, test } from '@playwright/test';
import {
  expectAudioDecay,
  sampleDestinationAudio,
  setSlider,
  setupErrorMonitoring,
  unlockAudioEngine,
  waitAudioBlocks,
  waitForAudioRms,
  waitForSilence,
} from './helpers/audio-diag';
import {
  buildMidiFile,
  dispatchMockMidi,
  injectMockMidi,
  tapWorkletMessages,
} from './helpers/midi-helpers';
import { downloadAndVerifyOfflineWav, waitForOfflineRenderDone } from './helpers/offline-helpers';

test.describe('Wave 1B: Offline Seek Pedals, WebMIDI & Touch Curve', () => {
  test('F12/I-3: OfflineRenderer isolates AudioWorklet, restores visual CC64/66/67 on seek(), and exports Full WAV', async ({
    page,
  }) => {
    const assertNoErrors = setupErrorMonitoring(page);
    await page.goto('./');
    await unlockAudioEngine(page);
    await tapWorkletMessages(page);

    const smf = buildMidiFile([
      { delta: 0, data: [0xb0, 64, 127] },
      { delta: 0, data: [0xb0, 66, 127] },
      { delta: 0, data: [0xb0, 67, 64] },
      { delta: 38, data: [0x90, 60, 100] },
      { delta: 442, data: [0x80, 60, 64] },
      { delta: 480, data: [0x90, 67, 100] },
      { delta: 480, data: [0x80, 67, 64] },
    ]);
    await page.getByTestId('midi-file-input').setInputFiles({
      name: 'lookahead.mid',
      mimeType: 'audio/midi',
      buffer: Buffer.from(smf),
    });
    await page.getByTestId('transport-play').click();
    await waitForOfflineRenderDone(page);
    await waitForAudioRms(page, 0.002);

    const workletMidiCount = await page.evaluate(() =>
      (window as unknown as { __MSG_LOG__: Array<{ type: string }> }).__MSG_LOG__.filter(
        (m) => m.type === 'noteOn' || m.type === 'noteOff' || m.type === 'pedal',
      ).length,
    );
    expect(workletMidiCount).toBe(0);

    const wav = await downloadAndVerifyOfflineWav(page, 'f32');
    expect(wav.nonFiniteCount).toBe(0);
    expect(wav.peak).toBeGreaterThan(0.005);
    expect(wav.peak).toBeLessThanOrEqual(1.0);

    await page.getByTestId('transport-pause').click();
    await setSlider(page, 'transport-seek', 0.8);
    await expect(page.getByTestId('sustain-toggle')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('pedal-sostenuto')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('pedal-una-corda')).toHaveAttribute('aria-pressed', 'true');
    expect(Number(await page.getByTestId('pedal-una-corda').getAttribute('data-value'))).toBeCloseTo(
      64 / 127,
      2,
    );

    await setSlider(page, 'transport-seek', 0.0);
    await expect(page.getByTestId('sustain-toggle')).toHaveAttribute('aria-pressed', 'false');
    await expect(page.getByTestId('pedal-una-corda')).toHaveAttribute('aria-pressed', 'false');
    await page.getByTestId('panic-btn').click();
    assertNoErrors();
  });

  test('I-3/I-4: WebMidiManager bidirectional UI, release velocity, CC67 & 3ms jitter window', async ({
    page,
  }) => {
    const assertNoErrors = setupErrorMonitoring(page);
    await page.goto('./');
    await injectMockMidi(page);
    await unlockAudioEngine(page);
    await expect(page.getByTestId('midi-device-select')).toContainText(
      'Mock Piano MIDI Controller',
    );
    await tapWorkletMessages(page);

    await dispatchMockMidi(page, [0x90, 60, 100]);
    await expect(page.locator('[data-midi="60"]')).toHaveClass(/is-active/);
    await waitAudioBlocks(page, 48);
    expect((await sampleDestinationAudio(page)).rms).toBeGreaterThan(0.002);
    await dispatchMockMidi(page, [0x80, 60, 90]);
    await expect(page.locator('[data-midi="60"]')).not.toHaveClass(/is-active/);

    await dispatchMockMidi(page, [0x90, 62, 95]);
    await expect(page.locator('[data-midi="62"]')).toHaveClass(/is-active/);
    await dispatchMockMidi(page, [0x90, 62, 0]);
    await expect(page.locator('[data-midi="62"]')).not.toHaveClass(/is-active/);

    await dispatchMockMidi(page, [0xb0, 67, 64]);
    await expect(page.getByTestId('pedal-una-corda')).toHaveAttribute('aria-pressed', 'true');
    expect(Number(await page.getByTestId('pedal-una-corda').getAttribute('data-value'))).toBeCloseTo(
      64 / 127,
      2,
    );
    await dispatchMockMidi(page, [0xb0, 67, 0]);
    await expect(page.getByTestId('pedal-una-corda')).toHaveAttribute('aria-pressed', 'false');

    const log = await page.evaluate(
      () =>
        (
          window as unknown as {
            __MSG_LOG__: Array<{
              type: string;
              releaseVelU16?: number;
              sampleOffset?: number;
            }>;
          }
        ).__MSG_LOG__,
    );
    const noteOns = log.filter((m) => m.type === 'noteOn');
    const noteOffs = log.filter((m) => m.type === 'noteOff');
    expect(noteOns.some((m) => (m.sampleOffset ?? 0) > 0 && (m.sampleOffset ?? 0) <= 200)).toBe(
      true,
    );
    expect(noteOffs[0]?.releaseVelU16).toBe(Math.trunc((90 * 65535) / 127));
    expect(noteOffs[1]?.releaseVelU16).toBe(32768);
    await page.getByTestId('panic-btn').click();
    assertNoErrors();
  });

  test('I-4: Incoming Web MIDI event auto-unlocks WebAudio without clicking Unlock Audio', async ({
    page,
  }) => {
    const assertNoErrors = setupErrorMonitoring(page);
    await page.addInitScript(() => {
      class InitMockMidi extends EventTarget {
        readonly id = 'auto-unlock-in';
        readonly name = 'Auto Unlock Controller';
        send(data: number[]) {
          this.dispatchEvent(
            Object.assign(new Event('midimessage'), {
              data: new Uint8Array(data),
              receivedTime: performance.now(),
            }),
          );
        }
      }
      const mock = new InitMockMidi();
      (window as unknown as { __MOCK_MIDI__: InitMockMidi }).__MOCK_MIDI__ = mock;
      (navigator as unknown as { requestMIDIAccess: () => Promise<unknown> }).requestMIDIAccess =
        async () => ({
          inputs: new Map([[mock.id, mock]]),
          outputs: new Map(),
        });
    });

    await page.goto('./');
    await dispatchMockMidi(page, [0x90, 69, 105]);
    await expect
      .poll(
        async () =>
          page.evaluate(() => `${window.__PIANO_DIAG__?.status}:${window.__PIANO_DIAG__?.contextState}`),
        { timeout: 15_000 },
      )
      .toBe('ready:running');
    await waitForAudioRms(page, 0.002);
    await dispatchMockMidi(page, [0x80, 69, 64]);
    await page.getByTestId('panic-btn').click();
    assertNoErrors();
  });

  test('I-2: Touch Curve (Light / Natural / Heavy) alters strike velocity RMS ordering', async ({
    page,
  }) => {
    const assertNoErrors = setupErrorMonitoring(page);
    await page.goto('./');
    await unlockAudioEngine(page);
    await setSlider(page, 'velocity-slider', 64);

    const strikeA4 = async (mode: 'light' | 'natural' | 'heavy'): Promise<number> => {
      await page.getByTestId(`touch-curve-${mode}`).click();
      await expect(page.getByTestId(`touch-curve-${mode}`)).toHaveAttribute('aria-checked', 'true');
      const a4 = page.locator('[data-midi="69"]');
      await a4.dispatchEvent('pointerdown', { buttons: 1 });
      await waitAudioBlocks(page, 96);
      const stats = await sampleDestinationAudio(page);
      await a4.dispatchEvent('pointerup', { buttons: 0 });
      await expectAudioDecay(page, stats.rms, 0.08, 3000);
      await page.getByTestId('panic-btn').click();
      await waitForSilence(page);
      return stats.rms;
    };

    const [lightRms, naturalRms, heavyRms] = [
      await strikeA4('light'),
      await strikeA4('natural'),
      await strikeA4('heavy'),
    ];
    expect(lightRms).toBeGreaterThan(naturalRms);
    expect(naturalRms).toBeGreaterThan(heavyRms);
    await page.getByTestId('touch-curve-natural').click();
    assertNoErrors();
  });
});
