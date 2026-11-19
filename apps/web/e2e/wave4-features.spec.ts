import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { expect, test } from '@playwright/test';
import { DEMO_SCORES } from '../src/midi/demo-scores';
import { parseSmf, type MidiCcEvent, type MidiNoteEvent } from '../src/midi/smf-parser';
import {
  expectAudioDecay,
  sampleDestinationAudio,
  setSlider,
  setupErrorMonitoring,
  unlockAudioEngine,
  waitAudioBlocks,
  waitForAudioRms,
  waitForPrepIdle,
  waitForSilence,
} from './helpers/audio-diag';
import { waitForOfflineRenderDone } from './helpers/offline-helpers';

test.describe('E2E Wave 4 IV-1 & IV-2 Features Suite', () => {
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

  test('IV-1: Widened concert pitch range (392.0 Hz French Baroque to 466.2 Hz Chorton)', async ({
    page,
  }) => {
    const a4Key = page.locator('[data-midi="69"]');

    await page.getByTestId('voicing-a4-chip-392').click();
    await expect(page.getByTestId('trigger-a4')).toHaveText('A4 = 392.0 Hz');
    await waitForPrepIdle(page);

    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await waitAudioBlocks(page, 96);
    const low392 = await sampleDestinationAudio(page, 360, 440);
    await a4Key.dispatchEvent('pointerup', { buttons: 0 });
    expect(low392.nonFiniteCount).toBe(0);
    expect(low392.rms).toBeGreaterThan(0.002);
    expect(low392.peakFreqHz).toBeGreaterThanOrEqual(386);
    expect(low392.peakFreqHz).toBeLessThanOrEqual(398);

    await page.getByTestId('panic-btn').click();
    await waitForSilence(page);

    await setSlider(page, 'voicing-a4', 466.2);
    await expect(page.getByTestId('trigger-a4')).toHaveText('A4 = 466.2 Hz');
    await waitForPrepIdle(page);

    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await waitAudioBlocks(page, 96);
    const high4662 = await sampleDestinationAudio(page, 430, 500);
    await a4Key.dispatchEvent('pointerup', { buttons: 0 });
    expect(high4662.nonFiniteCount).toBe(0);
    expect(high4662.rms).toBeGreaterThan(0.002);
    expect(high4662.peakFreqHz).toBeGreaterThanOrEqual(460);
    expect(high4662.peakFreqHz).toBeLessThanOrEqual(472);
    expect(high4662.peakFreqHz - low392.peakFreqHz).toBeGreaterThan(68);
  });

  test('IV-1: Classical temperaments (Equal, Werckmeister III, Kirnberger III, Vallotti) retune C5/E4 while keeping A4 anchored', async ({
    page,
  }) => {
    const tempSelect = page.getByTestId('voicing-temperament');
    const c5Key = page.locator('[data-midi="72"]');
    const e4Key = page.locator('[data-midi="64"]');
    const a4Key = page.locator('[data-midi="69"]');

    // Measure Equal (0) C5 (~523.25 Hz) and E4 (~329.63 Hz)
    await tempSelect.selectOption('0');
    await waitForPrepIdle(page);
    await c5Key.dispatchEvent('pointerdown', { buttons: 1 });
    await waitAudioBlocks(page, 96);
    const eqC5 = await sampleDestinationAudio(page, 480, 560);
    await c5Key.dispatchEvent('pointerup', { buttons: 0 });
    await expectAudioDecay(page, eqC5.rms, 0.15, 3000);

    await e4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await waitAudioBlocks(page, 96);
    const eqE4 = await sampleDestinationAudio(page, 300, 360);
    await e4Key.dispatchEvent('pointerup', { buttons: 0 });
    await expectAudioDecay(page, eqE4.rms, 0.15, 3000);

    // Measure Kirnberger III (2): A-anchored C5 is +10.26 cents (+3.1 Hz) and E4 is -3.42 cents (-0.65 Hz)
    await tempSelect.selectOption('2');
    await expect(page.getByTestId('preset-custom-badge')).toBeVisible();
    await waitForPrepIdle(page);

    await c5Key.dispatchEvent('pointerdown', { buttons: 1 });
    await waitAudioBlocks(page, 96);
    const kirnC5 = await sampleDestinationAudio(page, 480, 560);
    await c5Key.dispatchEvent('pointerup', { buttons: 0 });
    expect(kirnC5.nonFiniteCount).toBe(0);
    expect(kirnC5.rms).toBeGreaterThan(0.002);
    expect(kirnC5.peakFreqHz - eqC5.peakFreqHz).toBeGreaterThan(2.0);
    await expectAudioDecay(page, kirnC5.rms, 0.15, 3000);

    await e4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await waitAudioBlocks(page, 96);
    const kirnE4 = await sampleDestinationAudio(page, 300, 360);
    await e4Key.dispatchEvent('pointerup', { buttons: 0 });
    expect(kirnE4.nonFiniteCount).toBe(0);
    expect(kirnE4.rms).toBeGreaterThan(0.002);
    expect(eqE4.peakFreqHz - kirnE4.peakFreqHz).toBeGreaterThan(0.3);

    // Cycle remaining temperaments (Werckmeister III=1, Vallotti=3) and verify A4 stays anchored at ~440 Hz
    for (const id of ['1', '3']) {
      await page.getByTestId('panic-btn').click();
      await waitForSilence(page);
      await tempSelect.selectOption(id);
      await waitForPrepIdle(page);
      await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
      await waitAudioBlocks(page, 96);
      const a4Stats = await sampleDestinationAudio(page, 410, 470);
      await a4Key.dispatchEvent('pointerup', { buttons: 0 });
      expect(a4Stats.nonFiniteCount).toBe(0);
      expect(a4Stats.peak).toBeLessThanOrEqual(1.0);
      expect(a4Stats.peakFreqHz).toBeGreaterThanOrEqual(435);
      expect(a4Stats.peakFreqHz).toBeLessThanOrEqual(445);
    }
  });

  test('IV-1: Condition macro (0% vs 100%) alters multi-key acoustic profile deterministically and cleanly', async ({
    page,
  }) => {
    const c4Key = page.locator('[data-midi="60"]');
    const e4Key = page.locator('[data-midi="64"]');

    // Mint condition (0%)
    await page.getByTestId('preset-concert-grand').click();
    expect(Number(await page.getByTestId('voicing-condition').inputValue())).toBe(0);
    await waitForPrepIdle(page);

    await c4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await e4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await waitAudioBlocks(page, 96);
    const mintStats = await sampleDestinationAudio(page);
    await c4Key.dispatchEvent('pointerup', { buttons: 0 });
    await e4Key.dispatchEvent('pointerup', { buttons: 0 });

    await page.getByTestId('panic-btn').click();
    await waitForSilence(page);

    // Heavily aged condition (100%)
    await setSlider(page, 'voicing-condition', 100);
    await expect(page.getByTestId('preset-custom-badge')).toBeVisible();
    await waitForPrepIdle(page);

    await c4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await e4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await waitAudioBlocks(page, 96);
    const agedStats = await sampleDestinationAudio(page);
    await c4Key.dispatchEvent('pointerup', { buttons: 0 });
    await e4Key.dispatchEvent('pointerup', { buttons: 0 });

    expect(agedStats.nonFiniteCount).toBe(0);
    expect(agedStats.peak).toBeLessThanOrEqual(1.0);
    expect(agedStats.rms).toBeGreaterThan(0.002);
    const centroidDelta = Math.abs(agedStats.spectralCentroidHz - mintStats.spectralCentroidHz);
    const rmsDelta = Math.abs(agedStats.rms - mintStats.rms);
    expect(centroidDelta + rmsDelta * 1000).toBeGreaterThan(1.0);
  });

  test('IV-2: Keyboard Y-coordinate strike position modulates velocity dynamically', async ({
    page,
  }) => {
    const c4Key = page.locator('[data-midi="60"]');
    const box = await c4Key.boundingBox();
    if (!box) throw new Error('Missing C4 bounding box');

    const centerX = box.x + box.width * 0.5;
    const topY = box.y + box.height * 0.08;
    const bottomY = box.y + box.height * 0.92;

    // Upper key strike (soft)
    await page.mouse.move(centerX, topY);
    await page.mouse.down();
    await waitAudioBlocks(page, 96);
    const topStats = await sampleDestinationAudio(page);
    await page.mouse.up();
    expect(topStats.nonFiniteCount).toBe(0);
    expect(topStats.rms).toBeGreaterThan(0.0002);
    await expectAudioDecay(page, topStats.rms, 0.15, 3500);

    // Lower key strike (strong)
    await page.mouse.move(centerX, bottomY);
    await page.mouse.down();
    await waitAudioBlocks(page, 96);
    const bottomStats = await sampleDestinationAudio(page);
    await page.mouse.up();
    expect(bottomStats.nonFiniteCount).toBe(0);
    expect(bottomStats.peak).toBeLessThanOrEqual(1.0);
    expect(bottomStats.rms).toBeGreaterThan(topStats.rms * 1.8);
    expect(bottomStats.spectralCentroidHz).toBeGreaterThan(topStats.spectralCentroidHz);
  });

  test('IV-2: All 6 humanized demo scores contain micro-rolled chords, velocity voicing, and half-pedaling', async ({
    page,
  }) => {
    test.setTimeout(90_000);
    expect(DEMO_SCORES.length).toBe(6);
    for (const desc of DEMO_SCORES) {
      const bytes = fs.readFileSync(
        fileURLToPath(new URL(`../public/demos/${desc.fileName}`, import.meta.url)),
      );
      const smf = parseSmf(bytes);
      const noteOns = smf.events.filter((e): e is MidiNoteEvent => e.type === 'noteOn');
      const cc64s = smf.events.filter(
        (e): e is MidiCcEvent => e.type === 'cc' && e.controller === 64,
      );

      expect(noteOns.length, `${desc.id} noteOns`).toBeGreaterThan(10);
      let minVel = 127;
      let maxVel = 1;
      let hasMicroRoll = false;
      for (let i = 0; i < noteOns.length; i++) {
        const ev = noteOns[i];
        expect(ev.midiKey).toBeGreaterThanOrEqual(21);
        expect(ev.midiKey).toBeLessThanOrEqual(108);
        expect(ev.velocity).toBeGreaterThanOrEqual(1);
        expect(ev.velocity).toBeLessThanOrEqual(127);
        if (ev.velocity < minVel) minVel = ev.velocity;
        if (ev.velocity > maxVel) maxVel = ev.velocity;
        if (i > 0) {
          const dt = ev.timeSeconds - noteOns[i - 1].timeSeconds;
          if (dt > 0.002 && dt < 0.04) hasMicroRoll = true;
        }
      }
      expect(maxVel - minVel, `${desc.id} velocity contrast`).toBeGreaterThanOrEqual(18);
      expect(hasMicroRoll, `${desc.id} micro-roll`).toBe(true);
      const hasHalfPedal = cc64s.some((c) => c.value > 0 && c.value < 127);
      expect(hasHalfPedal, `${desc.id} half-pedal cc64`).toBe(true);
    }

    // Render and play humanized Debussy Clair de Lune via Offline MIDI Worker
    await page.getByTestId('demo-select').selectOption('debussy-claire');
    await page.getByTestId('transport-play').click();
    await waitForOfflineRenderDone(page);
    await waitForAudioRms(page, 0.002);
    await page.getByTestId('transport-stop').click();
  });

  test('IV-2: Live Web recording exports valid 24-bit TPDF and 32-bit Float RIFF/WAVE files', async ({
    page,
  }) => {
    const startBtn = page.getByTestId('record-start');
    const stopBtn = page.getByTestId('record-stop');
    const formatSel = page.getByTestId('record-format');
    const exportBtn = page.getByTestId('record-export');
    const statusEl = page.getByTestId('record-status');

    await expect(exportBtn).toBeDisabled();
    await startBtn.click();
    await expect(stopBtn).toBeEnabled();
    await expect(statusEl).toContainText('REC');

    // Record a real C major triad (~427 ms = 160 blocks > 0.25 s)
    for (const midi of [60, 64, 67]) {
      await page.locator(`[data-midi="${midi}"]`).dispatchEvent('pointerdown', { buttons: 1 });
    }
    await waitAudioBlocks(page, 160);
    for (const midi of [60, 64, 67]) {
      await page.locator(`[data-midi="${midi}"]`).dispatchEvent('pointerup', { buttons: 0 });
    }

    await stopBtn.click();
    await expect(statusEl).toContainText('Ready', { timeout: 3000 });
    await expect(exportBtn).toBeEnabled();

    // 1. Export 24-bit PCM TPDF WAV
    await formatSel.selectOption('pcm24');
    const [dl24] = await Promise.all([page.waitForEvent('download'), exportBtn.click()]);
    const path24 = await dl24.path();
    if (!path24) throw new Error('Missing download path for 24-bit WAV');
    const buf24 = fs.readFileSync(path24);

    expect(buf24.subarray(0, 4).toString('ascii')).toBe('RIFF');
    expect(buf24.subarray(8, 12).toString('ascii')).toBe('WAVE');
    expect(buf24.subarray(12, 16).toString('ascii')).toBe('fmt ');
    expect(buf24.readUInt16LE(20)).toBe(1); // PCM
    expect(buf24.readUInt16LE(22)).toBe(2); // Stereo
    expect(buf24.readUInt32LE(24)).toBe(48_000);
    expect(buf24.readUInt16LE(32)).toBe(6); // blockAlign
    expect(buf24.readUInt16LE(34)).toBe(24); // bitsPerSample
    expect(buf24.subarray(36, 40).toString('ascii')).toBe('data');
    const dataBytes24 = buf24.readUInt32LE(40);
    expect(dataBytes24).toBeGreaterThan(48_000 * 0.25 * 6);
    expect(buf24.length).toBe(44 + dataBytes24);

    // 2. Export 32-bit IEEE Float WAV
    await formatSel.selectOption('f32');
    const [dl32] = await Promise.all([page.waitForEvent('download'), exportBtn.click()]);
    const path32 = await dl32.path();
    if (!path32) throw new Error('Missing download path for 32-bit float WAV');
    const buf32 = fs.readFileSync(path32);

    expect(buf32.subarray(0, 4).toString('ascii')).toBe('RIFF');
    expect(buf32.subarray(8, 12).toString('ascii')).toBe('WAVE');
    expect(buf32.readUInt16LE(20)).toBe(3); // IEEE float
    expect(buf32.readUInt16LE(22)).toBe(2);
    expect(buf32.readUInt32LE(24)).toBe(48_000);
    expect(buf32.readUInt16LE(32)).toBe(8);
    expect(buf32.readUInt16LE(34)).toBe(32);
    const dataBytes32 = buf32.readUInt32LE(40);
    expect(dataBytes32).toBeGreaterThan(48_000 * 0.25 * 8);
    expect(buf32.length).toBe(44 + dataBytes32);

    let peakF32 = 0;
    let nonFiniteF32 = 0;
    const totalF32Samples = dataBytes32 / 4;
    for (let i = 0; i < totalF32Samples; i++) {
      const s = buf32.readFloatLE(44 + i * 4);
      if (!Number.isFinite(s)) {
        nonFiniteF32++;
        continue;
      }
      const a = Math.abs(s);
      if (a > peakF32) peakF32 = a;
    }
    expect(nonFiniteF32).toBe(0);
    expect(peakF32).toBeGreaterThan(0.005);
    expect(peakF32).toBeLessThanOrEqual(1.0);
  });
});
