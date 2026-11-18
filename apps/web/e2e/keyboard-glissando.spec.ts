import { expect, test, type Page } from '@playwright/test';
import {
  expectAudioDecay,
  sampleDestinationAudio,
  setSlider,
  setupErrorMonitoring,
  unlockAudioEngine,
} from './helpers/audio-diag';

async function getKeyCenter(page: Page, midi: number): Promise<{ x: number; y: number }> {
  const el = page.locator(`[data-midi="${midi}"]`);
  await expect(el).toBeAttached();
  const box = await el.boundingBox();
  if (!box) throw new Error(`Missing bounding box for key MIDI ${midi}`);
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

test.describe('88-Key Keyboard & Glissando E2E Suite', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('./');
    await unlockAudioEngine(page);
  });

  // ==========================================
  // Tier 1: 88-Key DOM Verification & Single Note
  // ==========================================
  test('Tier 1: 88-key DOM layout verification (A0=21 to C8=108, 52 white, 36 black)', async ({
    page,
  }) => {
    const assertNoErrors = setupErrorMonitoring(page);

    const keys = page.locator('.piano-key');
    await expect(keys).toHaveCount(88);

    const whiteKeys = page.locator('.piano-key.key-white, .piano-key.white-key');
    const blackKeys = page.locator('.piano-key.key-black, .piano-key.black-key');
    await expect(whiteKeys).toHaveCount(52);
    await expect(blackKeys).toHaveCount(36);

    // Verify boundary and contiguous MIDI ordering.
    await expect(keys.first()).toHaveAttribute('data-midi', '21'); // A0
    await expect(keys.last()).toHaveAttribute('data-midi', '108'); // C8

    const midiValues = await keys.evaluateAll((list) =>
      list.map((el) => Number(el.getAttribute('data-midi'))),
    );
    for (let i = 0; i < 88; i++) {
      expect(midiValues[i]).toBe(21 + i);
    }

    assertNoErrors();
  });

  test('Tier 1: single-key click/touch trigger, active class, and release decay', async ({
    page,
  }) => {
    const assertNoErrors = setupErrorMonitoring(page);

    // Test C4 (MIDI 60) white key.
    const c4Pos = await getKeyCenter(page, 60);
    await page.mouse.move(c4Pos.x, c4Pos.y);
    await page.mouse.down();
    await expect(page.locator('[data-midi="60"]')).toHaveClass(/is-active/);
    await page.waitForTimeout(420);

    const c4Stats = await sampleDestinationAudio(page);
    expect(c4Stats.nonFiniteCount).toBe(0);
    expect(c4Stats.peak).toBeLessThanOrEqual(1.0);
    expect(c4Stats.rms).toBeGreaterThan(0.002);
    expect(c4Stats.peakFreqHz).toBeGreaterThanOrEqual(255);
    expect(c4Stats.peakFreqHz).toBeLessThanOrEqual(268);

    await page.mouse.up();
    await expect(page.locator('[data-midi="60"]')).not.toHaveClass(/is-active/);
    await expectAudioDecay(page, c4Stats.rms);

    // Test F#4 (MIDI 66) black key.
    const fs4Pos = await getKeyCenter(page, 66);
    await page.mouse.move(fs4Pos.x, fs4Pos.y);
    await page.mouse.down();
    await expect(page.locator('[data-midi="66"]')).toHaveClass(/is-active/);
    await page.waitForTimeout(420);

    const fs4Stats = await sampleDestinationAudio(page);
    expect(fs4Stats.peakFreqHz).toBeGreaterThanOrEqual(360);
    expect(fs4Stats.peakFreqHz).toBeLessThanOrEqual(380);

    await page.mouse.up();
    await expect(page.locator('[data-midi="66"]')).not.toHaveClass(/is-active/);
    await expectAudioDecay(page, fs4Stats.rms);

    assertNoErrors();
  });

  // ==========================================
  // Tier 2: Boundary Keys, Velocities, Repetition & Blur
  // ==========================================
  test('Tier 2: boundary keys A0 (21) bass and C8 (108) treble acoustic response', async ({
    page,
  }) => {
    const assertNoErrors = setupErrorMonitoring(page);

    // A0 (27.5 Hz fundamental).
    const a0Pos = await getKeyCenter(page, 21);
    await page.mouse.move(a0Pos.x, a0Pos.y);
    await page.mouse.down();
    await page.waitForTimeout(450);
    const a0Stats = await sampleDestinationAudio(page);
    expect(a0Stats.nonFiniteCount).toBe(0);
    expect(a0Stats.rms).toBeGreaterThan(0.0005);
    await page.mouse.up();
    await expectAudioDecay(page, a0Stats.rms);

    // C8 (4186 Hz / 4358 Hz inharmonic stiffness partial).
    const c8Pos = await getKeyCenter(page, 108);
    await page.mouse.move(c8Pos.x, c8Pos.y);
    await page.mouse.down();
    await page.waitForTimeout(300);
    const c8Stats = await sampleDestinationAudio(page, { minFreq: 3500 });
    expect(c8Stats.nonFiniteCount).toBe(0);
    expect(c8Stats.rms).toBeGreaterThan(0.0005);
    expect(c8Stats.peakFreqHz).toBeGreaterThan(3900);
    await page.mouse.up();
    await expectAudioDecay(page, c8Stats.rms);

    assertNoErrors();
  });

  test('Tier 2: extreme velocities (pianissimo v=1 vs fortissimo v=127)', async ({ page }) => {
    const assertNoErrors = setupErrorMonitoring(page);
    const a4Pos = await getKeyCenter(page, 69);

    // Pianissimo v=1.
    await setSlider(page, 'velocity-slider', 1);
    await page.mouse.move(a4Pos.x, a4Pos.y);
    await page.mouse.down();
    await page.waitForTimeout(420);
    const ppStats = await sampleDestinationAudio(page);
    expect(ppStats.nonFiniteCount).toBe(0);
    expect(ppStats.rms).toBeGreaterThan(1e-5);
    await page.mouse.up();
    await expectAudioDecay(page, ppStats.rms);

    // Fortissimo v=127.
    await setSlider(page, 'velocity-slider', 127);
    await page.mouse.move(a4Pos.x, a4Pos.y);
    await page.mouse.down();
    await page.waitForTimeout(420);
    const ffStats = await sampleDestinationAudio(page);
    expect(ffStats.nonFiniteCount).toBe(0);
    expect(ffStats.peak).toBeLessThanOrEqual(1.0);
    expect(ffStats.rms).toBeGreaterThan(0.01);
    // Dynamic contrast ratio must exceed 18 dB (> 8x).
    expect(ffStats.rms / ppStats.rms).toBeGreaterThan(8.0);
    await page.mouse.up();
    await expectAudioDecay(page, ffStats.rms);

    assertNoErrors();
  });

  test('Tier 2: rapid repetition re-strikes maintain stability without voice leak', async ({
    page,
  }) => {
    const assertNoErrors = setupErrorMonitoring(page);
    const pos = await getKeyCenter(page, 69); // A4

    for (let i = 0; i < 5; i++) {
      await page.mouse.move(pos.x, pos.y);
      await page.mouse.down();
      await page.waitForTimeout(60);
      await page.mouse.up();
      await page.waitForTimeout(40);
    }
    await page.waitForTimeout(300);

    const stats = await sampleDestinationAudio(page);
    expect(stats.nonFiniteCount).toBe(0);
    expect(stats.peak).toBeLessThanOrEqual(1.0);
    await expectAudioDecay(page, stats.rms);

    assertNoErrors();
  });

  test('Tier 2: off-canvas pointer drag releases note cleanly', async ({ page }) => {
    const assertNoErrors = setupErrorMonitoring(page);
    const key = page.locator('[data-midi="64"]'); // E4
    const pos = await getKeyCenter(page, 64);

    await page.mouse.move(pos.x, pos.y);
    await page.mouse.down();
    await expect(key).toHaveClass(/is-active/);

    // Drag pointer outside the keyboard container.
    await page.mouse.move(pos.x, -100);
    await page.mouse.up();

    await expect(key).not.toHaveClass(/is-active/);
    const stats = await sampleDestinationAudio(page);
    await expectAudioDecay(page, Math.max(0.001, stats.rms));

    assertNoErrors();
  });

  test('Tier 2: window blur cancels active notes and silences audio', async ({ page }) => {
    const assertNoErrors = setupErrorMonitoring(page);
    const pos = await getKeyCenter(page, 60); // C4

    await page.mouse.move(pos.x, pos.y);
    await page.mouse.down();
    await expect(page.locator('[data-midi="60"]')).toHaveClass(/is-active/);
    await page.waitForTimeout(300);

    const sounding = await sampleDestinationAudio(page);
    expect(sounding.rms).toBeGreaterThan(0.001);

    // Dispatch window blur (simulating tab switch or OS focus loss).
    await page.evaluate(() => window.dispatchEvent(new Event('blur')));
    await expect(page.locator('[data-midi="60"]')).not.toHaveClass(/is-active/);
    await expectAudioDecay(page, sounding.rms);

    assertNoErrors();
  });

  // ==========================================
  // Tier 3: Octave Transpose & Multi-Touch
  // ==========================================
  test('Tier 3: PC keyboard octave transpose (Z/X keys) with frequency shift', async ({
    page,
  }) => {
    const assertNoErrors = setupErrorMonitoring(page);

    // Default octave: KeyA = C4 (261 Hz).
    await page.keyboard.down('KeyA');
    await page.waitForTimeout(420);
    const c4 = await sampleDestinationAudio(page);
    expect(c4.peakFreqHz).toBeGreaterThanOrEqual(255);
    expect(c4.peakFreqHz).toBeLessThanOrEqual(268);
    await page.keyboard.up('KeyA');
    await expectAudioDecay(page, c4.rms);

    // Transpose down with KeyZ: KeyA becomes C3 (130.8 Hz fundamental, 2nd partial 261.6 Hz).
    // Probe the 100..200 Hz band (analogous to C8) because finite hammer width (beta ~ 1/8)
    // and Zone 2 calibration make C3's 2nd partial louder than its 1st partial.
    await page.keyboard.press('KeyZ');
    await page.keyboard.down('KeyA');
    await expect(page.locator('[data-midi="48"]')).toHaveClass(/is-active/);
    await page.waitForTimeout(420);
    const c3 = await sampleDestinationAudio(page, { minFreq: 100, maxFreq: 200 });
    const c3RmsDb = 20 * Math.log10(Math.max(c3.rms, 1e-12));
    expect(c3.nonFiniteCount).toBe(0);
    expect(c3RmsDb).toBeGreaterThan(-45);
    expect(c3.peakFreqHz).toBeGreaterThanOrEqual(125);
    expect(c3.peakFreqHz).toBeLessThanOrEqual(136);
    await page.keyboard.up('KeyA');
    await expectAudioDecay(page, c3.rms);

    // Transpose up twice with KeyX: KeyA becomes C5 (523 Hz).
    await page.keyboard.press('KeyX');
    await page.keyboard.press('KeyX');
    await page.keyboard.down('KeyA');
    await page.waitForTimeout(420);
    const c5 = await sampleDestinationAudio(page);
    expect(c5.peakFreqHz).toBeGreaterThanOrEqual(510);
    expect(c5.peakFreqHz).toBeLessThanOrEqual(535);
    await page.keyboard.up('KeyA');
    await expectAudioDecay(page, c5.rms);

    assertNoErrors();
  });

  test('Tier 3: multi-touch concurrent polyphony (3-note triad and 5-note chord)', async ({
    page,
  }) => {
    const assertNoErrors = setupErrorMonitoring(page);

    // Trigger 3-note triad (C4, E4, G4 = 60, 64, 67) via concurrent PC keydowns.
    await page.keyboard.down('KeyA'); // C4
    await page.keyboard.down('KeyD'); // E4
    await page.keyboard.down('KeyG'); // G4
    await page.waitForTimeout(420);

    const triad = await sampleDestinationAudio(page);
    expect(triad.nonFiniteCount).toBe(0);
    expect(triad.peak).toBeLessThanOrEqual(1.0);
    expect(triad.rms).toBeGreaterThan(0.003);

    let voices = await page.evaluate(() => window.__PIANO_DIAG__?.telemetry.activeVoices ?? 0);
    expect(voices).toBeGreaterThanOrEqual(3);

    // Release triad notes.
    await page.keyboard.up('KeyA');
    await page.keyboard.up('KeyD');
    await page.keyboard.up('KeyG');
    await expectAudioDecay(page, triad.rms);

    // Trigger 5-note chord (C4, E4, G4, B4, D5) via concurrent PC keydowns.
    await page.keyboard.down('KeyA'); // C4
    await page.keyboard.down('KeyD'); // E4
    await page.keyboard.down('KeyG'); // G4
    await page.keyboard.down('KeyJ'); // B4
    await page.keyboard.down('KeyL'); // D5
    await page.waitForTimeout(420);

    const chord5 = await sampleDestinationAudio(page);
    expect(chord5.nonFiniteCount).toBe(0);
    expect(chord5.peak).toBeLessThanOrEqual(1.0);
    expect(chord5.rms).toBeGreaterThan(0.004);

    voices = await page.evaluate(() => window.__PIANO_DIAG__?.telemetry.activeVoices ?? 0);
    expect(voices).toBeGreaterThanOrEqual(5);

    await page.keyboard.up('KeyA');
    await page.keyboard.up('KeyD');
    await page.keyboard.up('KeyG');
    await page.keyboard.up('KeyJ');
    await page.keyboard.up('KeyL');
    await expectAudioDecay(page, chord5.rms);

    assertNoErrors();
  });

  // ==========================================
  // Tier 4: Continuous Glissando Sweep
  // ==========================================
  test('Tier 4: continuous glissando sweep across octaves with polyphony and RMS tracking', async ({
    page,
  }) => {
    const assertNoErrors = setupErrorMonitoring(page);

    const startPos = await getKeyCenter(page, 57); // A3
    const endPos = await getKeyCenter(page, 84); // C6

    await page.mouse.move(startPos.x, startPos.y);
    await page.mouse.down();

    // Smooth glissando sweep across 27 keys in 25 interpolated mousemove steps.
    await page.mouse.move(endPos.x, endPos.y, { steps: 25 });
    await page.waitForTimeout(300);

    const sweepAudio = await sampleDestinationAudio(page);
    expect(sweepAudio.nonFiniteCount).toBe(0);
    expect(sweepAudio.peak).toBeLessThanOrEqual(1.0);
    expect(sweepAudio.rms).toBeGreaterThan(0.005);

    const sweepVoices = await page.evaluate(
      () => window.__PIANO_DIAG__?.telemetry.activeVoices ?? 0,
    );
    expect(sweepVoices).toBeGreaterThanOrEqual(4);

    await page.mouse.up();
    await expectAudioDecay(page, sweepAudio.rms, 0.25, 4000);

    assertNoErrors();
  });
});
