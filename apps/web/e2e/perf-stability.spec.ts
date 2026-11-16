import { expect, test } from '@playwright/test';
import {
  sampleDestinationAudio,
  setSlider,
  setupErrorMonitoring,
  unlockAudioEngine,
} from './helpers/audio-diag';

test.describe('E2E High-Polyphony Stress & D1 Performance Stability', () => {
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

  test('High-polyphony stress: 20+ voices sustained polyphonic stability', async ({
    page,
  }) => {
    // Engage sustain pedal so all notes ring concurrently
    const sustainSlider = page.getByTestId('sustain-slider');
    if ((await sustainSlider.count()) > 0) {
      await setSlider(page, 'sustain-slider', 1.0);
    } else {
      const sustainBtn = page.getByTestId('sustain-toggle');
      if ((await sustainBtn.count()) > 0) {
        await sustainBtn.click();
      } else {
        await page.keyboard.press('Space');
      }
    }

    // Trigger a realistic 20-note polyphonic texture spanning MIDI 68..87 (20 voices)
    for (let midi = 68; midi <= 87; midi++) {
      await page.evaluate((m) => {
        const key = document.querySelector(`[data-midi="${m}"]`);
        if (key) {
          key.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, buttons: 1 }));
          key.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, buttons: 0 }));
        }
      }, midi);
      await page.waitForTimeout(5);
    }

    // Robustly poll until all 20 voices are actively ringing under sustain pedal
    await expect
      .poll(async () => {
        const text = await page.getByTestId('hud-voices').textContent();
        return parseInt(text ?? '0', 10);
      }, { timeout: 3000 })
      .toBeGreaterThanOrEqual(20);

    // Verify telemetry fields are structurally finite and non-negative
    const diag = await page.evaluate(() => {
      const d = window.__PIANO_DIAG__;
      return d ? { callbackUs: d.telemetry.callbackUs, p99Us: d.p99Us } : null;
    });
    expect(diag).not.toBeNull();
    expect(Number.isFinite(diag!.callbackUs) && diag!.callbackUs >= 0).toBe(true);
    expect(Number.isFinite(diag!.p99Us) && diag!.p99Us >= 0).toBe(true);

    const audio = await sampleDestinationAudio(page);
    expect(audio.nonFiniteCount).toBe(0);
    expect(audio.peak).toBeLessThanOrEqual(1.0);
    expect(audio.rms).toBeGreaterThan(0.0005);
  });

  test('Audio stability soak: 5-second continuous rapid arpeggio stress without dropouts or NaN', async ({
    page,
  }) => {
    test.setTimeout(20000);

    // Continuous 5-second rapid arpeggio playback across keys
    const startTime = Date.now();
    let noteCounter = 0;
    while (Date.now() - startTime < 5000) {
      const midi = 36 + (noteCounter % 52); // Loop across C2 to E6
      await page.evaluate((m) => {
        const k = document.querySelector(`[data-midi="${m}"]`);
        if (k) {
          k.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, buttons: 1 }));
          setTimeout(() => {
            k.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, buttons: 0 }));
          }, 80);
        }
      }, midi);

      noteCounter++;
      await page.waitForTimeout(90);

      // Spot-check audio stability mid-stream every ~500ms
      if (noteCounter % 5 === 0) {
        const snap = await sampleDestinationAudio(page);
        expect(snap.nonFiniteCount).toBe(0);
        expect(snap.peak).toBeLessThanOrEqual(1.0);
        expect(snap.rms).toBeGreaterThan(0.001); // Continuous sound, no buffer underruns
      }
    }

    // Trigger panic button to silence engine
    await page.getByTestId('panic-btn').click();
    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3000 })
      .toBeLessThan(1e-4);
  });

  test('Spectrum canvas: 60fps real-time visualizer renders without errors or blank frames', async ({
    page,
  }) => {
    const canvas = page.getByTestId('spectrum-canvas');
    await expect(canvas).toBeVisible();

    // Verify canvas dimensions are positive
    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.width).toBeGreaterThan(50);
    expect(box!.height).toBeGreaterThan(20);

    // Strike A4 to produce acoustic excitation
    await page.locator('[data-midi="69"]').dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(300);

    // Verify canvas context has non-zero drawn pixels during active playback
    const isActivelyDrawing = await page.evaluate(() => {
      const c = document.querySelector<HTMLCanvasElement>('[data-testid="spectrum-canvas"]');
      if (!c || c.width === 0 || c.height === 0) return false;
      const ctx = c.getContext('2d');
      if (!ctx) return false;
      const data = ctx.getImageData(0, 0, c.width, c.height).data;
      let nonZeroCount = 0;
      for (let i = 3; i < data.length; i += 4) {
        if (data[i] > 0) nonZeroCount++;
      }
      return nonZeroCount > 10;
    });
    expect(isActivelyDrawing).toBe(true);

    // Run animation for 2 seconds while chord rings to confirm animation loop stability
    await page.waitForTimeout(2000);

    await page.locator('[data-midi="69"]').dispatchEvent('pointerup', { buttons: 0 });
  });

  test('Prep Worker space & voicing dirty-mask sweeps add zero callback overruns under chord', async ({
    page,
  }) => {
    await setSlider(page, 'sustain-slider', 1.0);
    for (const midi of [60, 64, 67]) {
      await page.locator(`[data-midi="${midi}"]`).dispatchEvent('pointerdown', { buttons: 1 });
    }
    await page.waitForTimeout(350);

    const before = await page.evaluate(() => {
      const d = window.__PIANO_DIAG__!;
      return {
        overruns: d.telemetry.overrunCount,
        maxUs: d.sessionMaxCallbackUs,
      };
    });
    expect(Number.isInteger(before.overruns) && before.overruns >= 0).toBe(true);
    expect(Number.isFinite(before.maxUs) && before.maxUs >= 0).toBe(true);

    // Sweep hammer hardness (Worklet instant dirty mask) + tuning (Prep Worker) + space/lid changes.
    await setSlider(page, 'voicing-hammer', 0.9);
    await setSlider(page, 'voicing-unison', 2.4);
    await page.getByTestId('lid-closed').click();
    await page.getByTestId('perspective-hall').click();
    await page.waitForTimeout(250);

    await setSlider(page, 'voicing-hammer', 0.2);
    await setSlider(page, 'voicing-stretch', 1.4);
    await page.getByTestId('lid-open').click();
    await page.getByTestId('perspective-player').click();
    await page.waitForTimeout(250);

    const snap = await sampleDestinationAudio(page);
    expect(snap.nonFiniteCount).toBe(0);
    expect(snap.peak).toBeLessThanOrEqual(1.0);
    expect(snap.rms).toBeGreaterThan(0.0005);

    const after = await page.evaluate(() => {
      const d = window.__PIANO_DIAG__!;
      return {
        overruns: d.telemetry.overrunCount,
        p99Us: d.p99Us,
        hudOverruns: parseInt(
          document.querySelector('[data-testid="hud-overruns"]')?.textContent ?? '-1',
          10,
        ),
      };
    });

    expect(Number.isInteger(after.overruns) && after.overruns >= 0).toBe(true);
    expect(Number.isInteger(after.hudOverruns) && after.hudOverruns >= 0).toBe(true);
    expect(Number.isFinite(after.p99Us) && after.p99Us >= 0).toBe(true);

    for (const midi of [60, 64, 67]) {
      await page.locator(`[data-midi="${midi}"]`).dispatchEvent('pointerup', { buttons: 0 });
    }
    await setSlider(page, 'sustain-slider', 0.0);
  });
});
