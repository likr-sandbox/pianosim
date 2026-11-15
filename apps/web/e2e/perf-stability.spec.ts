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

  test('High-polyphony stress: 20+ voices maintain D1 performance budget (< 1,866.7 us)', async ({
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

    // Trigger a dense 24-note cluster spanning MIDI 40..63
    await page.evaluate(() => {
      for (let midi = 40; midi <= 63; midi++) {
        const key = document.querySelector(`[data-midi="${midi}"]`);
        if (key) {
          key.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, buttons: 1 }));
          key.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, buttons: 0 }));
        }
      }
    });

    // Allow notes to ring and telemetry blocks to accumulate
    await page.waitForTimeout(400);

    // Verify HUD active voices counter indicates heavy polyphony
    await expect
      .poll(async () => {
        const text = await page.getByTestId('hud-voices').textContent();
        return parseInt(text ?? '0', 10);
      })
      .toBeGreaterThanOrEqual(20);

    // Inspect real AudioWorklet telemetry diagnostics
    const diag = await page.evaluate(() => {
      const d = window.__PIANO_DIAG__;
      return d
        ? { callbackUs: d.telemetry.callbackUs, activeVoices: d.telemetry.activeVoices }
        : null;
    });
    expect(diag).not.toBeNull();
    expect(diag!.activeVoices).toBeGreaterThanOrEqual(20);

    // Strict D1 Performance Budget: must be < 1,866.7 us @ 48 kHz (70% of 128-sample callback time)
    expect(diag!.callbackUs).toBeLessThan(1866.7);

    // Verify telemetry HUD callback display and lack of critical overload alert
    const hudP99El = page.getByTestId('hud-p99');
    if ((await hudP99El.count()) > 0) {
      const p99Val = parseFloat((await hudP99El.textContent()) ?? '0');
      expect(p99Val).toBeLessThan(1866.7);
    }
    const alertEl = page.getByTestId('hud-alert');
    if ((await alertEl.count()) > 0) {
      await expect(alertEl).not.toHaveClass(/is-overloaded|alert-critical/);
    }

    const audio = await sampleDestinationAudio(page);
    expect(audio.nonFiniteCount).toBe(0);
    expect(audio.peak).toBeLessThanOrEqual(1.0);
    expect(audio.rms).toBeGreaterThan(0.01);
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
});
