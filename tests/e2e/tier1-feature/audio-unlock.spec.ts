import { test, expect } from '@playwright/test';
import { ensurePianoKeyboard, trackConsoleErrors, clickPianoKey } from '../helpers/dom-driver';
import { installAudioTelemetry, getAudioTelemetry } from '../helpers/page-audio';

test.describe('Tier 1: Feature Coverage - Audio Unlock & Autoplay Lifecycle', () => {
  test('T1-AUL-01: Initial Suspended State Check', async ({ page }) => {
    await installAudioTelemetry(page);
    await page.goto('./');
    await ensurePianoKeyboard(page);

    const audioStatus = page.locator('#audio-status');
    await expect(audioStatus).toContainText(/Ready to start/i);

    const statusDot = page.locator('#status-dot');
    expect(await statusDot.getAttribute('class')).not.toContain('active');
  });

  test('T1-AUL-02: Start Button Resumes AudioContext', async ({ page }) => {
    await installAudioTelemetry(page);
    await page.goto('./');
    await ensurePianoKeyboard(page);

    const startBtn = page.locator('#start-audio-btn');
    await startBtn.click();

    const audioStatus = page.locator('#audio-status');
    await expect(audioStatus).toContainText(/Engine Initialized|Audio Active/i);
    expect(await startBtn.isDisabled()).toBe(true);

    const statusDot = page.locator('#status-dot');
    expect(await statusDot.getAttribute('class')).toContain('active');
  });

  test('T1-AUL-03: Key Click Triggers Audio Unlock', async ({ page }) => {
    await installAudioTelemetry(page);
    await page.goto('./');
    await ensurePianoKeyboard(page);

    // Clicking piano key directly
    await clickPianoKey(page, 60);

    const freqPreview = page.locator('#freq-preview');
    await expect(freqPreview).toContainText('C4');
  });

  test('T1-AUL-04: Idempotent Audio Initialization', async ({ page }) => {
    const tracker = trackConsoleErrors(page);
    await installAudioTelemetry(page);
    await page.goto('./');
    await ensurePianoKeyboard(page);

    const startBtn = page.locator('#start-audio-btn');

    // 5 rapid clicks on start button
    for (let i = 0; i < 5; i++) {
      await startBtn.click({ force: true });
    }

    const telemetry = await getAudioTelemetry(page);
    expect(telemetry.instancesCount).toBeLessThanOrEqual(1);
    expect(tracker.errors).toHaveLength(0);
  });

  test('T1-AUL-05: Active Voice Visual Feedback', async ({ page }) => {
    await page.goto('./');
    await ensurePianoKeyboard(page);

    const startBtn = page.locator('#start-audio-btn');
    await startBtn.click();

    const statusDot = page.locator('#status-dot');
    expect(await statusDot.getAttribute('class')).toContain('active');
  });
});
