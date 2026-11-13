import { test, expect, Page } from '@playwright/test';
import { ensurePianoKeyboard, trackConsoleErrors, pressPianoKey, releasePianoKey } from '../helpers/dom-driver';
import { installAudioTap, tapReset, tapStats, tapSnapshot } from '../helpers/page-audio';
import { computeFFTPeak } from '../helpers/audio-analysis';

/**
 * Full path, nothing mocked: DOM gesture -> UI -> PianoAudioEngine -> AudioWorklet running the
 * production PianoSynthesizer -> AudioContext destination. A passive AnalyserNode tap observes the
 * samples that actually reach the destination.
 */
async function startAudio(page: Page) {
  await page.goto('./');
  await ensurePianoKeyboard(page);
  await page.locator('#start-audio-btn').click();
  await expect(page.locator('#audio-status')).toContainText(/Engine Initialized|Audio Active/i);
  await expect.poll(async () => (await tapStats(page)).polls, { timeout: 10000 }).toBeGreaterThan(5);
}

test.describe('Tier 1: UI -> real engine -> audio output', () => {
  test('T1-PATH-01: silence before any key; pressing a key produces finite, bounded audio', async ({ page }) => {
    const tracker = trackConsoleErrors(page);
    await installAudioTap(page);
    await startAudio(page);
    await page.waitForTimeout(300);
    await tapReset(page);
    await page.waitForTimeout(300);
    expect((await tapStats(page)).maxRms).toBeLessThan(1e-6);

    await pressPianoKey(page, 60);
    await expect.poll(async () => (await tapStats(page)).maxRms, { timeout: 5000 }).toBeGreaterThan(0.01);
    const st = await tapStats(page);
    expect(st.nonFinite).toBe(0);
    expect(st.absMax).toBeLessThanOrEqual(1.0);
    expect(tracker.errors).toHaveLength(0);
    expect(tracker.pageErrors).toHaveLength(0);
  });

  test('T1-PATH-02: A4 key yields a 440 Hz fundamental at the destination', async ({ page }) => {
    await installAudioTap(page);
    await startAudio(page);
    await pressPianoKey(page, 69);
    await page.waitForTimeout(900);
    const { sampleRate, samples } = await tapSnapshot(page);
    const hz = computeFFTPeak(samples.subarray(0, 32768), sampleRate, 440.0);
    expect(Math.abs(hz - 440.0)).toBeLessThanOrEqual(1.5);
  });

  test('T1-PATH-03: releasing a key damps the output; sustain pedal keeps it ringing', async ({ page }) => {
    await installAudioTap(page);
    await startAudio(page);

    const settledRms = async () => {
      await page.waitForTimeout(700);
      return (await tapStats(page)).lastRms;
    };

    // Without pedal: press, then release -> damped
    await pressPianoKey(page, 60);
    await page.waitForTimeout(300);
    const held = (await tapStats(page)).lastRms;
    expect(held).toBeGreaterThan(0.01);
    await releasePianoKey(page, 60);
    const damped = await settledRms();
    expect(damped).toBeLessThan(held * 0.1);

    // With pedal: press, release key, note continues to ring
    await page.keyboard.down('Space');
    await pressPianoKey(page, 64);
    await page.waitForTimeout(300);
    await releasePianoKey(page, 64);
    await page.waitForTimeout(400);
    const sustained = (await tapStats(page)).lastRms;
    expect(sustained).toBeGreaterThan(damped * 5);
    expect(sustained).toBeGreaterThan(0.005);

    // Pedal up -> damper drops
    await page.keyboard.up('Space');
    expect(await settledRms()).toBeLessThan(sustained * 0.1);
  });

  test('T1-PATH-04: velocity slider changes loudness reaching the output', async ({ page }) => {
    await installAudioTap(page);
    await startAudio(page);
    const slider = page.locator('#velocity-slider');

    const peakFor = async (vel: number) => {
      await slider.fill(String(vel));
      await page.waitForTimeout(900); // let previous note die
      await tapReset(page);
      await pressPianoKey(page, 60);
      await page.waitForTimeout(400);
      await releasePianoKey(page, 60);
      return (await tapStats(page)).maxRms;
    };
    const soft = await peakFor(25);
    const loud = await peakFor(120);
    expect(loud).toBeGreaterThan(soft * 1.5);
  });
});
