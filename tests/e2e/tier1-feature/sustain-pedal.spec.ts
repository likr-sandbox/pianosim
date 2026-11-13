import { test, expect } from '@playwright/test';
import { ensurePianoKeyboard } from '../helpers/dom-driver';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import { computeRMS, computeDynamicRangeDB } from '../helpers/audio-analysis';

test.describe('Tier 1: Feature Coverage - Sustain Pedal', () => {
  test('T1-SUS-01: Spacebar Sustain Pedal Toggle', async ({ page }) => {
    await page.goto('./');
    await ensurePianoKeyboard(page);

    const keyCount = await page.locator('#piano-keys .key').count();
    expect(keyCount).toBe(88);

    const pedalBadge = page.locator('#pedal-badge');

    // Space down -> ON
    await page.keyboard.down('Space');
    await expect(pedalBadge).toContainText('Sustain: ON');
    expect(await pedalBadge.getAttribute('class')).toContain('active');

    // Space up -> OFF
    await page.keyboard.up('Space');
    await expect(pedalBadge).toContainText('Sustain: OFF');
    expect(await pedalBadge.getAttribute('class')).not.toContain('active');
  });

  test('T1-SUS-02: Sustain Holds Note on Key Release', async () => {
    const synth = createPianoSynthesizer(44100);

    // Engage pedal
    synth.setSustainPedal(true);
    synth.noteOn({ midiNote: 60, velocity: 100 });

    // Render initial attack (200ms)
    const attackBuf = new Float32Array(8820);
    renderMono(synth, attackBuf, 8820);
    const attackRMS = computeRMS(attackBuf, 4410, 8820);

    // Release key while pedal is held
    synth.noteOff({ midiNote: 60 });

    // Render 1.0 second under sustain
    const sustainBuf = new Float32Array(44100);
    renderMono(synth, sustainBuf, 44100);
    const sustainRMS = computeRMS(sustainBuf, 22050, 44100);

    // Audio continues vibrating at > 20% of initial RMS energy
    expect(sustainRMS).toBeGreaterThan(attackRMS * 0.2);
  });

  test('T1-SUS-03: Damper Drops on Pedal Release', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.setSustainPedal(true);
    synth.noteOn({ midiNote: 60, velocity: 100 });

    // Initial attack / key hold (50ms / 2205 samples) before physical key release
    const attackBuf = new Float32Array(2205);
    renderMono(synth, attackBuf, 2205);

    // Release key (sustained by pedal)
    synth.noteOff({ midiNote: 60 });

    // Ring for 0.5s
    const ringBuf = new Float32Array(22050);
    renderMono(synth, ringBuf, 22050);
    const ringRMS = computeRMS(ringBuf, 11025, 22050);

    // Release pedal
    synth.setSustainPedal(false);

    // Render 150ms after pedal release (6615 samples)
    const releaseBuf = new Float32Array(6615);
    renderMono(synth, releaseBuf, 6615);
    const releasedRMS = computeRMS(releaseBuf, 4410, 6615);

    // Audio drops by > 20 dB
    const dbDrop = computeDynamicRangeDB(ringRMS, releasedRMS);
    expect(dbDrop).toBeGreaterThan(20.0);
  });

  test('T1-SUS-04: Pedal Depressed While Key Held', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.noteOn({ midiNote: 64, velocity: 100 });

    // Ring while key is held
    const heldBuf = new Float32Array(8820);
    renderMono(synth, heldBuf, 8820);
    expect(computeRMS(heldBuf)).toBeGreaterThan(0.01);

    // Press pedal down, then release key
    synth.setSustainPedal(true);
    synth.noteOff({ midiNote: 64 });

    const postBuf = new Float32Array(22050);
    renderMono(synth, postBuf, 22050);

    // Transitions seamlessly without cutting or silence
    expect(computeRMS(postBuf)).toBeGreaterThan(0.005);
  });

  test('T1-SUS-05: Visual Pedal Indicator Synchronization', async ({ page }) => {
    await page.goto('./');
    await ensurePianoKeyboard(page);

    const keyCount = await page.locator('#piano-keys .key').count();
    expect(keyCount).toBe(88);

    const pedalBadge = page.locator('#pedal-badge');
    expect(await pedalBadge.isVisible()).toBe(true);

    // Depress Space
    await page.keyboard.down('Space');
    await expect(pedalBadge).toContainText('Sustain: ON');

    // Release Space
    await page.keyboard.up('Space');
    await expect(pedalBadge).toContainText('Sustain: OFF');
  });
});
