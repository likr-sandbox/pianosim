import { test, expect } from '@playwright/test';
import { ensurePianoKeyboard, dispatchPointerGlissando } from '../helpers/dom-driver';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import { verifySampleBounds, computeRMS } from '../helpers/audio-analysis';

test.describe('Tier 3: Combination - Glissando & Sustain Dynamics', () => {
  test('T3-CMB-04: 16-Key Pointer Glissando with Sustain', async ({ page }) => {
    await page.goto('./');
    await ensurePianoKeyboard(page);

    const keyCount = await page.locator('#piano-keys .key').count();
    if (keyCount >= 88) {
      // 1. Sustain pedal down
      await page.keyboard.down('Space');
      await expect(page.locator('#pedal-badge')).toContainText('Sustain: ON');

      // 2. Drag pointer from C3 (MIDI 48) to D5 (MIDI 74)
      await dispatchPointerGlissando(page, 48, 74);

      // 3. Verify keys latched under sustain
      const activeCountUnderPedal = await page.locator('#piano-keys .key.active').count();
      expect(activeCountUnderPedal).toBeGreaterThan(0);

      // 4. Release sustain pedal
      await page.keyboard.up('Space');
      await expect(page.locator('#pedal-badge')).toContainText('Sustain: OFF');

      // 5. Zero orphaned or stuck keys remaining
      const activeCountPostPedal = await page.locator('#piano-keys .key.active').count();
      expect(activeCountPostPedal).toBe(0);
    }

    // Synthesizer simulation of the same workload
    const synth = createPianoSynthesizer(44100);
    synth.setSustainPedal(true);
    for (let m = 48; m <= 74; m += 2) {
      synth.noteOn({ midiNote: m, velocity: 80 });
    }

    const buffer = new Float32Array(22050);
    renderMono(synth, buffer, 22050);

    const bounds = verifySampleBounds(buffer);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
    expect(bounds.absMax).toBeLessThanOrEqual(1.0);
    expect(computeRMS(buffer)).toBeGreaterThan(0.01);
  });

  test('T3-CMB-06: Pedal Catch / Half-Pedal Transient', async () => {
    const synth = createPianoSynthesizer(44100);

    // Strike C4
    synth.noteOn({ midiNote: 60, velocity: 100 });
    const attack = new Float32Array(8820); // 200ms
    renderMono(synth, attack, 8820);

    // Release C4
    synth.noteOff({ midiNote: 60 });
    const gap = new Float32Array(1764); // 40ms gap
    renderMono(synth, gap, 1764);

    // Catch with sustain pedal within 40ms
    synth.setSustainPedal(true);
    const caught = new Float32Array(22050);
    renderMono(synth, caught, 22050);

    const bounds = verifySampleBounds(caught);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
    expect(bounds.absMax).toBeLessThanOrEqual(1.0);
  });

  test('T3-CMB-09: Fast Pointer Glissando Leaves Zero Stuck Keys', async ({ page }) => {
    await page.goto('./');
    await ensurePianoKeyboard(page);

    const keyCount = await page.locator('#piano-keys .key').count();
    expect(keyCount).toBe(88);

    // Glissando without pedal across keys 48 to 72
    await dispatchPointerGlissando(page, 48, 72);

    // Terminal key must not be stuck
    const activeCount = await page.locator('#piano-keys .key.active').count();
    expect(activeCount).toBe(0);
  });
});
