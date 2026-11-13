import { test, expect } from '@playwright/test';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import { installAudioTelemetry } from '../helpers/page-audio';
import { verifySampleBounds, computeRMS } from '../helpers/audio-analysis';
import { ensurePianoKeyboard, trackConsoleErrors } from '../helpers/dom-driver';

test.describe('Tier 2: Boundary - Invalid Inputs & Error Handling', () => {
  test('T2-BND-09: Out-of-Range Low MIDI Note', async () => {
    const synth = createPianoSynthesizer(44100);

    expect(() => {
      synth.noteOn({ midiNote: 10, velocity: 80 });
      synth.noteOn({ midiNote: -1, velocity: 80 });
    }).not.toThrow();

    const buffer = new Float32Array(2048);
    renderMono(synth, buffer, 2048);

    const bounds = verifySampleBounds(buffer);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
  });

  test('T2-BND-10: Out-of-Range High MIDI Note', async () => {
    const synth = createPianoSynthesizer(44100);

    expect(() => {
      synth.noteOn({ midiNote: 109, velocity: 80 });
      synth.noteOn({ midiNote: 255, velocity: 80 });
    }).not.toThrow();

    const buffer = new Float32Array(2048);
    renderMono(synth, buffer, 2048);

    const bounds = verifySampleBounds(buffer);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
  });

  test('T2-BND-11: Out-of-Range & Non-Numeric Velocity', async () => {
    const synth = createPianoSynthesizer(44100);

    expect(() => {
      synth.noteOn({ midiNote: 60, velocity: -10 });
      synth.noteOn({ midiNote: 60, velocity: 150 });
      synth.noteOn({ midiNote: 60, velocity: NaN });
      synth.noteOn({ midiNote: 60, velocity: Infinity });
    }).not.toThrow();

    const buffer = new Float32Array(4096);
    renderMono(synth, buffer, 4096);

    const bounds = verifySampleBounds(buffer);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
    expect(bounds.absMax).toBeLessThanOrEqual(1.0);
  });

  test('T2-BND-12: Redundant NoteOff Without NoteOn', async () => {
    const synth = createPianoSynthesizer(44100);

    // Call noteOff on completely idle engine
    expect(() => {
      synth.noteOff({ midiNote: 60 });
      synth.noteOff({ midiNote: 72 });
    }).not.toThrow();

    const buffer = new Float32Array(2048);
    renderMono(synth, buffer, 2048);

    expect(computeRMS(buffer)).toBeLessThanOrEqual(1e-6);
  });

  test('T2-BND-13: Rapid Audio Unlock Flapping', async ({ page }) => {
    const tracker = trackConsoleErrors(page);
    await installAudioTelemetry(page);
    await page.goto('./');
    await ensurePianoKeyboard(page);

    const startBtn = page.locator('#start-audio-btn');

    // Rapidly click 10 times
    for (let i = 0; i < 10; i++) {
      await startBtn.click({ force: true });
    }

    expect(tracker.errors).toHaveLength(0);
    expect(tracker.pageErrors).toHaveLength(0);
  });
});
