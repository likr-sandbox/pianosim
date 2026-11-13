import { test, expect } from '@playwright/test';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import { computeFFTPeak } from '../helpers/audio-analysis';
import { ensurePianoKeyboard, clickPianoKey } from '../helpers/dom-driver';

test.describe('Tier 1: Feature Coverage - Pitch & Frequency Accuracy', () => {
  test('T1-PIT-01: Concert Pitch A4 (440 Hz) FFT Peak', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.noteOn({ midiNote: 69, velocity: 100 });

    const buffer = new Float32Array(16384);
    renderMono(synth, buffer, 16384);

    const peakHz = computeFFTPeak(buffer, 44100, 440.0);
    expect(peakHz).toBeGreaterThanOrEqual(439.0);
    expect(peakHz).toBeLessThanOrEqual(441.0);
  });

  test('T1-PIT-02: Middle C (C4 = 261.63 Hz) FFT Peak', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.noteOn({ midiNote: 60, velocity: 100 });

    const buffer = new Float32Array(16384);
    renderMono(synth, buffer, 16384);

    const peakHz = computeFFTPeak(buffer, 44100, 261.63);
    expect(peakHz).toBeGreaterThanOrEqual(260.6);
    expect(peakHz).toBeLessThanOrEqual(262.6);
  });

  test('T1-PIT-03: Low Register A1 (55.0 Hz) FFT Peak', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.noteOn({ midiNote: 33, velocity: 100 });

    const buffer = new Float32Array(32768);
    renderMono(synth, buffer, 32768);

    const peakHz = computeFFTPeak(buffer, 44100, 55.0);
    expect(peakHz).toBeGreaterThanOrEqual(54.2);
    expect(peakHz).toBeLessThanOrEqual(55.8);
  });

  test('T1-PIT-04: High Register A5 (880.0 Hz) FFT Peak', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.noteOn({ midiNote: 81, velocity: 100 });

    const buffer = new Float32Array(16384);
    renderMono(synth, buffer, 16384);

    const peakHz = computeFFTPeak(buffer, 44100, 880.0);
    expect(peakHz).toBeGreaterThanOrEqual(878.0);
    expect(peakHz).toBeLessThanOrEqual(882.0);
  });

  test('T1-PIT-05: Semitone Pitch Ratio (C4 to C#4)', async () => {
    const synthC = createPianoSynthesizer(44100);
    synthC.noteOn({ midiNote: 60, velocity: 100 });
    const bufC = new Float32Array(16384);
    renderMono(synthC, bufC, 16384);
    const freqC = computeFFTPeak(bufC, 44100, 261.63);

    const synthCSharp = createPianoSynthesizer(44100);
    synthCSharp.noteOn({ midiNote: 61, velocity: 100 });
    const bufCSharp = new Float32Array(16384);
    renderMono(synthCSharp, bufCSharp, 16384);
    const freqCSharp = computeFFTPeak(bufCSharp, 44100, 277.18);

    const ratio = freqCSharp / freqC;
    const expectedRatio = Math.pow(2, 1 / 12); // ~1.059463
    expect(Math.abs(ratio - expectedRatio)).toBeLessThan(0.005);
  });

  test('T1-PIT-06: UI Frequency Display Output', async ({ page }) => {
    await page.goto('./');
    await ensurePianoKeyboard(page);

    await clickPianoKey(page, 60);

    const freqPreview = page.locator('#freq-preview');
    await expect(freqPreview).toContainText('261.63 Hz');
    await expect(freqPreview).toContainText('MIDI 60');
  });
});
