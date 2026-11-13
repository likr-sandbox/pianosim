import { test, expect } from '@playwright/test';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import { computeRMS, verifySampleBounds } from '../helpers/audio-analysis';

test.describe('Tier 1: Feature Coverage - Synthesis Engine', () => {
  test('T1-SYN-01: Offline Buffer Generation & Format', async () => {
    const synth = createPianoSynthesizer(44100);
    const sizes = [128, 512, 44100];

    for (const size of sizes) {
      const left = new Float32Array(size);
      const right = new Float32Array(size);

      synth.render(left, right, size);

      expect(left.length).toBe(size);
      expect(right.length).toBe(size);

      const boundsLeft = verifySampleBounds(left);
      const boundsRight = verifySampleBounds(right);

      expect(boundsLeft.hasNaN).toBe(false);
      expect(boundsLeft.hasInf).toBe(false);
      expect(boundsRight.hasNaN).toBe(false);
      expect(boundsRight.hasInf).toBe(false);
    }
  });

  test('T1-SYN-02: Silence Prior to Note Trigger', async () => {
    const synth = createPianoSynthesizer(44100);
    const left = new Float32Array(4096);
    const right = new Float32Array(4096);

    synth.render(left, right, 4096);

    const rmsLeft = computeRMS(left);
    const rmsRight = computeRMS(right);

    expect(rmsLeft).toBeLessThan(1e-7);
    expect(rmsRight).toBeLessThan(1e-7);
  });

  test('T1-SYN-03: Audio Energy Initiation on NoteOn', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.noteOn({ midiNote: 60, velocity: 100 });

    const left = new Float32Array(2048);
    const right = new Float32Array(2048);
    synth.render(left, right, 2048);

    // Initial 10ms (441 samples) should contain active energy
    const initialRMS = computeRMS(left, 0, 441);
    expect(initialRMS).toBeGreaterThan(0.005);
  });

  test('T1-SYN-04: Finite Bounded Output (Anti-Explosion)', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.noteOn({ midiNote: 69, velocity: 127 });

    const left = new Float32Array(44100);
    const right = new Float32Array(44100);
    synth.render(left, right, 44100);

    const bounds = verifySampleBounds(left);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
    expect(bounds.absMax).toBeLessThanOrEqual(1.0);
    expect(bounds.zeroClipping).toBe(true);
  });

  test('T1-SYN-05: Standard Sample Rate Support', async () => {
    const synth44 = createPianoSynthesizer(44100);
    const synth48 = createPianoSynthesizer(48000);

    synth44.noteOn({ midiNote: 69, velocity: 90 });
    synth48.noteOn({ midiNote: 69, velocity: 90 });

    const buf44 = new Float32Array(4096);
    const buf48 = new Float32Array(4096);
    renderMono(synth44, buf44, 4096);
    renderMono(synth48, buf48, 4096);

    expect(computeRMS(buf44)).toBeGreaterThan(0.01);
    expect(computeRMS(buf48)).toBeGreaterThan(0.01);

    const bounds44 = verifySampleBounds(buf44);
    const bounds48 = verifySampleBounds(buf48);
    expect(bounds44.hasNaN).toBe(false);
    expect(bounds48.hasNaN).toBe(false);
  });

  test('T1-SYN-06: AllNotesOff Immediate Voice Cut', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.noteOn({ midiNote: 60, velocity: 100 });
    synth.noteOn({ midiNote: 64, velocity: 100 });
    synth.noteOn({ midiNote: 67, velocity: 100 });
    synth.noteOn({ midiNote: 72, velocity: 100 });

    // Render active audio
    const activeBuf = new Float32Array(2048);
    renderMono(synth, activeBuf, 2048);
    expect(computeRMS(activeBuf)).toBeGreaterThan(0.01);

    // Call allNotesOff
    synth.allNotesOff();

    const postBuf = new Float32Array(4410); // 100ms
    renderMono(synth, postBuf, 4410);
    const postRMS = computeRMS(postBuf, 2000, 4410);
    expect(postRMS).toBeLessThan(1e-5);
  });
});
