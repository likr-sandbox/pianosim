import { test, expect } from '@playwright/test';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import { computeRMS, computeDynamicRangeDB } from '../helpers/audio-analysis';
import { ensurePianoKeyboard } from '../helpers/dom-driver';

test.describe('Tier 3: Combination - Chords & Sustain Dynamics', () => {
  test('T3-CMB-01: 4-Note Chord + Sustain Pedal Release', async () => {
    const synth = createPianoSynthesizer(44100);

    // Engage pedal
    synth.setSustainPedal(true);

    // Strike C major 7th chord: C4 (60), E4 (64), G4 (67), B4 (71)
    const chord = [60, 64, 67, 71];
    for (const note of chord) {
      synth.noteOn({ midiNote: note, velocity: 90 });
    }

    // Render initial chord strike (50ms = 2205 samples)
    const attackBuf = new Float32Array(2205);
    renderMono(synth, attackBuf, 2205);

    // Release all physical keys while pedal holds
    for (const note of chord) {
      synth.noteOff({ midiNote: note });
    }

    // Render under sustain
    const sustainBuf = new Float32Array(22050);
    renderMono(synth, sustainBuf, 22050);
    const sustainRMS = computeRMS(sustainBuf, 11025, 22050);
    expect(sustainRMS).toBeGreaterThan(0.01);

    // Release pedal
    synth.setSustainPedal(false);

    // Render 150ms after pedal release
    const dampedBuf = new Float32Array(6615);
    renderMono(synth, dampedBuf, 6615);
    const dampedRMS = computeRMS(dampedBuf, 4410, 6615);

    const dbDrop = computeDynamicRangeDB(sustainRMS, dampedRMS);
    expect(dbDrop).toBeGreaterThan(20.0);
  });

  test('T3-CMB-07: Rapid Pedal Pumping During Arpeggio', async () => {
    const synth = createPianoSynthesizer(44100);
    const notes = [60, 64, 67, 72]; // C-E-G-C

    for (let cycle = 0; cycle < 3; cycle++) {
      // Pedal ON
      synth.setSustainPedal(true);
      for (const note of notes) {
        synth.noteOn({ midiNote: note, velocity: 85 });
      }
      const sustained = new Float32Array(11025); // 250ms
      renderMono(synth, sustained, 11025);
      expect(computeRMS(sustained)).toBeGreaterThan(0.01);

      // Pedal OFF
      synth.setSustainPedal(false);
      for (const note of notes) {
        synth.noteOff({ midiNote: note });
      }
      const released = new Float32Array(11025); // 250ms
      renderMono(synth, released, 11025);
    }
  });

  test('T3-CMB-08: Panic AllNotesOff Under Polyphonic Sustain', async ({ page }) => {
    await page.goto('./');
    await ensurePianoKeyboard(page);

    const synth = createPianoSynthesizer(44100);
    synth.setSustainPedal(true);

    // 8 notes sounding
    for (let note = 60; note < 68; note++) {
      synth.noteOn({ midiNote: note, velocity: 90 });
    }

    const preBuf = new Float32Array(4410);
    renderMono(synth, preBuf, 4410);
    expect(computeRMS(preBuf)).toBeGreaterThan(0.01);

    // Panic cut
    synth.allNotesOff();
    synth.setSustainPedal(false);

    const postBuf = new Float32Array(3528); // 80ms
    renderMono(synth, postBuf, 3528);
    const postRMS = computeRMS(postBuf, 1764, 3528);
    expect(postRMS).toBeLessThan(1e-5);
  });
});
