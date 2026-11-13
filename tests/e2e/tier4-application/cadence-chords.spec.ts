import { test, expect } from '@playwright/test';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import { verifySampleBounds, computeRMS } from '../helpers/audio-analysis';

test.describe('Tier 4: Application Scenario - Cadence Chords', () => {
  test('T4-APP-03: Expressive ii-V-I Cadential Progression with Dynamic Swell', async () => {
    const synth = createPianoSynthesizer(44100);

    // Chords in C Major:
    // ii: Dm7 = D3 (50), F3 (53), A3 (57), C4 (60) - piano (p, vel=50)
    // V:  G7  = G2 (43), D3 (50), F3 (53), B3 (59) - mezzo-forte (mf, vel=85)
    // I:  Cmaj7 = C2 (36), G2 (43), E3 (52), B3 (59) - forte (f, vel=115)
    const dm7 = [50, 53, 57, 60];
    const g7 = [43, 50, 53, 59];
    const cmaj7 = [36, 43, 52, 59];

    // Pedal held for continuous resonance
    synth.setSustainPedal(true);

    const chordDuration = Math.round(44100 * 0.8); // 800ms per chord
    const rmsProgression: number[] = [];

    // Chord 1: Dm7 (piano)
    for (const m of dm7) synth.noteOn({ midiNote: m, velocity: 50 });
    const bufDm7 = new Float32Array(chordDuration);
    renderMono(synth, bufDm7, chordDuration);
    for (const m of dm7) synth.noteOff({ midiNote: m });
    rmsProgression.push(computeRMS(bufDm7, 0, Math.round(chordDuration * 0.5)));

    // Chord 2: G7 (mezzo-forte)
    for (const m of g7) synth.noteOn({ midiNote: m, velocity: 85 });
    const bufG7 = new Float32Array(chordDuration);
    renderMono(synth, bufG7, chordDuration);
    for (const m of g7) synth.noteOff({ midiNote: m });
    rmsProgression.push(computeRMS(bufG7, 0, Math.round(chordDuration * 0.5)));

    // Chord 3: Cmaj7 (forte)
    for (const m of cmaj7) synth.noteOn({ midiNote: m, velocity: 115 });
    const bufCmaj7 = new Float32Array(chordDuration * 3); // 2.4s long ring
    renderMono(synth, bufCmaj7, chordDuration * 3);
    for (const m of cmaj7) synth.noteOff({ midiNote: m });
    rmsProgression.push(computeRMS(bufCmaj7, 0, Math.round(chordDuration * 0.5)));

    // Assert dynamic swell: p < mf < f
    expect(rmsProgression[1]).toBeGreaterThan(rmsProgression[0]);
    expect(rmsProgression[2]).toBeGreaterThan(rmsProgression[1]);

    const bounds = verifySampleBounds(bufCmaj7);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
    expect(bounds.absMax).toBeLessThanOrEqual(1.0);
    expect(bounds.zeroClipping).toBe(true);
  });
});
