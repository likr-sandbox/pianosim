import { test, expect } from '@playwright/test';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import { computeRMS, computeDecayRateDBPerSec, computeDynamicRangeDB } from '../helpers/audio-analysis';

test.describe('Tier 1: Feature Coverage - Damping & Release Dynamics', () => {
  test('T1-DMP-01: Natural Sustain Decay Slope', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.noteOn({ midiNote: 60, velocity: 100 });

    // Render 2 seconds without noteOff
    const buffer = new Float32Array(88200);
    renderMono(synth, buffer, 88200);

    // Measure RMS in [0.2s, 0.5s] and [1.5s, 1.8s]
    const rmsEarly = computeRMS(buffer, 8820, 22050);
    const rmsLate = computeRMS(buffer, 66150, 79380);

    const decayRate = computeDecayRateDBPerSec(rmsEarly, rmsLate, 1.3);
    expect(decayRate).toBeLessThan(18.0);
    expect(decayRate).toBeGreaterThan(4.0);
  });

  test('T1-DMP-02: Rapid Damper Drop on NoteOff', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.noteOn({ midiNote: 60, velocity: 100 });

    // Ring for 0.5s (22050 samples)
    const held = new Float32Array(22050);
    renderMono(synth, held, 22050);
    const preReleaseRMS = computeRMS(held, 17640, 22050);

    // Release note
    synth.noteOff({ midiNote: 60 });

    // Render 150ms after noteOff (6615 samples)
    const releaseBuf = new Float32Array(6615);
    renderMono(synth, releaseBuf, 6615);
    const postReleaseRMS = computeRMS(releaseBuf, 4410, 6615);

    const dbDrop = computeDynamicRangeDB(preReleaseRMS, postReleaseRMS);
    expect(dbDrop).toBeGreaterThan(20.0);
  });

  test('T1-DMP-03: Damper vs Sustain Decay Slope Ratio', async () => {
    // Sustain decay
    const synthSustain = createPianoSynthesizer(44100);
    synthSustain.noteOn({ midiNote: 60, velocity: 100 });
    const bufSustain = new Float32Array(44100);
    renderMono(synthSustain, bufSustain, 44100);
    const rEarly = computeRMS(bufSustain, 8820, 17640);
    const rLate = computeRMS(bufSustain, 30870, 39690);
    const rateSustain = Math.max(0.5, computeDecayRateDBPerSec(rEarly, rLate, 0.5));

    // Damper release decay
    const synthDamper = createPianoSynthesizer(44100);
    synthDamper.noteOn({ midiNote: 60, velocity: 100 });
    const attack = new Float32Array(22050);
    renderMono(synthDamper, attack, 22050);
    const rBefore = computeRMS(attack, 17640, 22050);

    synthDamper.noteOff({ midiNote: 60 });
    const release = new Float32Array(4410); // 100ms
    renderMono(synthDamper, release, 4410);
    const rAfter = computeRMS(release, 2205, 4410);
    const rateDamper = computeDecayRateDBPerSec(rBefore, rAfter, 0.1);

    const ratio = rateDamper / rateSustain;
    expect(ratio).toBeGreaterThanOrEqual(10.0);
  });

  test('T1-DMP-04: High Register Undamped Ringing', async () => {
    // C7 (MIDI 96) has no physical damper in pianos: releasing the key must not change the
    // sound relative to the same note left sounding, whereas a damped note (C4) is silenced.
    const ringWindow = (midi: number, release: boolean): number => {
      const synth = createPianoSynthesizer(44100);
      synth.noteOn({ midiNote: midi, velocity: 100 });
      const attack = new Float32Array(8820); // 200 ms
      renderMono(synth, attack, 8820);
      if (release) synth.noteOff({ midiNote: midi });
      const after = new Float32Array(11025); // next 250 ms
      renderMono(synth, after, 11025);
      return computeRMS(after);
    };

    const c7Released = ringWindow(96, true);
    const c7Held = ringWindow(96, false);
    expect(c7Held).toBeGreaterThan(0);
    // Undamped: released within 6 dB of held
    expect(c7Released / c7Held).toBeGreaterThan(0.5);

    // Control: damped C4 is quenched by at least 10 dB when released
    const c4Released = ringWindow(60, true);
    const c4Held = ringWindow(60, false);
    expect(c4Released / c4Held).toBeLessThan(0.316);
  });

  test('T1-DMP-05: Anti-Click Smooth Release Envelope', async () => {
    const synth = createPianoSynthesizer(44100);
    synth.noteOn({ midiNote: 60, velocity: 120 });

    // Render 200ms
    const pre = new Float32Array(8820);
    renderMono(synth, pre, 8820);

    // Release note
    synth.noteOff({ midiNote: 60 });

    const release = new Float32Array(1024);
    renderMono(synth, release, 1024);

    // Assert no discontinuous jump > 0.2 between consecutive samples
    let maxDelta = 0;
    for (let i = 1; i < release.length; i++) {
      const delta = Math.abs(release[i] - release[i - 1]);
      if (delta > maxDelta) maxDelta = delta;
    }

    expect(maxDelta).toBeLessThan(0.2);
  });
});
