import { describe, it, expect } from 'vitest';
import { PianoSynthesizer } from '../../src/audio/PianoSynthesizer';
import {
  computeDecayRateDbPerSec,
  renderSynthToMono,
} from './dsp-utils';

describe('Acoustic Suite: Damper & Sustain Dynamics (damper.test.ts)', () => {
  const sampleRate = 44100;

  it('demonstrates natural sustained decay (< 25 dB/s) while key is held down', () => {
    const synth = new PianoSynthesizer(sampleRate);
    synth.noteOn({ midiNote: 60, velocity: 90 }); // C4 held for 2.0s

    const buffer = renderSynthToMono(synth, 2.0, sampleRate);
    // Measure decay rate between 0.2s and 1.0s
    const decayRate = computeDecayRateDbPerSec(buffer, sampleRate, 0.2, 1.0);

    expect(decayRate).toBeLessThan(25.0);
    expect(decayRate).toBeGreaterThan(1.0); // Verifies tone is ringing and naturally attenuating
  });

  it('demonstrates rapid damper damping (> 150 dB/s) upon noteOff key release', () => {
    const synth = new PianoSynthesizer(sampleRate);
    synth.noteOn({ midiNote: 60, velocity: 90 }); // C4

    // Key released at 0.5s
    const buffer = renderSynthToMono(synth, 1.0, sampleRate, {
      noteOffTimeSec: 0.5,
      midiNote: 60,
    });

    // Measure decay rate immediately after release between 0.50s and 0.65s (150 ms)
    const releaseDecayRate = computeDecayRateDbPerSec(buffer, sampleRate, 0.5, 0.65);

    expect(releaseDecayRate).toBeGreaterThan(150.0);
  });

  it('verifies damper acceleration factor > 5x faster than sustain decay', () => {
    // 1. Sustain note
    const synthSus = new PianoSynthesizer(sampleRate);
    synthSus.noteOn({ midiNote: 60, velocity: 90 });
    const bufSus = renderSynthToMono(synthSus, 1.5, sampleRate);
    const sustainRate = Math.max(
      computeDecayRateDbPerSec(bufSus, sampleRate, 0.2, 1.0),
      0.1
    );

    // 2. Released note
    const synthRel = new PianoSynthesizer(sampleRate);
    synthRel.noteOn({ midiNote: 60, velocity: 90 });
    const bufRel = renderSynthToMono(synthRel, 1.0, sampleRate, {
      noteOffTimeSec: 0.5,
      midiNote: 60,
    });
    const releaseRate = computeDecayRateDbPerSec(bufRel, sampleRate, 0.5, 0.65);

    const accelerationFactor = releaseRate / sustainRate;
    expect(accelerationFactor).toBeGreaterThan(5.0);
  });

  it('allows sustain pedal to hold notes ringing even after noteOff', () => {
    const synth = new PianoSynthesizer(sampleRate);
    synth.setSustainPedal(true);
    synth.noteOn({ midiNote: 60, velocity: 90 });

    // Note released at 0.5s while pedal remains depressed
    const buffer = renderSynthToMono(synth, 1.5, sampleRate, {
      noteOffTimeSec: 0.5,
      midiNote: 60,
    });

    // Measure decay rate between 0.5s and 1.0s
    const pedalDecayRate = computeDecayRateDbPerSec(buffer, sampleRate, 0.5, 1.0);

    // While pedal is down, note must not drop with rapid damper rate
    expect(pedalDecayRate).toBeLessThan(25.0);
  });
});
