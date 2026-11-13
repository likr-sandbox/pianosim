import { test, expect } from '@playwright/test';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import { verifySampleBounds, computeRMS } from '../helpers/audio-analysis';

test.describe('Tier 4: Application Scenario - J.S. Bach Prelude in C Major (BWV 846)', () => {
  test('T4-APP-01: Bach BWV 846 Arpeggios with Continuous Sustain Resonance', async () => {
    const synth = createPianoSynthesizer(44100);

    // Well-Tempered Clavier Prelude in C Major opening motif:
    // C4 (60), E4 (64), G4 (67), C5 (72), E5 (76), G4 (67), C5 (72), E5 (76) (repeated twice)
    const arpeggio = [60, 64, 67, 72, 76, 67, 72, 76];
    const fullSequence = [...arpeggio, ...arpeggio];

    // Sustain pedal engaged throughout the bar
    synth.setSustainPedal(true);

    const noteDurationSamples = Math.round(44100 * 0.12); // 120ms per sixteenth note
    const totalSamples = noteDurationSamples * fullSequence.length + 44100; // sequence + 1s ring
    const masterOutput = new Float32Array(totalSamples);

    let offset = 0;
    for (const note of fullSequence) {
      synth.noteOn({ midiNote: note, velocity: 75 });
      const slice = masterOutput.subarray(offset, offset + noteDurationSamples);
      renderMono(synth, slice, noteDurationSamples);
      // Key release (noteOff), but pedal sustains resonance
      synth.noteOff({ midiNote: note });
      offset += noteDurationSamples;
    }

    // Render remaining ringing decay
    const remaining = totalSamples - offset;
    const ringSlice = masterOutput.subarray(offset, offset + remaining);
    renderMono(synth, ringSlice, remaining);

    // Assertions
    const bounds = verifySampleBounds(masterOutput);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
    expect(bounds.absMax).toBeLessThanOrEqual(1.0);
    expect(bounds.zeroClipping).toBe(true);

    // Soundboard resonance accumulation: RMS during final arpeggio should be sustained
    const finalArpeggioRMS = computeRMS(
      masterOutput,
      noteDurationSamples * 12,
      noteDurationSamples * 16
    );
    expect(finalArpeggioRMS).toBeGreaterThan(0.01);
  });
});
