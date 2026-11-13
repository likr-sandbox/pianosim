import { describe, it, expect } from 'vitest';
import { PianoSynthesizer } from '../../src/audio/PianoSynthesizer';
import { PianoVoice } from '../../src/audio/PianoVoice';
import { PianoAudioEngine } from '../../src/audio/PianoAudioEngine';
import {
  checkSignalStability,
  renderSynthToMono,
} from './dsp-utils';

describe('Acoustic Suite: Numerical Stability & Headroom (stability.test.ts)', () => {
  const sampleRate = 44100;

  it('renders 8-note simultaneous chord at max velocity (v=127) with 0 NaN, 0 Inf, peak <= 1.0, and DC < 0.005', () => {
    const synth = new PianoSynthesizer(sampleRate);
    // 8-note full chord across registers: C2, G2, C3, E3, G3, C4, E4, G4
    const chordNotes = [36, 43, 48, 52, 55, 60, 64, 67];
    for (const note of chordNotes) {
      synth.noteOn({ midiNote: note, velocity: 127 });
    }

    // Render 3.0 seconds (132,300 samples)
    const buffer = renderSynthToMono(synth, 3.0, sampleRate);
    const { hasNaN, hasInf, peak, dcOffset, rms } = checkSignalStability(buffer);

    expect(hasNaN).toBe(false);
    expect(hasInf).toBe(false);
    expect(peak).toBeLessThanOrEqual(1.0); // Master soft saturator prevents hard digital clipping
    expect(dcOffset).toBeLessThan(0.005);  // DC blocking filter suppresses DC accumulation
    expect(rms).toBeGreaterThan(0.01);    // Audio is clearly active and resonant
  });

  it('maintains stability during voice-stealing barrage (32 rapid noteOn triggers)', () => {
    const synth = new PianoSynthesizer(sampleRate);
    const totalDuration = 2.0;
    const totalSamples = Math.floor(totalDuration * sampleRate);
    const mono = new Float32Array(totalSamples);
    const blockSize = 128;
    const blockL = new Float32Array(blockSize);
    const blockR = new Float32Array(blockSize);

    let offset = 0;
    let nextNote = 36;

    while (offset < totalSamples) {
      const chunk = Math.min(blockSize, totalSamples - offset);
      // Trigger a new note every ~25ms (approx 1100 samples)
      if (offset % 1100 < blockSize && nextNote <= 72) {
        synth.noteOn({ midiNote: nextNote++, velocity: 110 });
      }

      synth.render(blockL, blockR, chunk);
      for (let i = 0; i < chunk; i++) {
        mono[offset + i] = 0.5 * (blockL[i] + blockR[i]);
      }
      offset += chunk;
    }

    const { hasNaN, hasInf, peak } = checkSignalStability(mono);
    expect(hasNaN).toBe(false);
    expect(hasInf).toBe(false);
    expect(peak).toBeLessThanOrEqual(1.0);
  });

  it('remains stable on boundary registers A0 (MIDI 21) and C8 (MIDI 108)', () => {
    const synth = new PianoSynthesizer(sampleRate);
    synth.noteOn({ midiNote: 21, velocity: 127 }); // Lowest note A0
    synth.noteOn({ midiNote: 108, velocity: 127 }); // Highest note C8

    const buffer = renderSynthToMono(synth, 1.5, sampleRate);
    const { hasNaN, hasInf, peak, dcOffset } = checkSignalStability(buffer);

    expect(hasNaN).toBe(false);
    expect(hasInf).toBe(false);
    expect(peak).toBeLessThanOrEqual(1.0);
    expect(dcOffset).toBeLessThan(0.005);
  });

  it('cleanly flushes denormals/subnormals to zero post-release', () => {
    const synth = new PianoSynthesizer(sampleRate);
    synth.noteOn({ midiNote: 72, velocity: 80 });

    // Release note at 0.3s and render until 3.0s
    const buffer = renderSynthToMono(synth, 3.0, sampleRate, {
      noteOffTimeSec: 0.3,
      midiNote: 72,
    });

    // Examine tail between 2.0s and 3.0s
    const tailStart = Math.floor(2.0 * sampleRate);
    let subnormalFound = false;

    for (let i = tailStart; i < buffer.length; i++) {
      const val = buffer[i];
      // Values below 1e-15 should be flushed to 0.0 by denormal killer
      if (Math.abs(val) > 0 && Math.abs(val) < 1e-15) {
        subnormalFound = true;
      }
    }

    expect(subnormalFound).toBe(false);
  });

  it('guarantees voice stealing jump discontinuity is strictly < 1e-5 (exact 0.000000 jump)', () => {
    // Voice-level test across multiple notes, velocities, and sample phases
    const testCases = [
      { note1: 60, note2: 64, vel: 100, samplesBefore: 800 },
      { note1: 48, note2: 52, vel: 120, samplesBefore: 500 },
      { note1: 72, note2: 76, vel: 80, samplesBefore: 1200 },
      { note1: 36, note2: 40, vel: 110, samplesBefore: 1600 },
      { note1: 84, note2: 88, vel: 95, samplesBefore: 600 },
    ];

    for (const tc of testCases) {
      const voice = new PianoVoice(sampleRate);
      voice.trigger({ midiNote: tc.note1, velocity: tc.vel }, 0);

      for (let i = 0; i < tc.samplesBefore; i++) {
        voice.processSample();
      }

      const sampleBeforeSteal = voice.processSample();
      expect(Math.abs(sampleBeforeSteal)).toBeGreaterThan(0.005); // actively vibrating

      // Steal and retrigger with new note: dcX1 and dcY1 are reset
      voice.stealAndRetrigger({ midiNote: tc.note2, velocity: tc.vel }, tc.samplesBefore + 1);
      const sampleAfterSteal = voice.processSample();

      // Discontinuity step |y[0] - y_last| must be strictly < 1e-5 (exact 0.000000)
      const stepJump = Math.abs(sampleAfterSteal - sampleBeforeSteal);
      expect(stepJump).toBeLessThan(1e-5);
    }
  });

  it('auto-deactivates undamped high treble notes (MIDI 96, 108) to IDLE after decay', () => {
    const synth = new PianoSynthesizer(sampleRate);
    synth.noteOn({ midiNote: 96, velocity: 100 });
    synth.noteOn({ midiNote: 108, velocity: 100 });
    expect(synth.getActiveVoiceCount()).toBe(2);

    const blockSize = 128;
    const blockL = new Float32Array(blockSize);
    const blockR = new Float32Array(blockSize);

    // Hold keys for 0.2s
    const holdSamples = Math.floor(0.2 * sampleRate);
    let rendered = 0;
    while (rendered < holdSamples) {
      synth.render(blockL, blockR, blockSize);
      rendered += blockSize;
    }

    // Release keys
    synth.noteOff({ midiNote: 96 });
    synth.noteOff({ midiNote: 108 });

    // Render 3.5s total duration for notes to complete natural decay
    const totalSamples = Math.floor(3.5 * sampleRate);
    while (rendered < totalSamples) {
      synth.render(blockL, blockR, blockSize);
      rendered += blockSize;
    }

    // High treble voices must auto-deactivate to IDLE once decayed
    expect(synth.getActiveVoiceCount()).toBe(0);
  });

  it('treats MIDI noteOn with velocity 0 as noteOff', () => {
    const synth = new PianoSynthesizer(sampleRate);
    synth.noteOn({ midiNote: 60, velocity: 90 });
    expect(synth.getActiveVoiceCount()).toBe(1);

    const blockSize = 128;
    const blockL = new Float32Array(blockSize);
    const blockR = new Float32Array(blockSize);

    // Render 0.1s
    synth.render(blockL, blockR, blockSize);

    // Send NoteOn with velocity 0 (standard MIDI NoteOff)
    synth.noteOn({ midiNote: 60, velocity: 0 });

    // Render 0.8s for damper release to take effect and deactivate voice
    const releaseSamples = Math.floor(0.8 * sampleRate);
    let rendered = 0;
    while (rendered < releaseSamples) {
      synth.render(blockL, blockR, blockSize);
      rendered += blockSize;
    }

    expect(synth.getActiveVoiceCount()).toBe(0);
  });

  it('serializes concurrent PianoAudioEngine.init() calls into a single shared promise', async () => {
    const engine = new PianoAudioEngine();
    const p1 = engine.init();
    const p2 = engine.init();
    expect(p1).toBe(p2);
    await Promise.all([p1, p2]);
  });
});
