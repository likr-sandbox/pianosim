/**
 * PianoSynthesizer.ts
 * Pure TypeScript physical simulation synthesizer engine.
 * Fully decoupled from browser audio hardware, deterministic, and zero-allocation in render loop.
 */

import { PianoVoice, VoiceState } from './PianoVoice';
import { SoundboardResonance } from './SoundboardResonance';

export interface NoteOnOptions {
  midiNote: number;      // 21 (A0) to 108 (C8)
  velocity: number;      // 0 to 127
  sampleOffset?: number; // sample-accurate intra-quantum timing
}

export interface NoteOffOptions {
  midiNote: number;
  sampleOffset?: number;
}

export class PianoSynthesizer {
  private readonly sampleRate: number;
  private readonly MAX_VOICES = 24;
  private readonly voices: PianoVoice[] = [];
  private readonly soundboard: SoundboardResonance;

  private sustainPedalDown: boolean = false;
  private currentSampleClock: number = 0;
  private readonly masterGain: number = 0.45;

  constructor(sampleRate: number) {
    this.sampleRate = sampleRate;
    this.soundboard = new SoundboardResonance(sampleRate);

    // Pre-allocate fixed pool of 24 polyphonic voices
    for (let i = 0; i < this.MAX_VOICES; i++) {
      this.voices.push(new PianoVoice(sampleRate));
    }
  }

  public noteOn(options: NoteOnOptions): void {
    const { midiNote, velocity, sampleOffset } = options;
    if (midiNote < 21 || midiNote > 108) {
      return;
    }

    // Standard MIDI specification: NoteOn with velocity 0 indicates NoteOff
    if (velocity === 0) {
      this.noteOff({ midiNote, sampleOffset });
      return;
    }

    if (velocity < 0) {
      return;
    }

    const voice = this.allocateVoice(midiNote);
    if (voice.isActive()) {
      // 3ms anti-click voice stealing crossfade
      voice.stealAndRetrigger(options, this.currentSampleClock);
    } else {
      voice.trigger(options, this.currentSampleClock);
    }
  }

  public noteOff(options: NoteOffOptions): void {
    const { midiNote, sampleOffset = 0 } = options;
    for (let i = 0; i < this.MAX_VOICES; i++) {
      const v = this.voices[i];
      if (v.isActive() && v.getMidiNote() === midiNote) {
        v.noteOff(sampleOffset, this.sustainPedalDown);
      }
    }
  }

  public setSustainPedal(down: boolean): void {
    this.sustainPedalDown = down;
    for (let i = 0; i < this.MAX_VOICES; i++) {
      this.voices[i].onSustainPedalChange(down);
    }
  }

  public allNotesOff(): void {
    this.sustainPedalDown = false;
    for (let i = 0; i < this.MAX_VOICES; i++) {
      this.voices[i].reset();
    }
    this.soundboard.reset();
  }

  /**
   * High-performance audio render loop. Guaranteed zero GC allocations.
   * Renders into caller-provided Float32Array left and right channel buffers.
   */
  public render(outputLeft: Float32Array, outputRight: Float32Array, numSamples: number): void {
    // 1. Zero out destination buffers
    outputLeft.fill(0.0, 0, numSamples);
    outputRight.fill(0.0, 0, numSamples);

    // 2. Render and accumulate each active voice with stereo spatial panning
    for (let v = 0; v < this.MAX_VOICES; v++) {
      const voice = this.voices[v];
      if (!voice.isActive()) continue;

      const midi = voice.getMidiNote();
      // Keyboard stereo panning: bass left (-0.6), treble right (+0.6)
      const pan = -0.6 + 1.2 * ((midi - 21.0) / 87.0);
      const panL = Math.cos((Math.PI / 4.0) * (1.0 + pan));
      const panR = Math.sin((Math.PI / 4.0) * (1.0 + pan));

      for (let i = 0; i < numSamples; i++) {
        const sample = voice.processSample();
        outputLeft[i] += sample * panL;
        outputRight[i] += sample * panR;
      }
    }

    // 3. Process soundboard modal sympathetic resonance
    for (let i = 0; i < numSamples; i++) {
      const monoSum = (outputLeft[i] + outputRight[i]) * 0.5;
      const res = this.soundboard.processResonanceOnly(monoSum);
      outputLeft[i] += res * 0.35;
      outputRight[i] += res * 0.35;
    }

    // 4. Master Soft Limiter: Math.tanh saturation guarantees output strictly in [-1.0, 1.0]
    for (let i = 0; i < numSamples; i++) {
      let l = Math.tanh(outputLeft[i] * this.masterGain);
      let r = Math.tanh(outputRight[i] * this.masterGain);
      if (Math.abs(l) < 1e-14) l = 0.0;
      if (Math.abs(r) < 1e-14) r = 0.0;
      outputLeft[i] = l;
      outputRight[i] = r;
    }

    this.currentSampleClock += numSamples;
  }

  public getActiveVoiceCount(): number {
    let count = 0;
    for (let i = 0; i < this.MAX_VOICES; i++) {
      if (this.voices[i].isActive()) count++;
    }
    return count;
  }

  public getSampleRate(): number {
    return this.sampleRate;
  }

  /**
   * Intelligent Voice Allocation & Stealing:
   * Priority: Re-strike -> Idle -> Lowest Envelope Released -> Oldest/Quietest Active
   */
  private allocateVoice(midiNote: number): PianoVoice {
    // 1. Re-strike: if same note is already sounding, retrigger existing voice
    for (let i = 0; i < this.MAX_VOICES; i++) {
      const v = this.voices[i];
      if (v.isActive() && v.getMidiNote() === midiNote) {
        return v;
      }
    }

    // 2. IDLE voice search: allocate first unused voice
    for (let i = 0; i < this.MAX_VOICES; i++) {
      const v = this.voices[i];
      if (v.getState() === VoiceState.IDLE) {
        return v;
      }
    }

    // 3. Steal from voices in RELEASE state (pick lowest envelope)
    let bestReleaseVoice: PianoVoice | null = null;
    let minReleaseEnv = Infinity;
    for (let i = 0; i < this.MAX_VOICES; i++) {
      const v = this.voices[i];
      if (v.getState() === VoiceState.RELEASE) {
        const env = v.getEnvelope();
        if (env < minReleaseEnv) {
          minReleaseEnv = env;
          bestReleaseVoice = v;
        }
      }
    }
    if (bestReleaseVoice !== null) {
      return bestReleaseVoice;
    }

    // 4. All 24 voices in ATTACK/SUSTAIN: steal oldest active voice with lowest envelope
    let bestActiveVoice = this.voices[0];
    let minScore = Infinity;
    for (let i = 0; i < this.MAX_VOICES; i++) {
      const v = this.voices[i];
      const ageSeconds = (this.currentSampleClock - v.getStartTime()) / this.sampleRate;
      // Older notes have higher age, yielding lower score (prefer to steal)
      const score = v.getEnvelope() / (1.0 + ageSeconds);
      if (score < minScore) {
        minScore = score;
        bestActiveVoice = v;
      }
    }
    return bestActiveVoice;
  }
}
