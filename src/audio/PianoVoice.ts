/**
 * PianoVoice.ts
 * Polyphonic voice unit simulating acoustic piano multi-string unisons,
 * hammer excitation, damper dynamics, bridge coupling, and voice stealing crossfade.
 */

import { WaveguideString } from './WaveguideString';
import { HammerModel } from './HammerModel';

export enum VoiceState {
  IDLE = 0,
  ATTACK = 1,
  SUSTAIN = 2,
  RELEASE = 3,
  FADING_OUT = 4,
}

export interface VoiceTriggerOptions {
  midiNote: number;
  velocity: number;
  sampleOffset?: number;
}

export class PianoVoice {
  private readonly sampleRate: number;
  private state: VoiceState = VoiceState.IDLE;

  // Unison strings: 1 for bass, 2 for tenor, 3 for treble
  private readonly strings: WaveguideString[] = [];
  private numStrings: number = 1;
  private readonly couplingCoeff: number = 0.02;

  // Hammer excitation model
  private readonly hammer: HammerModel;

  // Note parameters
  private midiNote: number = 0;
  private f0: number = 440.0;
  private velocity: number = 0;
  private keyHeld: boolean = false;
  private damperDown: boolean = false;
  private hasDamper: boolean = true;
  private startTimeSamples: number = 0;
  private releaseSamplesCount: number = 0;

  // Amplitude and state tracking
  private envelope: number = 0.0;
  private lastOutput: number = 0.0;

  // Anti-click voice stealing crossfade (3ms linear ramp)
  private stolenResidual: number = 0.0;
  private fadeSamplesRemaining: number = 0;
  private readonly fadeLengthSamples: number;

  // DC blocking filter state (y[n] = x[n] - x[n-1] + 0.995 y[n-1])
  private dcX1: number = 0.0;
  private dcY1: number = 0.0;
  private readonly dcR: number = 0.995;

  // Sample offset for intra-quantum scheduling
  private sampleOffsetRemaining: number = 0;

  constructor(sampleRate: number) {
    this.sampleRate = sampleRate;
    this.fadeLengthSamples = Math.max(1, Math.round(0.003 * sampleRate)); // ~3ms (132 samples @ 44.1kHz)
    this.hammer = new HammerModel(sampleRate);

    // Pre-allocate maximum 3 unison waveguide strings per voice
    for (let i = 0; i < 3; i++) {
      this.strings.push(new WaveguideString(sampleRate));
    }
  }

  public trigger(options: VoiceTriggerOptions, currentSampleClock: number): void {
    const { midiNote, velocity, sampleOffset = 0 } = options;
    this.midiNote = midiNote;
    this.velocity = velocity;
    this.keyHeld = true;
    this.damperDown = false;
    this.startTimeSamples = currentSampleClock;
    this.sampleOffsetRemaining = Math.max(0, sampleOffset);
    this.releaseSamplesCount = 0;

    // Acoustic grand pianos lack dampers above ~F#6 (MIDI 89)
    this.hasDamper = midiNote < 89;

    // Fundamental frequency: f0 = 440 * 2^((m - 69) / 12)
    this.f0 = 440.0 * Math.pow(2.0, (midiNote - 69.0) / 12.0);

    // Register-based unison string allocation & detuning
    if (midiNote < 40) {
      // Bass: 1 string (monochord)
      this.numStrings = 1;
      this.strings[0].setFrequency(this.f0, midiNote);
    } else if (midiNote < 60) {
      // Tenor: 2 strings (bichord), detuned +/- 0.075 Hz (span 0.15 Hz)
      this.numStrings = 2;
      const detune = 0.15;
      this.strings[0].setFrequency(this.f0 - detune * 0.5, midiNote);
      this.strings[1].setFrequency(this.f0 + detune * 0.5, midiNote);
    } else {
      // Treble: 3 strings (trichord), detuned -0.20, 0, +0.20 Hz
      this.numStrings = 3;
      const detune = 0.20;
      this.strings[0].setFrequency(this.f0 - detune, midiNote);
      this.strings[1].setFrequency(this.f0, midiNote);
      this.strings[2].setFrequency(this.f0 + detune, midiNote);
    }

    // Set loop damping to sustained mode and reset string delay states
    for (let i = 0; i < this.numStrings; i++) {
      this.strings[i].setDamper(false, midiNote, this.f0);
      this.strings[i].reset();
    }

    // Trigger non-linear hammer excitation pulse
    this.hammer.strike(this.f0, velocity);

    this.state = VoiceState.ATTACK;
  }

  public stealAndRetrigger(options: VoiceTriggerOptions, currentSampleClock: number): void {
    // 3ms linear anti-click crossfade: capture current output sample
    this.stolenResidual = this.lastOutput;
    this.fadeSamplesRemaining = this.fadeLengthSamples;

    // Reset DC blocking filter memory to prevent double-counting residual
    this.dcX1 = 0.0;
    this.dcY1 = 0.0;

    // Retrigger with new note parameters
    this.trigger(options, currentSampleClock);
  }

  public noteOff(_sampleOffset: number = 0, sustainPedalDown: boolean = false): void {
    this.keyHeld = false;
    if (!sustainPedalDown) {
      this.applyDamper();
    }
  }

  public onSustainPedalChange(pedalDown: boolean): void {
    if (!pedalDown && !this.keyHeld && (this.state === VoiceState.SUSTAIN || this.state === VoiceState.ATTACK)) {
      this.applyDamper();
    }
  }

  private applyDamper(): void {
    if (!this.hasDamper) {
      // In undamped high treble register (MIDI >= 89), there are no dampers physically,
      // but releasing the key transitions the voice to RELEASE state so it can auto-deactivate
      // once natural sustain decay finishes, without altering string loop damping.
      this.state = VoiceState.RELEASE;
      this.releaseSamplesCount = 0;
      return;
    }
    this.damperDown = true;
    this.state = VoiceState.RELEASE;
    this.releaseSamplesCount = 0;
    for (let i = 0; i < this.numStrings; i++) {
      this.strings[i].setDamper(true, this.midiNote, this.f0);
    }
  }

  public processSample(): number {
    if (this.state === VoiceState.IDLE) {
      return 0.0;
    }

    // Intra-quantum sample offset delay handling
    if (this.sampleOffsetRemaining > 0) {
      this.sampleOffsetRemaining--;
      if (this.fadeSamplesRemaining > 0) {
        const fadeGain = this.fadeSamplesRemaining / this.fadeLengthSamples;
        this.fadeSamplesRemaining--;
        return this.stolenResidual * fadeGain;
      }
      return 0.0;
    }

    // 1. Generate hammer excitation force
    let hammerForce = 0.0;
    if (this.state === VoiceState.ATTACK) {
      hammerForce = this.hammer.processSample();
      if (!this.hammer.isActive()) {
        this.state = VoiceState.SUSTAIN;
      }
    }

    // 2. Process unison waveguide strings with Weinreich bridge reflection coupling
    let stringSum = 0.0;
    if (this.numStrings === 1) {
      stringSum = this.strings[0].process(hammerForce);
    } else if (this.numStrings === 2) {
      const s0 = this.strings[0].process(hammerForce);
      const s1 = this.strings[1].process(hammerForce);
      const c = this.couplingCoeff;
      const c0 = (1.0 - c) * s0 + c * s1;
      const c1 = c * s0 + (1.0 - c) * s1;
      this.strings[0].injectCoupling(c0 - s0);
      this.strings[1].injectCoupling(c1 - s1);
      stringSum = (s0 + s1) * 0.70710678; // 1 / sqrt(2)
    } else {
      const s0 = this.strings[0].process(hammerForce);
      const s1 = this.strings[1].process(hammerForce);
      const s2 = this.strings[2].process(hammerForce);
      const c = this.couplingCoeff;
      const c0 = (1.0 - 2.0 * c) * s0 + c * (s1 + s2);
      const c1 = (1.0 - 2.0 * c) * s1 + c * (s0 + s2);
      const c2 = (1.0 - 2.0 * c) * s2 + c * (s0 + s1);
      this.strings[0].injectCoupling(c0 - s0);
      this.strings[1].injectCoupling(c1 - s1);
      this.strings[2].injectCoupling(c2 - s2);
      stringSum = (s0 + s1 + s2) * 0.57735027; // 1 / sqrt(3)
    }

    // 3. DC Blocker filter: y[n] = x[n] - x[n-1] + R * y[n-1]
    const dcOut = stringSum - this.dcX1 + this.dcR * this.dcY1;
    this.dcX1 = stringSum;
    this.dcY1 = dcOut;

    // 4. Anti-click voice stealing crossfade addition
    let finalSample = dcOut;
    if (this.fadeSamplesRemaining > 0) {
      const fadeFactor = this.fadeSamplesRemaining / this.fadeLengthSamples;
      finalSample += this.stolenResidual * fadeFactor;
      this.fadeSamplesRemaining--;
    }

    // 5. Subnormal / denormal protection
    if (Math.abs(finalSample) < 1e-15) {
      finalSample = 0.0;
    }

    // 6. Envelope tracking for voice stealing & silence detection
    const absSample = Math.abs(finalSample);
    this.envelope = this.envelope * 0.999 + absSample * 0.001;

    // 7. Auto-deactivation when note has decayed completely
    if (this.state === VoiceState.RELEASE) {
      this.releaseSamplesCount++;
      // Deactivate if envelope dropped below threshold or safety timeout reached
      const envThreshold = this.hasDamper ? 1e-6 : 1e-4;
      const timeoutSamples = this.hasDamper
        ? Math.floor(0.8 * this.sampleRate)
        : Math.floor(3.0 * this.sampleRate);
      if (this.envelope < envThreshold || this.releaseSamplesCount > timeoutSamples) {
        this.state = VoiceState.IDLE;
        this.reset();
        return 0.0;
      }
    } else if (!this.keyHeld && this.envelope < 1e-4) {
      this.state = VoiceState.IDLE;
      this.reset();
      return 0.0;
    }

    this.lastOutput = finalSample;
    return finalSample;
  }

  public reset(): void {
    this.state = VoiceState.IDLE;
    this.envelope = 0.0;
    this.lastOutput = 0.0;
    this.dcX1 = 0.0;
    this.dcY1 = 0.0;
    this.stolenResidual = 0.0;
    this.fadeSamplesRemaining = 0;
    this.releaseSamplesCount = 0;
    this.hammer.reset();
    for (let i = 0; i < 3; i++) {
      this.strings[i].reset();
    }
  }

  // Getters for voice management
  public getState(): VoiceState { return this.state; }
  public isActive(): boolean { return this.state !== VoiceState.IDLE; }
  public getMidiNote(): number { return this.midiNote; }
  public getVelocity(): number { return this.velocity; }
  public isDamperDown(): boolean { return this.damperDown; }
  public getEnvelope(): number { return this.envelope; }
  public getStartTime(): number { return this.startTimeSamples; }
}
