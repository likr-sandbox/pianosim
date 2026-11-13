/**
 * piano-worklet-processor.ts
 * Real-time Web Audio AudioWorkletProcessor script and inline Blob URL loader.
 * Completely self-contained to eliminate MIME-type and asset path routing issues across
 * Vite dev, preview, and GitHub Pages deployments.
 */

export type WorkletInboundMessage =
  | { type: 'noteOn'; midiNote: number; velocity: number; sampleOffset?: number }
  | { type: 'noteOff'; midiNote: number; sampleOffset?: number }
  | { type: 'setSustainPedal'; down: boolean; sampleOffset?: number }
  | { type: 'allNotesOff' };

export type WorkletOutboundMessage =
  | { type: 'ready'; sampleRate: number }
  | { type: 'voiceCount'; count: number };

/**
 * Self-contained AudioWorkletProcessor script string.
 * Embedded DSP core ensures 100% offline and in-memory execution in AudioWorkletGlobalScope.
 */
export const PIANO_WORKLET_PROCESSOR_CODE = `
class WaveguideString {
  constructor(sampleRate) {
    this.sampleRate = sampleRate;
    const BUFFER_SIZE = 4096;
    this.buffer = new Float32Array(BUFFER_SIZE);
    this.bufferMask = BUFFER_SIZE - 1;
    this.writeIndex = 0;
    this.currentMidiNote = 69;
    this.currentF0 = 440.0;
    this.integerDelay = 10;
    this.thiranA = 0.0;
    this.thiranState = 0.0;
    this.dispA = -0.25;
    this.dispState = 0.0;
    this.lossGain = 0.997;
    this.lossGainTarget = 0.997;
    this.lossS = 0.15;
    this.lossSTarget = 0.15;
    this.lossState = 0.0;
    this.fadeCoeff = 1.0 - Math.exp(-1.0 / (0.005 * this.sampleRate));
    this.configureTuning(69, 440.0);
  }

  setFrequency(f0, midiNote) {
    const note = midiNote || Math.max(21, Math.min(108, Math.round(69.0 + 12.0 * Math.log2(f0 / 440.0))));
    this.configureTuning(note, f0);
  }

  configureTuning(midiNote, f0) {
    this.currentMidiNote = midiNote;
    this.currentF0 = f0;
    const w0 = (2.0 * Math.PI * f0) / this.sampleRate;

    this.dispA = -0.15 - 0.20 * Math.max(0.0, Math.min(1.0, (108.0 - midiNote) / 87.0));
    const numRe = this.dispA + Math.cos(w0);
    const numIm = -Math.sin(w0);
    const denRe = 1.0 + this.dispA * Math.cos(w0);
    const denIm = -this.dispA * Math.sin(w0);
    const phiDisp = Math.atan2(numIm, numRe) - Math.atan2(denIm, denRe);
    const tauDisp = -phiDisp / w0;

    this.lossS = 0.15;
    this.lossSTarget = 0.15;
    const lossRe = (1.0 - this.lossS) + this.lossS * Math.cos(w0);
    const lossIm = -this.lossS * Math.sin(w0);
    const phiLoss = Math.atan2(lossIm, lossRe);
    const tauLoss = -phiLoss / w0;

    const targetDelay = this.sampleRate / f0;
    const remainingDelay = targetDelay - tauLoss - tauDisp;

    this.integerDelay = Math.max(1, Math.floor(remainingDelay - 0.5));
    const dFrac = remainingDelay - this.integerDelay;
    this.thiranA = (1.0 - dFrac) / (1.0 + dFrac);

    const t60Sus = 1.2 + 10.8 * Math.pow(Math.max(0.0, (108.0 - midiNote) / 87.0), 1.2);
    const gSus = Math.pow(10.0, -3.0 / (f0 * t60Sus));
    this.lossGain = gSus;
    this.lossGainTarget = gSus;
  }

  setDamper(damped, midiNote, f0) {
    const note = midiNote || this.currentMidiNote;
    const freq = f0 || this.currentF0;
    if (damped) {
      if (note <= 88) {
        const t60Rel = 0.09 + 0.16 * Math.max(0.0, (36.0 - note) / 15.0);
        this.lossGainTarget = Math.pow(10.0, -3.0 / (freq * t60Rel));
        this.lossSTarget = 0.42;
      } else {
        const t60Sus = 1.2 + 10.8 * Math.pow(Math.max(0.0, (108.0 - note) / 87.0), 1.2);
        this.lossGainTarget = Math.pow(10.0, -3.0 / (freq * t60Sus));
        this.lossSTarget = 0.15;
      }
    } else {
      const t60Sus = 1.2 + 10.8 * Math.pow(Math.max(0.0, (108.0 - note) / 87.0), 1.2);
      this.lossGainTarget = Math.pow(10.0, -3.0 / (freq * t60Sus));
      this.lossSTarget = 0.15;
    }
  }

  injectCoupling(delta) {
    const idx = (this.writeIndex - 1 + this.buffer.length) & this.bufferMask;
    this.buffer[idx] += delta;
  }

  reset() {
    this.buffer.fill(0.0);
    this.writeIndex = 0;
    this.thiranState = 0.0;
    this.dispState = 0.0;
    this.lossState = 0.0;
  }

  processSample(excitation) {
    this.lossGain += this.fadeCoeff * (this.lossGainTarget - this.lossGain);
    this.lossS += this.fadeCoeff * (this.lossSTarget - this.lossS);

    const readIndex = (this.writeIndex - this.integerDelay + this.buffer.length) & this.bufferMask;
    const delayOut = this.buffer[readIndex] + excitation;

    const thiranOut = this.thiranA * delayOut + this.thiranState;
    this.thiranState = delayOut - this.thiranA * thiranOut;

    const dispOut = this.dispA * thiranOut + this.dispState;
    this.dispState = thiranOut - this.dispA * dispOut;

    let lossOut = this.lossGain * ((1.0 - this.lossS) * dispOut + this.lossS * this.lossState);
    this.lossState = dispOut;

    if (Math.abs(lossOut) < 1e-15) {
      lossOut = 0.0;
      this.lossState = 0.0;
    }

    this.buffer[this.writeIndex] = lossOut;
    this.writeIndex = (this.writeIndex + 1) & this.bufferMask;
    return lossOut;
  }

  process(excitation) {
    return this.processSample(excitation);
  }
}

class HammerModel {
  constructor(sampleRate) {
    this.sampleRate = sampleRate;
    this.pulseBuffer = new Float32Array(1024);
    this.pulseLength = 0;
    this.pulseIndex = 0;
  }

  strike(f0OrMidi, midiVelocity) {
    let f0 = f0OrMidi;
    if (Number.isInteger(f0OrMidi) && f0OrMidi >= 21 && f0OrMidi <= 108 && f0OrMidi > 55) {
      f0 = 440.0 * Math.pow(2.0, (f0OrMidi - 69.0) / 12.0);
    }
    const v = Math.max(0.01, Math.min(1.0, midiVelocity / 127.0));
    const amplitude = Math.pow(v, 1.5);
    const tau0 = 0.0018 * Math.pow(440.0 / Math.max(100.0, f0), 0.15);
    const tauC = Math.max(2.0 / this.sampleRate, Math.min(0.008, tau0 * Math.pow(v, -0.4)));
    const pulseLen = Math.max(4, Math.min(512, Math.round(tauC * this.sampleRate)));

    const targetDelay = this.sampleRate / f0;
    const strikeDelay = Math.max(1, Math.round(targetDelay / 8.0));
    const totalLen = pulseLen + strikeDelay;
    this.pulseLength = Math.min(totalLen, this.pulseBuffer.length);
    this.pulseIndex = 0;

    for (let n = 0; n < this.pulseLength; n++) {
      let p1 = 0.0;
      if (n < pulseLen) {
        const s1 = Math.sin((Math.PI * n) / pulseLen);
        p1 = amplitude * s1 * s1;
      }
      let p2 = 0.0;
      const idx2 = n - strikeDelay;
      if (idx2 >= 0 && idx2 < pulseLen) {
        const s2 = Math.sin((Math.PI * idx2) / pulseLen);
        p2 = amplitude * s2 * s2;
      }
      this.pulseBuffer[n] = p1 - p2;
    }
  }

  getSample() {
    if (this.pulseIndex < this.pulseLength) {
      return this.pulseBuffer[this.pulseIndex++];
    }
    return 0.0;
  }

  processSample() {
    return this.getSample();
  }

  isStriking() {
    return this.pulseIndex < this.pulseLength;
  }

  isActive() {
    return this.pulseIndex < this.pulseLength;
  }

  reset() {
    this.pulseLength = 0;
    this.pulseIndex = 0;
  }
}

class SoundboardResonance {
  constructor(sampleRate) {
    this.sampleRate = sampleRate;
    this.radiationMix = 0.35;
    this.dcR = 0.995;
    this.dcX1 = 0.0;
    this.dcY1 = 0.0;

    const modeConfigs = [
      { f: 58.0,  Q: 20.0, g: 0.35 },
      { f: 115.0, Q: 22.0, g: 0.30 },
      { f: 182.0, Q: 18.0, g: 0.25 },
      { f: 275.0, Q: 15.0, g: 0.20 },
      { f: 410.0, Q: 14.0, g: 0.15 },
      { f: 620.0, Q: 12.0, g: 0.12 },
      { f: 910.0, Q: 10.0, g: 0.08 },
      { f: 1450.0,Q: 8.0,  g: 0.05 },
    ];

    this.resonators = modeConfigs.map((m) => {
      const w0 = (2.0 * Math.PI * m.f) / this.sampleRate;
      const alpha = Math.sin(w0) / (2.0 * m.Q);
      const a0 = 1.0 + alpha;
      return {
        b0: (alpha / a0) * m.g,
        a1: (-2.0 * Math.cos(w0)) / a0,
        a2: (1.0 - alpha) / a0,
        w1: 0.0,
        w2: 0.0,
      };
    });
  }

  processResonanceOnly(input) {
    let modalSum = 0.0;
    const len = this.resonators.length;
    for (let i = 0; i < len; i++) {
      const r = this.resonators[i];
      let w0 = input - r.a1 * r.w1 - r.a2 * r.w2;
      if (Math.abs(w0) < 1e-12) {
        w0 = 0.0;
        r.w1 = 0.0;
        r.w2 = 0.0;
      }
      const y = r.b0 * (w0 - r.w2);
      r.w2 = r.w1;
      r.w1 = w0;
      modalSum += y;
    }
    let dcOut = modalSum - this.dcX1 + this.dcR * this.dcY1;
    this.dcX1 = modalSum;
    this.dcY1 = dcOut;
    if (Math.abs(dcOut) < 1e-12) {
      dcOut = 0.0;
      this.dcY1 = 0.0;
      this.dcX1 = 0.0;
    }
    return dcOut;
  }

  reset() {
    for (let i = 0; i < this.resonators.length; i++) {
      this.resonators[i].w1 = 0.0;
      this.resonators[i].w2 = 0.0;
    }
    this.dcX1 = 0.0;
    this.dcY1 = 0.0;
  }
}

const VoiceState = {
  IDLE: 0,
  ATTACK: 1,
  SUSTAIN: 2,
  RELEASE: 3,
  FADING_OUT: 4,
};

class PianoVoice {
  constructor(sampleRate) {
    this.sampleRate = sampleRate;
    this.state = VoiceState.IDLE;
    this.strings = [
      new WaveguideString(sampleRate),
      new WaveguideString(sampleRate),
      new WaveguideString(sampleRate)
    ];
    this.numStrings = 1;
    this.couplingCoeff = 0.02;
    this.hammer = new HammerModel(sampleRate);
    this.midiNote = 0;
    this.f0 = 440.0;
    this.velocity = 0;
    this.keyHeld = false;
    this.damperDown = false;
    this.hasDamper = true;
    this.startTimeSamples = 0;
    this.releaseSamplesCount = 0;
    this.envelope = 0.0;
    this.lastOutput = 0.0;
    this.stolenResidual = 0.0;
    this.fadeSamplesRemaining = 0;
    this.fadeLengthSamples = Math.max(1, Math.round(0.003 * sampleRate));
    this.dcX1 = 0.0;
    this.dcY1 = 0.0;
    this.dcR = 0.995;
    this.sampleOffsetRemaining = 0;
  }

  trigger(options, currentSampleClock) {
    const { midiNote, velocity, sampleOffset = 0 } = options;
    this.midiNote = midiNote;
    this.velocity = velocity;
    this.keyHeld = true;
    this.damperDown = false;
    this.startTimeSamples = currentSampleClock;
    this.sampleOffsetRemaining = Math.max(0, sampleOffset);
    this.releaseSamplesCount = 0;
    this.hasDamper = midiNote < 89;
    this.f0 = 440.0 * Math.pow(2.0, (midiNote - 69.0) / 12.0);

    if (midiNote < 40) {
      this.numStrings = 1;
      this.strings[0].setFrequency(this.f0, midiNote);
    } else if (midiNote < 60) {
      this.numStrings = 2;
      const detune = 0.15;
      this.strings[0].setFrequency(this.f0 - detune * 0.5, midiNote);
      this.strings[1].setFrequency(this.f0 + detune * 0.5, midiNote);
    } else {
      this.numStrings = 3;
      const detune = 0.20;
      this.strings[0].setFrequency(this.f0 - detune, midiNote);
      this.strings[1].setFrequency(this.f0, midiNote);
      this.strings[2].setFrequency(this.f0 + detune, midiNote);
    }

    for (let i = 0; i < this.numStrings; i++) {
      this.strings[i].setDamper(false, midiNote, this.f0);
      this.strings[i].reset();
    }

    this.hammer.strike(this.f0, velocity);
    this.state = VoiceState.ATTACK;
  }

  stealAndRetrigger(options, currentSampleClock) {
    this.stolenResidual = this.lastOutput;
    this.fadeSamplesRemaining = this.fadeLengthSamples;
    this.dcX1 = 0.0;
    this.dcY1 = 0.0;
    this.trigger(options, currentSampleClock);
  }

  noteOff(sampleOffset = 0, sustainPedalDown = false) {
    this.keyHeld = false;
    if (!sustainPedalDown) {
      this.applyDamper();
    }
  }

  onSustainPedalChange(pedalDown) {
    if (!pedalDown && !this.keyHeld && (this.state === VoiceState.SUSTAIN || this.state === VoiceState.ATTACK)) {
      this.applyDamper();
    }
  }

  applyDamper() {
    if (!this.hasDamper) {
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

  processSample() {
    if (this.state === VoiceState.IDLE) return 0.0;

    if (this.sampleOffsetRemaining > 0) {
      this.sampleOffsetRemaining--;
      if (this.fadeSamplesRemaining > 0) {
        const fadeGain = this.fadeSamplesRemaining / this.fadeLengthSamples;
        this.fadeSamplesRemaining--;
        return this.stolenResidual * fadeGain;
      }
      return 0.0;
    }

    let hammerForce = 0.0;
    if (this.state === VoiceState.ATTACK) {
      hammerForce = this.hammer.processSample();
      if (!this.hammer.isActive()) {
        this.state = VoiceState.SUSTAIN;
      }
    }

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
      stringSum = (s0 + s1) * 0.70710678;
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
      stringSum = (s0 + s1 + s2) * 0.57735027;
    }

    const dcOut = stringSum - this.dcX1 + this.dcR * this.dcY1;
    this.dcX1 = stringSum;
    this.dcY1 = dcOut;

    let finalSample = dcOut;
    if (this.fadeSamplesRemaining > 0) {
      const fadeFactor = this.fadeSamplesRemaining / this.fadeLengthSamples;
      finalSample += this.stolenResidual * fadeFactor;
      this.fadeSamplesRemaining--;
    }

    if (Math.abs(finalSample) < 1e-15) finalSample = 0.0;

    const absSample = Math.abs(finalSample);
    this.envelope = this.envelope * 0.999 + absSample * 0.001;

    if (this.state === VoiceState.RELEASE) {
      this.releaseSamplesCount++;
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

  reset() {
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

  getState() { return this.state; }
  isActive() { return this.state !== VoiceState.IDLE; }
  getMidiNote() { return this.midiNote; }
  getEnvelope() { return this.envelope; }
  getStartTime() { return this.startTimeSamples; }
}

class PianoSynthesizer {
  constructor(sampleRate) {
    this.sampleRate = sampleRate;
    this.MAX_VOICES = 24;
    this.voices = [];
    this.soundboard = new SoundboardResonance(sampleRate);
    this.sustainPedalDown = false;
    this.currentSampleClock = 0;
    this.masterGain = 0.45;

    for (let i = 0; i < this.MAX_VOICES; i++) {
      this.voices.push(new PianoVoice(sampleRate));
    }
  }

  noteOn(options) {
    const { midiNote, velocity, sampleOffset } = options;
    if (midiNote < 21 || midiNote > 108) return;
    if (velocity === 0) {
      this.noteOff({ midiNote, sampleOffset });
      return;
    }
    if (velocity < 0) return;

    const voice = this.allocateVoice(midiNote);
    if (voice.isActive()) {
      voice.stealAndRetrigger(options, this.currentSampleClock);
    } else {
      voice.trigger(options, this.currentSampleClock);
    }
  }

  noteOff(options) {
    const { midiNote, sampleOffset = 0 } = options;
    for (let i = 0; i < this.MAX_VOICES; i++) {
      const v = this.voices[i];
      if (v.isActive() && v.getMidiNote() === midiNote) {
        v.noteOff(sampleOffset, this.sustainPedalDown);
      }
    }
  }

  setSustainPedal(down) {
    this.sustainPedalDown = down;
    for (let i = 0; i < this.MAX_VOICES; i++) {
      this.voices[i].onSustainPedalChange(down);
    }
  }

  allNotesOff() {
    this.sustainPedalDown = false;
    for (let i = 0; i < this.MAX_VOICES; i++) {
      this.voices[i].reset();
    }
    this.soundboard.reset();
  }

  render(outputLeft, outputRight, numSamples) {
    outputLeft.fill(0.0, 0, numSamples);
    outputRight.fill(0.0, 0, numSamples);

    for (let v = 0; v < this.MAX_VOICES; v++) {
      const voice = this.voices[v];
      if (!voice.isActive()) continue;

      const midi = voice.getMidiNote();
      const pan = -0.6 + 1.2 * ((midi - 21.0) / 87.0);
      const panL = Math.cos((Math.PI / 4.0) * (1.0 + pan));
      const panR = Math.sin((Math.PI / 4.0) * (1.0 + pan));

      for (let i = 0; i < numSamples; i++) {
        const sample = voice.processSample();
        outputLeft[i] += sample * panL;
        outputRight[i] += sample * panR;
      }
    }

    for (let i = 0; i < numSamples; i++) {
      const monoSum = (outputLeft[i] + outputRight[i]) * 0.5;
      const res = this.soundboard.processResonanceOnly(monoSum);
      outputLeft[i] += res * 0.35;
      outputRight[i] += res * 0.35;
    }

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

  getActiveVoiceCount() {
    let count = 0;
    for (let i = 0; i < this.MAX_VOICES; i++) {
      if (this.voices[i].isActive()) count++;
    }
    return count;
  }

  allocateVoice(midiNote) {
    for (let i = 0; i < this.MAX_VOICES; i++) {
      const v = this.voices[i];
      if (v.isActive() && v.getMidiNote() === midiNote) return v;
    }
    for (let i = 0; i < this.MAX_VOICES; i++) {
      const v = this.voices[i];
      if (v.getState() === VoiceState.IDLE) return v;
    }
    let bestReleaseVoice = null;
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
    if (bestReleaseVoice !== null) return bestReleaseVoice;

    let bestActiveVoice = this.voices[0];
    let minScore = Infinity;
    for (let i = 0; i < this.MAX_VOICES; i++) {
      const v = this.voices[i];
      const ageSeconds = (this.currentSampleClock - v.getStartTime()) / this.sampleRate;
      const score = v.getEnvelope() / (1.0 + ageSeconds);
      if (score < minScore) {
        minScore = score;
        bestActiveVoice = v;
      }
    }
    return bestActiveVoice;
  }
}

class PianoWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.synthesizer = new PianoSynthesizer(sampleRate);
    this.eventQueue = [];
    this.telemetryCounter = 0;

    this.port.onmessage = (event) => {
      if (event.data) {
        this.eventQueue.push(event.data);
      }
    };

    this.port.postMessage({ type: 'ready', sampleRate: sampleRate });
  }

  process(inputs, outputs, parameters) {
    const output = outputs[0];
    if (!output || output.length === 0) return true;

    const leftChannel = output[0];
    const rightChannel = output.length > 1 ? output[1] : output[0];
    const numSamples = leftChannel.length;

    while (this.eventQueue.length > 0) {
      const msg = this.eventQueue.shift();
      switch (msg.type) {
        case 'noteOn':
          this.synthesizer.noteOn({
            midiNote: msg.midiNote,
            velocity: msg.velocity,
            sampleOffset: msg.sampleOffset || 0
          });
          break;
        case 'noteOff':
          this.synthesizer.noteOff({
            midiNote: msg.midiNote,
            sampleOffset: msg.sampleOffset || 0
          });
          break;
        case 'setSustainPedal':
          this.synthesizer.setSustainPedal(msg.down);
          break;
        case 'allNotesOff':
          this.synthesizer.allNotesOff();
          break;
      }
    }

    this.synthesizer.render(leftChannel, rightChannel, numSamples);

    this.telemetryCounter += numSamples;
    if (this.telemetryCounter >= 4410) {
      this.telemetryCounter = 0;
      this.port.postMessage({
        type: 'voiceCount',
        count: this.synthesizer.getActiveVoiceCount()
      });
    }

    return true;
  }
}

registerProcessor('piano-worklet-processor', PianoWorkletProcessor);
`;

/**
 * Creates an in-memory Blob URL for audioWorklet.addModule().
 */
export function createPianoWorkletBlobUrl(customCode?: string): string {
  const code = customCode ?? PIANO_WORKLET_PROCESSOR_CODE;
  const blob = new Blob([code], { type: 'application/javascript' });
  return URL.createObjectURL(blob);
}
