/**
 * WaveguideString.ts
 * High-performance digital waveguide string physical simulation module.
 * Incorporates Thiran 1st-order fractional delay, loop loss filtering,
 * and cascaded allpass dispersion filters for stiff string inharmonicity.
 */

export interface WaveguideStringConfig {
  sampleRate: number;
  midiNote?: number;
  frequency?: number;
}

export class WaveguideString {
  private readonly sampleRate: number;
  private readonly buffer: Float32Array;
  private readonly bufferMask: number;
  private writeIndex: number = 0;

  // Track current tuning
  private currentMidiNote: number = 69;
  private currentF0: number = 440.0;

  // Waveguide loop parameters
  private integerDelay: number = 10;
  private thiranA: number = 0.0;
  private thiranState: number = 0.0;

  // Dispersion allpass filter (1 stage)
  private dispA: number = -0.25;
  private dispState: number = 0.0;

  // Loop loss filter parameters
  private lossGain: number = 0.997;
  private lossGainTarget: number = 0.997;
  private lossS: number = 0.15;
  private lossSTarget: number = 0.15;
  // Sustained-mode loop lowpass coefficient (register dependent, see sustainLossS)
  private lossSSus: number = 0.15;
  private lossState: number = 0.0;
  private readonly fadeCoeff: number;

  constructor(configOrSampleRate: number | WaveguideStringConfig) {
    if (typeof configOrSampleRate === 'number') {
      this.sampleRate = configOrSampleRate;
    } else {
      this.sampleRate = configOrSampleRate.sampleRate;
    }

    // Pre-allocate power-of-two buffer (4096 floats fits <= 96kHz A0)
    const BUFFER_SIZE = 4096;
    this.buffer = new Float32Array(BUFFER_SIZE);
    this.bufferMask = BUFFER_SIZE - 1;

    // 5ms smoothing rate for click-free parameter transitions
    this.fadeCoeff = 1.0 - Math.exp(-1.0 / (0.005 * this.sampleRate));

    if (typeof configOrSampleRate !== 'number') {
      const midiNote = configOrSampleRate.midiNote ?? 69;
      const f0 = configOrSampleRate.frequency ?? 440.0 * Math.pow(2.0, (midiNote - 69.0) / 12.0);
      this.configureTuning(midiNote, f0);
    } else {
      this.configureTuning(69, 440.0);
    }
  }

  /**
   * Sets frequency and optional midi note number, updating tuning coefficients.
   */
  public setFrequency(f0: number, midiNote?: number): void {
    const note = midiNote ?? Math.max(21, Math.min(108, Math.round(69.0 + 12.0 * Math.log2(f0 / 440.0))));
    this.configureTuning(note, f0);
  }

  /**
   * Calculates loop delay, fractional allpass, and dispersion coefficients
   * with exact phase delay compensation at fundamental frequency omega0.
   */
  public configureTuning(midiNote: number, f0: number): void {
    this.currentMidiNote = midiNote;
    this.currentF0 = f0;

    const w0 = (2.0 * Math.PI * f0) / this.sampleRate;

    // 1. Dispersion allpass coefficient
    // Bass strings have higher inharmonicity; treble wire has lower
    this.dispA = -0.15 - 0.20 * Math.max(0.0, Math.min(1.0, (108.0 - midiNote) / 87.0));

    // Exact phase delay of dispersion filter at w0
    const numRe = this.dispA + Math.cos(w0);
    const numIm = -Math.sin(w0);
    const denRe = 1.0 + this.dispA * Math.cos(w0);
    const denIm = -this.dispA * Math.sin(w0);
    const phiDisp = Math.atan2(numIm, numRe) - Math.atan2(denIm, denRe);
    const tauDisp = -phiDisp / w0;

    // 2. Loop loss filter phase delay at w0 (register dependent s, 0.15 for bass/mid)
    const t60Sus = WaveguideString.sustainT60(midiNote);
    this.lossSSus = WaveguideString.sustainLossS(f0, t60Sus, this.sampleRate);
    this.lossS = this.lossSSus;
    this.lossSTarget = this.lossSSus;
    const lossRe = (1.0 - this.lossS) + this.lossS * Math.cos(w0);
    const lossIm = -this.lossS * Math.sin(w0);
    const phiLoss = Math.atan2(lossIm, lossRe);
    const tauLoss = -phiLoss / w0;

    // 3. Exact target delay and fractional split
    const targetDelay = this.sampleRate / f0;
    const remainingDelay = targetDelay - tauLoss - tauDisp;

    // Stabilized delay splitting: d strictly in [0.5, 1.5)
    this.integerDelay = Math.max(1, Math.floor(remainingDelay - 0.5));
    const dFrac = remainingDelay - this.integerDelay;
    this.thiranA = (1.0 - dFrac) / (1.0 + dFrac);

    // 4. Configure natural sustain and release decay times
    const gSus = this.sustainLoopGain(f0, t60Sus);

    this.lossGain = gSus;
    this.lossGainTarget = gSus;
  }

  /** Natural (undamped) T60 of the fundamental in seconds. */
  private static sustainT60(midiNote: number): number {
    return 1.8 + 10.2 * Math.pow(Math.max(0.0, (108.0 - midiNote) / 87.0), 1.2);
  }

  /**
   * Loop lowpass coefficient s. High strings circulate many times per second, so the
   * 1-pole loss filter (which attenuates the fundamental by ~s(1-s)(1-cos w0) per pass)
   * would otherwise dominate the decay and mute the treble. Cap s so the filter consumes at
   * most half of the per-pass loss budget implied by the target T60.
   */
  private static sustainLossS(f0: number, t60: number, sampleRate: number): number {
    const w0 = (2.0 * Math.PI * f0) / sampleRate;
    const oneMinusCos = Math.max(1e-9, 1.0 - Math.cos(w0));
    const s = 3.45 / (f0 * t60 * oneMinusCos);
    return Math.max(0.005, Math.min(0.15, s));
  }

  /**
   * Per-pass gain g such that the fundamental (attenuated by |H_lp(w0)| each pass) decays
   * with the requested T60. g < 1 is guaranteed by sustainLossS (DC gain stays < 1).
   */
  private sustainLoopGain(f0: number, t60: number): number {
    const w0 = (2.0 * Math.PI * f0) / this.sampleRate;
    const s = this.lossSSus;
    const mag = Math.sqrt(
      (1.0 - s) * (1.0 - s) + s * s + 2.0 * s * (1.0 - s) * Math.cos(w0),
    );
    const g = Math.pow(10.0, -3.0 / (f0 * t60)) / Math.max(1e-6, mag);
    return Math.min(0.99995, g);
  }

  /**
   * Sets damper down (noteOff) or lifted (noteOn).
   */
  public setDamper(damped: boolean, midiNote?: number, f0?: number): void {
    const note = midiNote ?? this.currentMidiNote;
    const freq = f0 ?? this.currentF0;

    if (damped) {
      if (note <= 88) {
        const t60Rel = 0.09 + 0.16 * Math.max(0.0, (36.0 - note) / 15.0);
        this.lossGainTarget = Math.pow(10.0, -3.0 / (freq * t60Rel));
        this.lossSTarget = 0.42;
      } else {
        // High treble register (> F#6) has no dampers on acoustic pianos
        this.lossGainTarget = this.sustainLoopGain(freq, WaveguideString.sustainT60(note));
        this.lossSTarget = this.lossSSus;
      }
    } else {
      this.lossGainTarget = this.sustainLoopGain(freq, WaveguideString.sustainT60(note));
      this.lossSTarget = this.lossSSus;
    }
  }

  /**
   * Injects bridge coupling delta into the circulating wave.
   */
  public injectCoupling(delta: number): void {
    const idx = (this.writeIndex - 1 + this.buffer.length) & this.bufferMask;
    this.buffer[idx] += delta;
  }

  /**
   * Resets all internal delay and filter states.
   */
  public reset(): void {
    this.buffer.fill(0.0);
    this.writeIndex = 0;
    this.thiranState = 0.0;
    this.dispState = 0.0;
    this.lossState = 0.0;
  }

  /**
   * Processes one audio sample. Zero allocations.
   * @param excitation Sample from hammer model to inject into the loop.
   * @returns Synthesized string output sample.
   */
  public processSample(excitation: number): number {
    // Parameter smoothing for click-free damping transitions
    this.lossGain += this.fadeCoeff * (this.lossGainTarget - this.lossGain);
    this.lossS += this.fadeCoeff * (this.lossSTarget - this.lossS);

    // 1. Read from integer delay line
    const readIndex = (this.writeIndex - this.integerDelay + this.buffer.length) & this.bufferMask;
    const delayOut = this.buffer[readIndex] + excitation;

    // 2. 1st-Order Thiran fractional delay allpass
    const thiranOut = this.thiranA * delayOut + this.thiranState;
    this.thiranState = delayOut - this.thiranA * thiranOut;

    // 3. 1st-Order allpass dispersion filter
    const dispOut = this.dispA * thiranOut + this.dispState;
    this.dispState = thiranOut - this.dispA * dispOut;

    // 4. 1st-Order loop loss filter: H_loss(z) = g * ((1 - s) + s * z^-1)
    let lossOut = this.lossGain * ((1.0 - this.lossS) * dispOut + this.lossS * this.lossState);
    this.lossState = dispOut;

    // Denormal killer on loop state
    if (Math.abs(lossOut) < 1e-15) {
      lossOut = 0.0;
      this.lossState = 0.0;
    }

    // 5. Write back to circular delay buffer
    this.buffer[this.writeIndex] = lossOut;
    this.writeIndex = (this.writeIndex + 1) & this.bufferMask;

    return lossOut;
  }

  /**
   * Alias for processSample to match alternative naming conventions.
   */
  public process(excitation: number): number {
    return this.processSample(excitation);
  }
}
