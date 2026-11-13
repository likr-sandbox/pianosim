/**
 * HammerModel.ts
 * Non-linear felt compression excitation model with strike position comb filtering.
 * Evaluates F = K * (delta_y)^p, scaling contact duration (tau_c ~ v^-0.4),
 * peak amplitude (A ~ v^1.5), and suppresses 7th/8th harmonics at x0 = L/8.
 */

export class HammerModel {
  private readonly sampleRate: number;
  private readonly pulseBuffer: Float32Array;
  private pulseLength: number = 0;
  private pulseIndex: number = 0;

  constructor(sampleRate: number) {
    this.sampleRate = sampleRate;
    // Pre-allocate buffer for maximum possible pulse length (1024 samples)
    this.pulseBuffer = new Float32Array(1024);
  }

  /**
   * Computes velocity-dependent excitation pulse and strike comb filter.
   * Accepts fundamental frequency f0 or MIDI note number.
   * @param f0OrMidi Fundamental frequency in Hz or MIDI note number (21 to 108).
   * @param midiVelocity MIDI velocity (1 to 127).
   */
  public strike(f0OrMidi: number, midiVelocity: number): void {
    let f0 = f0OrMidi;
    // Automatically convert integer MIDI note numbers to Hz if in [21, 108] and > 55
    if (Number.isInteger(f0OrMidi) && f0OrMidi >= 21 && f0OrMidi <= 108 && f0OrMidi > 55) {
      f0 = 440.0 * Math.pow(2.0, (f0OrMidi - 69.0) / 12.0);
    }

    const v = Math.max(0.01, Math.min(1.0, midiVelocity / 127.0));

    // 1. Amplitude scaling: A(v) proportional to v^1.5
    const amplitude = Math.pow(v, 1.5);

    // 2. Contact duration: tau_c proportional to v^-0.4
    const tau0 = 0.0018 * Math.pow(440.0 / Math.max(100.0, f0), 0.15);
    const tauC = Math.max(2.0 / this.sampleRate, Math.min(0.008, tau0 * Math.pow(v, -0.4)));
    const pulseLen = Math.max(4, Math.min(512, Math.round(tauC * this.sampleRate)));

    // 3. Strike position comb filter delay (x0 = L/8)
    const targetDelay = this.sampleRate / f0;
    const strikeDelay = Math.max(1, Math.round(targetDelay / 8.0));

    const totalLen = pulseLen + strikeDelay;
    this.pulseLength = Math.min(totalLen, this.pulseBuffer.length);
    this.pulseIndex = 0;

    // 4. Generate raised-sine pulse with comb cancellation: x[n] = p[n] - p[n - D_strike]
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

  /**
   * Explicit MIDI note strike method.
   */
  public strikeNote(midiNote: number, midiVelocity: number): void {
    const f0 = 440.0 * Math.pow(2.0, (midiNote - 69.0) / 12.0);
    this.strike(f0, midiVelocity);
  }

  /**
   * Returns next excitation sample. Outputs 0.0 when hammer is no longer contacting string.
   */
  public getSample(): number {
    if (this.pulseIndex < this.pulseLength) {
      return this.pulseBuffer[this.pulseIndex++];
    }
    return 0.0;
  }

  /**
   * Alias for getSample to match processSample convention.
   */
  public processSample(): number {
    return this.getSample();
  }

  /**
   * Returns true while hammer is actively contacting and striking the string.
   */
  public isStriking(): boolean {
    return this.pulseIndex < this.pulseLength;
  }

  /**
   * Alias for isStriking.
   */
  public isActive(): boolean {
    return this.pulseIndex < this.pulseLength;
  }

  public reset(): void {
    this.pulseLength = 0;
    this.pulseIndex = 0;
  }
}
