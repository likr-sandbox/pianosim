/**
 * SoundboardResonance.ts
 * Soundboard modal resonator bank, DC blocking filter, denormal killer, and soft saturation.
 * Simulates spruce soundboard body acoustic radiation and modal dispersion.
 */

interface ModalResonator {
  b0: number;
  a1: number;
  a2: number;
  w1: number;
  w2: number;
}

export class SoundboardResonance {
  private readonly sampleRate: number;
  private readonly resonators: ModalResonator[];
  private readonly radiationMix: number = 0.35;

  // DC blocking filter state (R = 0.995)
  private readonly dcR: number = 0.995;
  private dcX1: number = 0.0;
  private dcY1: number = 0.0;

  constructor(sampleRate: number) {
    this.sampleRate = sampleRate;

    // Physical modal frequencies (Hz), Q factors, and radiation gains
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

  /**
   * Processes a single audio sample through the soundboard modal bank,
   * DC blocker, denormal killer, and master soft saturation.
   * @param input Sample from active string voices.
   * @returns Resonant, DC-free, bounded output sample in [-1.0, 1.0].
   */
  public processSample(input: number): number {
    // 1. Parallel modal resonator bank (Direct Form II)
    let modalSum = 0.0;
    const len = this.resonators.length;

    for (let i = 0; i < len; i++) {
      const r = this.resonators[i];
      let w0 = input - r.a1 * r.w1 - r.a2 * r.w2;

      // Denormal prevention in resonator state
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

    // 2. Mix direct string radiation with soundboard body modes
    const mixed = (1.0 - this.radiationMix) * input + this.radiationMix * modalSum;

    // 3. DC Blocking Filter: y[n] = x[n] - x[n-1] + R * y[n-1]
    let dcOut = mixed - this.dcX1 + this.dcR * this.dcY1;
    this.dcX1 = mixed;
    this.dcY1 = dcOut;

    // 4. Denormal Killer on output
    if (Math.abs(dcOut) < 1e-12) {
      dcOut = 0.0;
      this.dcY1 = 0.0;
      this.dcX1 = 0.0;
    }

    // 5. Master Soft Saturation (tanh) to strictly bound output within [-1.0, 1.0]
    return Math.tanh(dcOut);
  }

  /**
   * Alias for processSample.
   */
  public process(input: number): number {
    return this.processSample(input);
  }

  /**
   * Evaluates and returns only the modal resonator bank response (DC filtered),
   * suitable for parallel stereo bus insertion.
   */
  public processResonanceOnly(input: number): number {
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

    // DC block modal sum
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

  /**
   * Clears all resonator states and DC filter memory.
   */
  public reset(): void {
    const len = this.resonators.length;
    for (let i = 0; i < len; i++) {
      this.resonators[i].w1 = 0.0;
      this.resonators[i].w2 = 0.0;
    }
    this.dcX1 = 0.0;
    this.dcY1 = 0.0;
  }
}
