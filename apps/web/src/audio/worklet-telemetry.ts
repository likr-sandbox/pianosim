/** AudioWorklet callback timing, Date.now() 1 ms quantization guard, and 32-block stride telemetry. */

import type { WorkletOutboundMessage } from './abi';

const TELEMETRY_STRIDE_BLOCKS = 32;
const DATE_SINGLE_BLOCK_STALL_US = 4_000;
const INTER_ARRIVAL_STALL_MS = 25;

export const hasPerfNow = (): boolean => {
  const p = typeof performance !== 'undefined' ? performance : globalThis?.performance;
  return typeof p?.now === 'function';
};

export const workletNowMs = (): number => (hasPerfNow() ? performance.now() : Date.now());

export class WorkletTelemetryTracker {
  private blockCount = 0;
  private strideBlockCount = 0;
  private windowSumUs = 0;
  private windowMaxUs = 0;
  private windowPeakL = 0;
  private windowPeakR = 0;
  private windowSqSum = 0;
  private windowSamples = 0;
  private windowClipped = false;
  private totalOverruns = 0;
  private strideOverruns = 0;
  private singleBlockOverruns = 0;
  private stallCount = 0;
  private prevProcessMs = 0;

  recordProcessStart(t0Ms: number): void {
    if (this.prevProcessMs > 0 && t0Ms - this.prevProcessMs > INTER_ARRIVAL_STALL_MS) {
      this.stallCount += 1;
    }
    this.prevProcessMs = t0Ms;
  }

  recordBlock(
    outL: Float32Array,
    outR: Float32Array | null,
    totalFrames: number,
    sampleRateHz: number,
    dtUs: number,
    perfAvailable: boolean,
    activeVoices: number,
    canEmit: boolean,
    postOutbound: (msg: WorkletOutboundMessage) => void,
  ): void {
    const d1BudgetUs = (totalFrames / sampleRateHz) * 700_000.0;
    if (perfAvailable ? dtUs > d1BudgetUs : dtUs >= DATE_SINGLE_BLOCK_STALL_US) {
      this.singleBlockOverruns += 1;
      this.totalOverruns += 1;
    }
    this.windowSumUs += dtUs;
    if (dtUs > this.windowMaxUs) this.windowMaxUs = dtUs;

    let blockPeakL = 0;
    let blockPeakR = 0;
    let blockSqSum = 0;
    for (let i = 0; i < totalFrames; i++) {
      const l = outL[i];
      const r = outR ? outR[i] : l;
      const al = Math.abs(l);
      const ar = Math.abs(r);
      if (al > blockPeakL) blockPeakL = al;
      if (ar > blockPeakR) blockPeakR = ar;
      const m = 0.5 * (l + r);
      blockSqSum += m * m;
    }
    if (blockPeakL > this.windowPeakL) this.windowPeakL = blockPeakL;
    if (blockPeakR > this.windowPeakR) this.windowPeakR = blockPeakR;
    if (blockPeakL >= 0.999 || blockPeakR >= 0.999) this.windowClipped = true;
    this.windowSqSum += blockSqSum;
    this.windowSamples += totalFrames;
    this.strideBlockCount += 1;
    this.blockCount += 1;

    if (this.strideBlockCount >= TELEMETRY_STRIDE_BLOCKS && canEmit) {
      const strideMeanUs = this.windowSumUs / this.strideBlockCount;
      if (strideMeanUs > d1BudgetUs) {
        this.strideOverruns += 1;
        if (!perfAvailable) {
          this.totalOverruns += 1;
        }
      }
      postOutbound({
        type: 'telemetry',
        activeVoices,
        peakL: this.windowPeakL,
        peakR: this.windowPeakR,
        rmsMono: this.windowSamples > 0 ? Math.sqrt(this.windowSqSum / this.windowSamples) : 0,
        callbackUs: strideMeanUs,
        maxCallbackUs: this.windowMaxUs,
        p50Us: 0,
        p99Us: 0,
        overrunCount: this.totalOverruns,
        strideOverruns: this.strideOverruns,
        singleBlockOverruns: this.singleBlockOverruns,
        stallCount: this.stallCount,
        timingSource: perfAvailable ? 'perf' : 'date',
        clipped: this.windowClipped,
        blockCount: this.blockCount,
      });
      this.strideBlockCount = 0;
      this.windowSumUs = 0;
      this.windowMaxUs = 0;
      this.windowPeakL = 0;
      this.windowPeakR = 0;
      this.windowSqSum = 0;
      this.windowSamples = 0;
      this.windowClipped = false;
    }
  }
}
