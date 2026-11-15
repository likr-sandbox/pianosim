import type { EngineStatus, PianoDiagnostics, TelemetrySnapshot } from './abi';

const EMPTY_TELEMETRY: TelemetrySnapshot = {
  activeVoices: 0,
  peakL: 0,
  peakR: 0,
  rmsMono: 0,
  callbackUs: 0,
  maxCallbackUs: 0,
  p50Us: 0,
  p99Us: 0,
  blockCount: 0,
};

export type DiagnosticsListener = (diag: PianoDiagnostics) => void;

declare global {
  interface Window {
    __PIANO_DIAG__?: PianoDiagnostics;
  }
}

/** Tracks live AudioWorklet telemetry and updates `window.__PIANO_DIAG__`. */
export class TelemetryClient {
  /**
   * Capacity of rolling telemetry history (32 packets * 32 blocks * 2.67 ms = ~2.73 seconds at 48 kHz).
   * 32 snapshots provide sufficient smoothing while allowing attack transients to purge within ~2.7s.
   */
  private static readonly HISTORY_CAPACITY = 32;
  private readonly callbackHistory = new Float32Array(TelemetryClient.HISTORY_CAPACITY);
  private readonly sortScratch = new Float32Array(TelemetryClient.HISTORY_CAPACITY);
  private historyCount = 0;
  private historyIndex = 0;
  private p50Us = 0;
  private p99Us = 0;

  private status: EngineStatus = 'locked';
  private sampleRate = 0;
  private contextState: AudioContextState | 'none' = 'none';
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private telemetry: TelemetrySnapshot = EMPTY_TELEMETRY;
  private telemetryBlocksReceived = 0;
  private error: string | null = null;
  private readonly listeners = new Set<DiagnosticsListener>();

  constructor() {
    this.syncGlobalDiag();
  }

  subscribe(listener: DiagnosticsListener): () => void {
    this.listeners.add(listener);
    listener(this.getDiagnostics());
    return () => {
      this.listeners.delete(listener);
    };
  }

  getDiagnostics(): PianoDiagnostics {
    return {
      status: this.status,
      sampleRate: this.sampleRate,
      contextState: this.contextState,
      audioContext: this.audioContext,
      analyser: this.analyser,
      workletNode: this.workletNode,
      telemetry: this.telemetry,
      telemetryBlocksReceived: this.telemetryBlocksReceived,
      p50Us: this.p50Us,
      p99Us: this.p99Us,
      error: this.error,
    };
  }

  setStatus(status: EngineStatus, error: string | null = null): void {
    this.status = status;
    this.error = error;
    this.emit();
  }

  setAudioGraph(
    audioContext: AudioContext | null,
    analyser: AnalyserNode | null,
    workletNode: AudioWorkletNode | null,
  ): void {
    this.audioContext = audioContext;
    this.analyser = analyser;
    this.workletNode = workletNode;
    this.sampleRate = audioContext ? audioContext.sampleRate : 0;
    this.contextState = audioContext ? audioContext.state : 'none';
    this.emit();
  }

  syncContextState(): void {
    this.contextState = this.audioContext ? this.audioContext.state : 'none';
    this.sampleRate = this.audioContext ? this.audioContext.sampleRate : 0;
    this.emit();
  }

  updateTelemetry(snapshot: TelemetrySnapshot): void {
    const us = snapshot.callbackUs;
    this.callbackHistory[this.historyIndex] = us;
    this.historyIndex = (this.historyIndex + 1) % TelemetryClient.HISTORY_CAPACITY;
    if (this.historyCount < TelemetryClient.HISTORY_CAPACITY) {
      this.historyCount += 1;
    }

    const n = this.historyCount;
    this.sortScratch.set(this.callbackHistory.subarray(0, n));
    const sorted = this.sortScratch.subarray(0, n);
    sorted.sort();

    // Median (50th percentile)
    this.p50Us = sorted[Math.floor(n * 0.5)];

    // 99th percentile with small-sample robustness:
    // When n >= 4, clamps to at most n - 2 to protect against a single anomalous CPU/OS interrupt spike.
    // For n < 4, uses the highest available sample (n - 1).
    let p99Idx = 0;
    if (n >= 4) {
      p99Idx = Math.min(n - 2, Math.max(0, Math.floor(0.99 * (n - 1))));
    } else if (n > 1) {
      p99Idx = n - 1;
    }
    this.p99Us = sorted[p99Idx];

    this.telemetry = {
      ...snapshot,
      p50Us: this.p50Us,
      p99Us: this.p99Us,
    };
    this.telemetryBlocksReceived += 1;
    this.emit();
  }

  reset(): void {
    this.status = 'locked';
    this.sampleRate = 0;
    this.contextState = 'none';
    this.audioContext = null;
    this.analyser = null;
    this.workletNode = null;
    this.telemetry = EMPTY_TELEMETRY;
    this.telemetryBlocksReceived = 0;
    this.historyCount = 0;
    this.historyIndex = 0;
    this.p50Us = 0;
    this.p99Us = 0;
    this.error = null;
    this.emit();
  }

  private syncGlobalDiag(): void {
    if (typeof window !== 'undefined') {
      window.__PIANO_DIAG__ = this.getDiagnostics();
    }
  }

  private emit(): void {
    const diag = this.getDiagnostics();
    if (typeof window !== 'undefined') {
      window.__PIANO_DIAG__ = diag;
    }
    for (const listener of this.listeners) {
      listener(diag);
    }
  }
}

export function formatPeakDb(peakL: number, peakR: number): string {
  const peak = Math.max(Math.abs(peakL), Math.abs(peakR));
  if (!Number.isFinite(peak)) {
    return 'NaN';
  }
  if (peak <= 1e-6) {
    return '-inf dB';
  }
  return `${(20 * Math.log10(peak)).toFixed(1)} dB`;
}
