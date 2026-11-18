import {
  SPL_CALIBRATION_OFFSET_DB,
  type EngineStatus,
  type PianoDiagnostics,
  type TelemetrySnapshot,
} from './abi';

const EMPTY_TELEMETRY: TelemetrySnapshot = {
  activeVoices: 0,
  peakL: 0,
  peakR: 0,
  rmsMono: 0,
  callbackUs: 0,
  maxCallbackUs: 0,
  p50Us: 0,
  p99Us: 0,
  overrunCount: 0,
  clipped: false,
  blockCount: 0,
};

const PEAK_HOLD_MS = 1_200;
const PEAK_DECAY_DB_PER_S = 18.0;

export type DiagnosticsListener = (diag: PianoDiagnostics) => void;

declare global {
  interface Window {
    __PIANO_DIAG__?: PianoDiagnostics;
  }
}

export function linearToDbfs(amp: number): number {
  if (!Number.isFinite(amp)) {
    return NaN;
  }
  if (amp <= 1e-6) {
    return -Infinity;
  }
  return 20.0 * Math.log10(amp);
}

/** Tracks live AudioWorklet telemetry, full-block peak hold / SPL, and honest p99 / max load. */
export class TelemetryClient {
  private static readonly HISTORY_CAPACITY = 32;
  private readonly callbackHistory = new Float32Array(TelemetryClient.HISTORY_CAPACITY);
  private readonly sortScratch = new Float32Array(TelemetryClient.HISTORY_CAPACITY);
  private historyCount = 0;
  private historyIndex = 0;
  private p50Us = 0;
  private p99Us = 0;
  private sessionMaxCallbackUs = 0;

  private peakHold = 0;
  private peakHoldStampMs = 0;
  private peakHoldDbfs = -Infinity;
  private rmsDbfs = -Infinity;
  private splDb = -Infinity;
  private clipLatched = false;

  private status: EngineStatus = 'locked';
  private sampleRate = 0;
  private contextState: AudioContextState | 'none' = 'none';
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private telemetry: TelemetrySnapshot = EMPTY_TELEMETRY;
  private telemetryBlocksReceived = 0;
  private requestedSpaceSeq = 0;
  private appliedSpaceSeq = 0;
  private requestedVoicingSeq = 0;
  private appliedVoicingSeq = 0;
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
      requestedSpaceSeq: this.requestedSpaceSeq,
      appliedSpaceSeq: this.appliedSpaceSeq,
      requestedVoicingSeq: this.requestedVoicingSeq,
      appliedVoicingSeq: this.appliedVoicingSeq,
      peakHold: this.peakHold,
      peakHoldDbfs: this.peakHoldDbfs,
      rmsDbfs: this.rmsDbfs,
      splDb: this.splDb,
      clipLatched: this.clipLatched,
      sessionMaxCallbackUs: this.sessionMaxCallbackUs,
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

  markSpaceRequested(seq: number): void {
    if (seq > this.requestedSpaceSeq) {
      this.requestedSpaceSeq = seq;
      this.syncGlobalDiag();
    }
  }

  markVoicingRequested(seq: number): void {
    if (seq > this.requestedVoicingSeq) {
      this.requestedVoicingSeq = seq;
      this.syncGlobalDiag();
    }
  }

  updatePrepApplied(spaceSeq: number, voicingSeq: number): void {
    let changed = false;
    if (spaceSeq > this.appliedSpaceSeq) {
      this.appliedSpaceSeq = spaceSeq;
      changed = true;
    }
    if (voicingSeq > this.appliedVoicingSeq) {
      this.appliedVoicingSeq = voicingSeq;
      changed = true;
    }
    if (changed) {
      this.syncGlobalDiag();
    }
  }

  clearClip(): void {
    this.clipLatched = false;
    this.peakHold = Math.max(Math.abs(this.telemetry.peakL), Math.abs(this.telemetry.peakR));
    this.peakHoldDbfs = linearToDbfs(this.peakHold);
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

    this.p50Us = sorted[Math.floor(n * 0.5)];
    // Honest nearest-rank 99th percentile without n - 2 masking.
    const p99Idx = n > 0 ? Math.min(n - 1, Math.max(0, Math.ceil(0.99 * n) - 1)) : 0;
    this.p99Us = sorted[p99Idx];

    if (snapshot.maxCallbackUs > this.sessionMaxCallbackUs) {
      this.sessionMaxCallbackUs = snapshot.maxCallbackUs;
    }

    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const blockPeak = Math.max(Math.abs(snapshot.peakL), Math.abs(snapshot.peakR));
    if (blockPeak >= this.peakHold) {
      this.peakHold = blockPeak;
      this.peakHoldStampMs = now;
    } else if (this.peakHold > 0 && now - this.peakHoldStampMs > PEAK_HOLD_MS) {
      const elapsedS = (now - this.peakHoldStampMs - PEAK_HOLD_MS) / 1000.0;
      const decayFactor = Math.pow(10.0, (-PEAK_DECAY_DB_PER_S * elapsedS) / 20.0);
      this.peakHold = Math.max(blockPeak, this.peakHold * decayFactor);
      if (this.peakHold <= 1e-6) {
        this.peakHold = 0;
      }
    }
    this.peakHoldDbfs = linearToDbfs(this.peakHold);
    this.rmsDbfs = linearToDbfs(snapshot.rmsMono);
    this.splDb = Number.isFinite(this.peakHoldDbfs)
      ? this.peakHoldDbfs + SPL_CALIBRATION_OFFSET_DB
      : -Infinity;
    if (snapshot.clipped || blockPeak >= 0.999) {
      this.clipLatched = true;
    }

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
    this.requestedSpaceSeq = 0;
    this.appliedSpaceSeq = 0;
    this.requestedVoicingSeq = 0;
    this.appliedVoicingSeq = 0;
    this.historyCount = 0;
    this.historyIndex = 0;
    this.p50Us = 0;
    this.p99Us = 0;
    this.sessionMaxCallbackUs = 0;
    this.peakHold = 0;
    this.peakHoldStampMs = 0;
    this.peakHoldDbfs = -Infinity;
    this.rmsDbfs = -Infinity;
    this.splDb = -Infinity;
    this.clipLatched = false;
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
  const db = linearToDbfs(peak);
  if (Number.isNaN(db)) {
    return 'NaN';
  }
  if (!Number.isFinite(db)) {
    return '-inf dB';
  }
  return `${db.toFixed(1)} dB`;
}

export function formatDbfs(db: number): string {
  if (Number.isNaN(db)) return 'NaN';
  if (!Number.isFinite(db)) return '-inf dBFS';
  return `${db.toFixed(1)} dBFS`;
}
