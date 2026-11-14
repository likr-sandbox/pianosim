import type { EngineStatus, PianoDiagnostics, TelemetrySnapshot } from './abi';

const EMPTY_TELEMETRY: TelemetrySnapshot = {
  activeVoices: 0,
  peakL: 0,
  peakR: 0,
  rmsMono: 0,
  callbackUs: 0,
  maxCallbackUs: 0,
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
    this.telemetry = snapshot;
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
