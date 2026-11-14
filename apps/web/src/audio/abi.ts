/** Shared C-ABI and AudioWorklet IPC types for piano-wasm <-> apps/web. */

export const WORKLET_PROCESSOR_NAME = 'piano-wasm-processor';
export const MAX_RENDER_FRAMES = 512;
export const TELEMETRY_FLOATS = 8;
export const TELEMETRY_INTERVAL_BLOCKS = 16;

export const PEDAL_SUSTAIN = 0 as const;
export const PEDAL_UNA_CORDA = 1 as const;
export const PEDAL_SOSTENUTO = 2 as const;

export type PedalKind =
  | typeof PEDAL_SUSTAIN
  | typeof PEDAL_UNA_CORDA
  | typeof PEDAL_SOSTENUTO;

export type EngineStatus = 'locked' | 'initializing' | 'ready' | 'error';

export interface PianoWasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;
  piano_init(sampleRate: number): number;
  piano_destroy(ctx: number): void;
  piano_note_on(ctx: number, midiKey: number, velocityU16: number, sampleOffset: number): void;
  piano_note_off(ctx: number, midiKey: number, releaseVelU16: number, sampleOffset: number): void;
  piano_pedal(ctx: number, pedalKind: number, valueU16: number, sampleOffset: number): void;
  piano_all_notes_off(ctx: number): void;
  piano_render(ctx: number, frames: number): void;
  piano_out_l_ptr(ctx: number): number;
  piano_out_r_ptr(ctx: number): number;
  piano_telemetry_ptr(ctx: number): number;
}

export type WorkletInboundMessage =
  | { type: 'init'; wasmBytes: ArrayBuffer }
  | { type: 'noteOn'; midiKey: number; velocityU16: number; sampleOffset: number }
  | { type: 'noteOff'; midiKey: number; releaseVelU16: number; sampleOffset: number }
  | { type: 'pedal'; pedalKind: PedalKind; valueU16: number; sampleOffset: number }
  | { type: 'allNotesOff' }
  | { type: 'destroy' };

export interface TelemetrySnapshot {
  readonly activeVoices: number;
  readonly peakL: number;
  readonly peakR: number;
  readonly rmsMono: number;
  readonly callbackUs: number;
  readonly maxCallbackUs: number;
  readonly blockCount: number;
}

export type WorkletOutboundMessage =
  | { type: 'ready'; sampleRate: number }
  | ({ type: 'telemetry' } & TelemetrySnapshot)
  | { type: 'error'; message: string };

export interface PianoDiagnostics {
  readonly status: EngineStatus;
  readonly sampleRate: number;
  readonly contextState: AudioContextState | 'none';
  readonly audioContext: AudioContext | null;
  readonly analyser: AnalyserNode | null;
  readonly workletNode: AudioWorkletNode | null;
  readonly telemetry: TelemetrySnapshot;
  readonly telemetryBlocksReceived: number;
  readonly error: string | null;
}

/** Maps a 7-bit MIDI value (`0..=127`) onto 16-bit unsigned (`0..=65535`). */
export function u7ToU16(v: number): number {
  const clamped = Math.max(0, Math.min(127, Math.trunc(v)));
  return Math.trunc((clamped * 65535) / 127);
}
