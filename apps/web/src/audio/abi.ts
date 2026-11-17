/** Shared C-ABI, Prep Worker, and AudioWorklet IPC types for piano-wasm <-> apps/web. */

export const WORKLET_PROCESSOR_NAME = 'piano-wasm-processor';
export const MAX_RENDER_FRAMES = 512;
export const TELEMETRY_FLOATS = 8;
export const TELEMETRY_INTERVAL_BLOCKS = 32;
export const WEB_AUDIO_SAMPLE_RATE = 48_000;
export const SPL_CALIBRATION_OFFSET_DB = 100.0; // -18 dBFS == 82.0 dB SPL

export const PEDAL_SUSTAIN = 0 as const;
export const PEDAL_UNA_CORDA = 1 as const;
export const PEDAL_SOSTENUTO = 2 as const;

export type PedalKind =
  | typeof PEDAL_SUSTAIN
  | typeof PEDAL_UNA_CORDA
  | typeof PEDAL_SOSTENUTO;

export const PERSPECTIVE_PLAYER = 0 as const;
export const PERSPECTIVE_CLOSE = 1 as const;
export const PERSPECTIVE_HALL = 2 as const;
export const PERSPECTIVE_BINAURAL = 3 as const;

export type ListenerPerspective =
  | typeof PERSPECTIVE_PLAYER
  | typeof PERSPECTIVE_CLOSE
  | typeof PERSPECTIVE_HALL
  | typeof PERSPECTIVE_BINAURAL;

export const LID_CLOSED = 0 as const;
export const LID_HALF = 1 as const;
export const LID_OPEN = 2 as const;

export type LidState =
  | typeof LID_CLOSED
  | typeof LID_HALF
  | typeof LID_OPEN;

export const HAMMER_HARDNESS_MIN = 0.0;
export const HAMMER_HARDNESS_MAX = 1.0;
export const HAMMER_HARDNESS_DEFAULT = 0.5;

export const UNISON_DETUNE_MIN = 0.0;
export const UNISON_DETUNE_MAX = 4.0;
export const UNISON_DETUNE_DEFAULT = 1.0;

export const STRETCH_TUNING_MIN = 0.0;
export const STRETCH_TUNING_MAX = 2.0;
export const STRETCH_TUNING_DEFAULT = 1.0;

export const STRING_SCALE_MIN = 0.5;
export const STRING_SCALE_MAX = 2.0;
export const STRING_SCALE_DEFAULT = 1.0;

export type EngineStatus = 'locked' | 'initializing' | 'ready' | 'error';

export interface PianoWasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;
  piano_init(sampleRate: number): number;
  piano_destroy(ctx: number): void;
  piano_note_on(ctx: number, midiKey: number, velocityU16: number, sampleOffset: number): void;
  piano_note_off(ctx: number, midiKey: number, releaseVelU16: number, sampleOffset: number): void;
  piano_pedal(ctx: number, pedalKind: number, valueU16: number, sampleOffset: number): void;
  piano_all_notes_off(ctx: number): void;
  piano_set_voicing(
    ctx: number,
    hammerHardness: number,
    unisonDetune: number,
    stretch: number,
    lengthScale: number,
  ): number;
  piano_set_hammer_hardness(ctx: number, hammerHardness: number): number;
  piano_set_space(ctx: number, perspective: number, lid: number): number;
  piano_staging_ptr(ctx: number, lenF32: number): number;
  piano_staging_offset_ptr(ctx: number, offF32: number, lenF32: number): number;
  piano_stage_space_chunk(
    ctx: number,
    perspective: number,
    lid: number,
    blobOff: number,
    chunkLen: number,
    totalLen: number,
  ): number;
  piano_commit_space(ctx: number, perspective: number, lid: number): number;
  piano_is_space_crossfading(ctx: number): number;
  piano_apply_space_blob(ctx: number, perspective: number, lid: number, lenF32: number): number;
  piano_apply_voicing_blob(
    ctx: number,
    hammerHardness: number,
    unisonDetune: number,
    stretch: number,
    lengthScale: number,
    lenF32: number,
  ): number;
  piano_render(ctx: number, frames: number): void;
  piano_out_l_ptr(ctx: number): number;
  piano_out_r_ptr(ctx: number): number;
  piano_telemetry_ptr(ctx: number): number;
}

export interface PianoPrepWasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;
  piano_prep_init(sampleRate: number): number;
  piano_prep_destroy(prep: number): void;
  piano_prep_space(prep: number, perspective: number, lid: number): number;
  piano_prep_voicing(
    prep: number,
    hammerHardness: number,
    unisonDetune: number,
    stretch: number,
    lengthScale: number,
  ): number;
  piano_prep_out_ptr(prep: number): number;
}

export type WorkletInboundMessage =
  | { type: 'init'; wasmModule?: WebAssembly.Module; wasmBytes?: ArrayBuffer; prepPort?: MessagePort }
  | { type: 'noteOn'; midiKey: number; velocityU16: number; sampleOffset: number }
  | { type: 'noteOff'; midiKey: number; releaseVelU16: number; sampleOffset: number }
  | { type: 'pedal'; pedalKind: PedalKind; valueU16: number; sampleOffset: number }
  | { type: 'allNotesOff' }
  | { type: 'setHammer'; hammerHardness: number }
  | {
      type: 'setVoicing';
      hammerHardness: number;
      unisonDetune: number;
      stretchTuning: number;
      stringLengthScale: number;
    }
  | { type: 'setSpace'; perspective: ListenerPerspective; lidState: LidState }
  | { type: 'destroy' };

export type PrepInboundMessage =
  | {
      type: 'init';
      wasmModule?: WebAssembly.Module;
      wasmBytes?: ArrayBuffer;
      sampleRate: number;
      workletPort: MessagePort;
    }
  | { type: 'prepSpace'; perspective: ListenerPerspective; lidState: LidState; seq: number }
  | {
      type: 'prepVoicing';
      hammerHardness: number;
      unisonDetune: number;
      stretchTuning: number;
      stringLengthScale: number;
      seq: number;
    }
  | { type: 'destroy' };

export type PrepToWorkletMessage =
  | {
      type: 'spaceBlob';
      perspective: ListenerPerspective;
      lidState: LidState;
      seq: number;
      blob: ArrayBuffer;
    }
  | {
      type: 'voicingBlob';
      hammerHardness: number;
      unisonDetune: number;
      stretchTuning: number;
      stringLengthScale: number;
      seq: number;
      blob: ArrayBuffer;
    };

export type PrepOutboundMessage =
  | { type: 'ready' }
  | { type: 'spaceDone'; seq: number }
  | { type: 'voicingDone'; seq: number }
  | { type: 'error'; message: string };

export interface TelemetrySnapshot {
  readonly activeVoices: number;
  readonly peakL: number;
  readonly peakR: number;
  readonly rmsMono: number;
  readonly callbackUs: number;
  readonly maxCallbackUs: number;
  readonly p50Us: number;
  readonly p99Us: number;
  readonly overrunCount: number;
  readonly clipped: boolean;
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
  readonly peakHold: number;
  readonly peakHoldDbfs: number;
  readonly rmsDbfs: number;
  readonly splDb: number;
  readonly clipLatched: boolean;
  readonly sessionMaxCallbackUs: number;
  readonly p50Us: number;
  readonly p99Us: number;
  readonly error: string | null;
}

/** Maps a 7-bit MIDI value (`0..=127`) linearly onto 16-bit unsigned (`0..=65535`). */
export function u7ToU16(v: number): number {
  const clamped = Number.isFinite(v) ? Math.max(0, Math.min(127, Math.trunc(v))) : 0;
  if (clamped === 0) return 0;
  return Math.trunc((clamped * 65535) / 127);
}

/** Maps a 7-bit MIDI continuous controller value (`0..=127`) linearly onto `0..=65535`. */
export function cc7ToU16(v: number): number {
  const clamped = Number.isFinite(v) ? Math.max(0, Math.min(127, Math.trunc(v))) : 0;
  return Math.trunc((clamped * 65535) / 127);
}
