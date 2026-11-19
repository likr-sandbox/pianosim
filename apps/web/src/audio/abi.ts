/** Shared C-ABI, Prep Worker, and AudioWorklet IPC types for piano-wasm <-> apps/web. */

import type { PerfPresetId, PerfProfileConfig } from './perf-abi';

export * from './perf-abi';

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

export const BODY_CONCERT_GRAND = 0 as const;
export const BODY_UPRIGHT_STUDIO = 1 as const;
export const BODY_VINTAGE_SALOON = 2 as const;

export type BodyId =
  | typeof BODY_CONCERT_GRAND
  | typeof BODY_UPRIGHT_STUDIO
  | typeof BODY_VINTAGE_SALOON;

export interface BodyOption {
  readonly id: BodyId;
  readonly slug: string;
  readonly label: string;
  readonly testId: string;
}

export const BODY_OPTIONS: ReadonlyArray<BodyOption> = [
  { id: BODY_CONCERT_GRAND, slug: 'concert-grand', label: 'Concert Grand', testId: 'body-concert-grand' },
  { id: BODY_UPRIGHT_STUDIO, slug: 'upright-studio', label: 'Upright Studio', testId: 'body-upright-studio' },
  { id: BODY_VINTAGE_SALOON, slug: 'vintage-saloon', label: 'Vintage Saloon', testId: 'body-vintage-saloon' },
];

export const REVERB_DECAY_MIN = 0.0;
export const REVERB_DECAY_MAX = 3.2;
export const REVERB_DECAY_DEFAULT = 1.6;
export const REVERB_DECAY_STEP = 0.05;

export const REVERB_MIX_MIN = 0.0;
export const REVERB_MIX_MAX = 1.0;
export const REVERB_MIX_DEFAULT = 0.18;

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

export const A4_FREQ_MIN = 392.0;
export const A4_FREQ_MAX = 466.2;
export const A4_FREQ_DEFAULT = 440.0;
export const A4_FREQ_STEP = 0.1;

export const TEMPERAMENT_EQUAL = 0 as const;
export const TEMPERAMENT_WERCKMEISTER_III = 1 as const;
export const TEMPERAMENT_KIRNBERGER_III = 2 as const;
export const TEMPERAMENT_VALLOTTI = 3 as const;

export type TemperamentId =
  | typeof TEMPERAMENT_EQUAL
  | typeof TEMPERAMENT_WERCKMEISTER_III
  | typeof TEMPERAMENT_KIRNBERGER_III
  | typeof TEMPERAMENT_VALLOTTI;

export interface TemperamentOption {
  readonly id: TemperamentId;
  readonly label: string;
  readonly slug: string;
  readonly testId: string;
}

export const TEMPERAMENT_OPTIONS: ReadonlyArray<TemperamentOption> = [
  { id: TEMPERAMENT_EQUAL, label: 'Equal', slug: 'equal', testId: 'temperament-equal' },
  {
    id: TEMPERAMENT_WERCKMEISTER_III,
    label: 'Werckmeister III',
    slug: 'werckmeister-iii',
    testId: 'temperament-werckmeister-iii',
  },
  {
    id: TEMPERAMENT_KIRNBERGER_III,
    label: 'Kirnberger III',
    slug: 'kirnberger-iii',
    testId: 'temperament-kirnberger-iii',
  },
  { id: TEMPERAMENT_VALLOTTI, label: 'Vallotti', slug: 'vallotti', testId: 'temperament-vallotti' },
];

export const CONDITION_MIN = 0.0;
export const CONDITION_MAX = 1.0;
export const CONDITION_DEFAULT = 0.0;

export type WavFormat = 'pcm24' | 'f32';

export type EngineStatus = 'locked' | 'initializing' | 'ready' | 'error';

export interface PianoWasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;
  piano_init(sampleRate: number): number;
  piano_init_with_profile(sampleRate: number, presetId: number): number;
  piano_init_with_builder(sampleRate: number, builder: number): number;
  piano_destroy(ctx: number): void;
  piano_perf_builder_new(presetId: number): number;
  piano_perf_builder_set(builder: number, knobId: number, value: number): number;
  piano_perf_builder_destroy(builder: number): void;
  piano_set_perf_preset(ctx: number, presetId: number): number;
  piano_set_perf_knob(ctx: number, knobId: number, value: number): number;
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
    a4FreqHz: number,
  ): number;
  piano_set_voicing_full(
    ctx: number,
    hammerHardness: number,
    unisonDetune: number,
    stretch: number,
    lengthScale: number,
    concertPitchHz: number,
    temperamentId: number,
    condition: number,
  ): number;
  piano_set_concert_pitch(ctx: number, concertPitchHz: number): number;
  piano_set_temperament(ctx: number, temperamentId: number): number;
  piano_set_condition(ctx: number, condition: number): number;
  piano_set_hammer_hardness(ctx: number, hammerHardness: number): number;
  piano_set_space(ctx: number, perspective: number, lid: number): number;
  piano_set_reverb(ctx: number, decayS: number, mix: number): number;
  piano_set_body_space(ctx: number, bodyId: number, perspective: number, lid: number): number;
  piano_space_chunk_ptr(ctx: number): number;
  piano_is_voicing_applying(ctx: number): number;
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
  piano_stage_body_space_chunk(
    ctx: number,
    bodyId: number,
    perspective: number,
    lid: number,
    blobOff: number,
    chunkLen: number,
    totalLen: number,
  ): number;
  piano_commit_space(ctx: number, perspective: number, lid: number): number;
  piano_commit_body_space(ctx: number, bodyId: number, perspective: number, lid: number): number;
  piano_is_space_crossfading(ctx: number): number;
  piano_apply_space_blob(ctx: number, perspective: number, lid: number, lenF32: number): number;
  piano_apply_voicing_blob(
    ctx: number,
    hammerHardness: number,
    unisonDetune: number,
    stretch: number,
    lengthScale: number,
    a4FreqHz: number,
    lenF32: number,
  ): number;
  piano_apply_voicing_blob_ex(
    ctx: number,
    hammerHardness: number,
    unisonDetune: number,
    stretch: number,
    lengthScale: number,
    concertPitchHz: number,
    temperamentId: number,
    condition: number,
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
  piano_prep_space_for_body(prep: number, bodyId: number, perspective: number, lid: number): number;
  piano_prep_voicing(
    prep: number,
    hammerHardness: number,
    unisonDetune: number,
    stretch: number,
    lengthScale: number,
    a4FreqHz: number,
  ): number;
  piano_prep_voicing_ex(
    prep: number,
    hammerHardness: number,
    unisonDetune: number,
    stretch: number,
    lengthScale: number,
    concertPitchHz: number,
    temperamentId: number,
    condition: number,
  ): number;
  piano_prep_out_ptr(prep: number): number;
  piano_prep_set_perf_preset(prep: number, presetId: number): number;
  piano_prep_set_stage4_profile(
    prep: number,
    irDurationSecs: number,
    spatialTailSplitTap: number,
    firHeadTaps: number,
    firBlockFft: number,
  ): number;
  piano_perf_builder_new(presetId: number): number;
  piano_perf_builder_set(builder: number, knobId: number, value: number): number;
  piano_perf_builder_destroy(builder: number): void;
  piano_prep_set_perf_builder(prep: number, builder: number): number;
}

export interface CachedWorkletState {
  readonly bodyId: BodyId;
  readonly perspective: ListenerPerspective;
  readonly lidState: LidState;
  readonly reverbDecayS: number;
  readonly reverbMix: number;
  readonly hammerHardness: number;
  readonly unisonDetune: number;
  readonly stretchTuning: number;
  readonly stringLengthScale: number;
  readonly a4FreqHz: number;
  readonly temperamentId: TemperamentId;
  readonly condition: number;
}

export type WorkletInboundMessage =
  | {
      type: 'init';
      wasmModule?: WebAssembly.Module;
      wasmBytes?: ArrayBuffer;
      prepPort?: MessagePort;
      perfPreset?: PerfPresetId;
      perfConfig?: PerfProfileConfig;
    }
  | {
      type: 'setPerfProfile';
      presetId: PerfPresetId;
      perfConfig: PerfProfileConfig;
      seq: number;
      rebuildRequired: boolean;
      minSpaceSeq?: number;
      minVoicingSeq?: number;
      cachedState?: CachedWorkletState;
    }
  | {
      type: 'setPerfKnob';
      knobId: number;
      value: number;
      presetId: PerfPresetId;
      perfConfig: PerfProfileConfig;
      seq: number;
    }
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
      a4FreqHz: number;
      temperamentId: TemperamentId;
      condition: number;
    }
  | { type: 'setConcertPitch'; concertPitchHz: number }
  | { type: 'setTemperament'; temperamentId: TemperamentId }
  | { type: 'setCondition'; condition: number }
  | { type: 'setSpace'; bodyId?: BodyId; perspective: ListenerPerspective; lidState: LidState }
  | { type: 'setReverb'; decayS: number; mix: number }
  | { type: 'setRecording'; active: boolean }
  | { type: 'recReturn'; samples: ArrayBuffer }
  | { type: 'destroy' };

export type PrepInboundMessage =
  | {
      type: 'init';
      wasmModule?: WebAssembly.Module;
      wasmBytes?: ArrayBuffer;
      sampleRate: number;
      workletPort: MessagePort;
      perfConfig?: PerfProfileConfig;
    }
  | { type: 'setPerfProfile'; perfConfig: PerfProfileConfig }
  | { type: 'prepSpace'; bodyId: BodyId; perspective: ListenerPerspective; lidState: LidState; seq: number }
  | {
      type: 'prepVoicing';
      hammerHardness: number;
      unisonDetune: number;
      stretchTuning: number;
      stringLengthScale: number;
      a4FreqHz: number;
      temperamentId: TemperamentId;
      condition: number;
      seq: number;
    }
  | { type: 'destroy' };

export type PrepToWorkletMessage =
  | {
      type: 'spaceBlob';
      bodyId: BodyId;
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
      a4FreqHz: number;
      temperamentId: TemperamentId;
      condition: number;
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
  readonly strideOverruns: number;
  readonly singleBlockOverruns: number;
  readonly stallCount: number;
  readonly timingSource: 'perf' | 'date';
  readonly clipped: boolean;
  readonly blockCount: number;
}

export type WorkletOutboundMessage =
  | { type: 'ready'; sampleRate: number }
  | ({ type: 'telemetry' } & TelemetrySnapshot)
  | { type: 'prepApplied'; spaceSeq: number; voicingSeq: number }
  | { type: 'perfApplied'; perfSeq: number; presetId: PerfPresetId; perfConfig: PerfProfileConfig }
  | { type: 'recChunk'; samples: ArrayBuffer; frames: number }
  | { type: 'recStopped' }
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
  readonly perfPreset: PerfPresetId;
  readonly perfConfig: PerfProfileConfig;
  readonly requestedPerfSeq: number;
  readonly appliedPerfSeq: number;
  readonly requestedSpaceSeq: number;
  readonly appliedSpaceSeq: number;
  readonly requestedVoicingSeq: number;
  readonly appliedVoicingSeq: number;
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
