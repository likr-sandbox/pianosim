/** Wasm export validation, PerfProfile builder init, Active-12 JIT warmup, and state restoration. */

import {
  A4_FREQ_DEFAULT,
  BODY_CONCERT_GRAND,
  CONDITION_DEFAULT,
  encodeKnobValue,
  HAMMER_HARDNESS_DEFAULT,
  LID_OPEN,
  PERF_KNOB_SPECS,
  PERSPECTIVE_PLAYER,
  REVERB_DECAY_DEFAULT,
  REVERB_MIX_DEFAULT,
  STRING_SCALE_DEFAULT,
  STRETCH_TUNING_DEFAULT,
  TEMPERAMENT_EQUAL,
  UNISON_DETUNE_DEFAULT,
  type CachedWorkletState,
  type PerfProfileConfig,
  type PianoWasmExports,
} from './abi';
import { warmPrepInstallers } from './blob-streamer';

const ACTIVE12_WARMUP_KEYS = [21, 28, 36, 43, 48, 55, 60, 64, 67, 76, 88, 100];

export const REQUIRED_EXPORTS: ReadonlyArray<keyof PianoWasmExports> = [
  'memory',
  'piano_init',
  'piano_init_with_profile',
  'piano_init_with_builder',
  'piano_perf_builder_new',
  'piano_perf_builder_set',
  'piano_perf_builder_destroy',
  'piano_set_perf_preset',
  'piano_set_perf_knob',
  'piano_destroy',
  'piano_note_on',
  'piano_note_off',
  'piano_pedal',
  'piano_all_notes_off',
  'piano_set_voicing',
  'piano_set_voicing_full',
  'piano_set_concert_pitch',
  'piano_set_temperament',
  'piano_set_condition',
  'piano_set_hammer_hardness',
  'piano_set_space',
  'piano_set_reverb',
  'piano_set_body_space',
  'piano_staging_ptr',
  'piano_staging_offset_ptr',
  'piano_stage_space_chunk',
  'piano_stage_body_space_chunk',
  'piano_commit_space',
  'piano_commit_body_space',
  'piano_is_space_crossfading',
  'piano_apply_space_blob',
  'piano_apply_voicing_blob',
  'piano_apply_voicing_blob_ex',
  'piano_render',
  'piano_out_l_ptr',
  'piano_out_r_ptr',
  'piano_telemetry_ptr',
];

export function validateWasmExports(raw: WebAssembly.Exports): PianoWasmExports {
  for (const name of REQUIRED_EXPORTS) {
    const val = raw[name];
    if (name === 'memory' ? !(val instanceof WebAssembly.Memory) : typeof val !== 'function') {
      throw new Error(`Missing or invalid Wasm export: ${name}`);
    }
  }
  return raw as PianoWasmExports;
}

export function buildEngineContext(
  exports: PianoWasmExports,
  sr: number,
  perfConfig: PerfProfileConfig,
): number {
  const builder = exports.piano_perf_builder_new(0) >>> 0;
  if (builder === 0) {
    throw new Error('piano_perf_builder_new returned null pointer');
  }
  let ctx = 0;
  try {
    for (const spec of PERF_KNOB_SPECS) {
      const rc = exports.piano_perf_builder_set(
        builder,
        spec.knobId,
        encodeKnobValue(spec.key, perfConfig),
      );
      if (rc !== 0) {
        throw new Error(`piano_perf_builder_set(${spec.knobId}) failed with ${rc}`);
      }
    }
    ctx = exports.piano_init_with_builder(sr, builder) >>> 0;
  } finally {
    exports.piano_perf_builder_destroy(builder);
  }
  if (ctx === 0) {
    throw new Error('piano_init_with_builder returned a null context pointer');
  }
  return ctx;
}

export function applyRuntimePerfKnobs(
  exports: PianoWasmExports,
  ctx: number,
  perfConfig: PerfProfileConfig,
): void {
  for (const spec of PERF_KNOB_SPECS) {
    if (spec.knobId <= 9) {
      exports.piano_set_perf_knob(ctx, spec.knobId, encodeKnobValue(spec.key, perfConfig));
    }
  }
}

export function runActive12Warmup(
  raw: WebAssembly.Exports,
  exports: PianoWasmExports,
  ctx: number,
  sr: number,
  syncMemForBlob: () => Float32Array,
  perfConfig: PerfProfileConfig,
): void {
  // Active-12 full-bucket JIT warmup covering open Sustain, SpaceCrossfade, hammer
  // dirty mask, sustained dry A4, and Prep staging/voicing installers before ready.
  exports.piano_pedal(ctx, 0, 40_000, 0);
  for (const midi of ACTIVE12_WARMUP_KEYS) exports.piano_note_on(ctx, midi, 42_000, 0);
  for (let i = 0; i < 120; i++) exports.piano_render(ctx, 128);
  exports.piano_set_body_space(ctx, 1, 1, 1);
  exports.piano_set_reverb(ctx, 2.0, 0.25);
  for (let i = 0; i < 4; i++) exports.piano_render(ctx, 128);
  exports.piano_set_body_space(ctx, BODY_CONCERT_GRAND, 0, 2);
  exports.piano_set_reverb(ctx, REVERB_DECAY_DEFAULT, REVERB_MIX_DEFAULT);
  exports.piano_set_hammer_hardness(ctx, 0.65);
  for (let i = 0; i < 32; i++) exports.piano_render(ctx, 128);
  exports.piano_set_hammer_hardness(ctx, 0.5);
  exports.piano_set_temperament(ctx, 1);
  exports.piano_set_condition(ctx, 0.25);
  exports.piano_set_concert_pitch(ctx, 415.0);
  for (let i = 0; i < 4; i++) exports.piano_render(ctx, 128);
  exports.piano_set_voicing_full(ctx, 0.5, 1.0, 1.0, 1.0, 440.0, 0, 0.0);
  for (const midi of ACTIVE12_WARMUP_KEYS) exports.piano_note_off(ctx, midi, 32_768, 0);
  exports.piano_pedal(ctx, 0, 21_671, 0);
  for (let i = 0; i < 24; i++) exports.piano_render(ctx, 128);
  exports.piano_pedal(ctx, 0, 23_220, 0);
  for (let i = 0; i < 12; i++) exports.piano_render(ctx, 128);
  exports.piano_pedal(ctx, 0, 0, 0);
  for (let i = 0; i < 24; i++) exports.piano_render(ctx, 128);
  exports.piano_pedal(ctx, 0, 30_000, 0);
  for (let i = 0; i < 8; i++) exports.piano_render(ctx, 128);
  exports.piano_pedal(ctx, 0, 0, 0);
  exports.piano_all_notes_off(ctx);
  for (let i = 0; i < 16; i++) exports.piano_render(ctx, 128);
  exports.piano_note_on(ctx, 69, 48_000, 0);
  for (let i = 0; i < 160; i++) exports.piano_render(ctx, 128);
  exports.piano_all_notes_off(ctx);
  for (let i = 0; i < 8; i++) exports.piano_render(ctx, 128);
  warmPrepInstallers(raw, exports, ctx, sr, syncMemForBlob, perfConfig);
}

export function defaultCachedWorkletState(): CachedWorkletState {
  return {
    bodyId: BODY_CONCERT_GRAND,
    perspective: PERSPECTIVE_PLAYER,
    lidState: LID_OPEN,
    reverbDecayS: REVERB_DECAY_DEFAULT,
    reverbMix: REVERB_MIX_DEFAULT,
    hammerHardness: HAMMER_HARDNESS_DEFAULT,
    unisonDetune: UNISON_DETUNE_DEFAULT,
    stretchTuning: STRETCH_TUNING_DEFAULT,
    stringLengthScale: STRING_SCALE_DEFAULT,
    a4FreqHz: A4_FREQ_DEFAULT,
    temperamentId: TEMPERAMENT_EQUAL,
    condition: CONDITION_DEFAULT,
  };
}

export function restoreCachedEngineState(
  exports: PianoWasmExports,
  ctx: number,
  st: CachedWorkletState,
): void {
  exports.piano_set_body_space(ctx, st.bodyId, st.perspective, st.lidState);
  exports.piano_set_reverb(ctx, st.reverbDecayS, st.reverbMix);
  exports.piano_set_voicing_full(
    ctx,
    st.hammerHardness,
    st.unisonDetune,
    st.stretchTuning,
    st.stringLengthScale,
    st.a4FreqHz,
    st.temperamentId,
    st.condition,
  );
  exports.piano_render(ctx, 128);
}
