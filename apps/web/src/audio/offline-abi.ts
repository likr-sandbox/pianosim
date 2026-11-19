/** Message contracts and parameter types for the Offline MIDI Render Web Worker. */

import {
  A4_FREQ_DEFAULT,
  BODY_CONCERT_GRAND,
  CONDITION_DEFAULT,
  HAMMER_HARDNESS_DEFAULT,
  LID_OPEN,
  PERSPECTIVE_PLAYER,
  REVERB_DECAY_DEFAULT,
  REVERB_MIX_DEFAULT,
  STRING_SCALE_DEFAULT,
  STRETCH_TUNING_DEFAULT,
  TEMPERAMENT_EQUAL,
  UNISON_DETUNE_DEFAULT,
  type BodyId,
  type LidState,
  type ListenerPerspective,
  type PerfProfileConfig,
  type TemperamentId,
} from './abi';
import type { MidiPlaybackEvent } from '../midi/smf-parser';

export type OfflineRenderStatus =
  | 'idle'
  | 'rendering'
  | 'buffering'
  | 'ready'
  | 'playing'
  | 'paused'
  | 'error';

export interface OfflineVoicingParams {
  readonly hammerHardness: number;
  readonly unisonDetune: number;
  readonly stretchTuning: number;
  readonly stringLengthScale: number;
  readonly a4FreqHz: number;
  readonly temperamentId: TemperamentId;
  readonly condition: number;
}

export interface OfflineSpaceParams {
  readonly bodyId: BodyId;
  readonly perspective: ListenerPerspective;
  readonly lidState: LidState;
  readonly reverbDecayS: number;
  readonly reverbMix: number;
}

export interface OfflineRenderRequest {
  readonly jobId: number;
  readonly scoreKey: string;
  readonly events: ReadonlyArray<MidiPlaybackEvent>;
  readonly durationSeconds: number;
  readonly voicing: OfflineVoicingParams;
  readonly space: OfflineSpaceParams;
  readonly perfConfig: PerfProfileConfig;
  readonly firstChunkSec?: number;
  readonly chunkSec?: number;
}

export type OfflineWorkerInbound =
  | {
      type: 'render';
      wasmModule?: WebAssembly.Module;
      wasmBytes?: ArrayBuffer;
      request: OfflineRenderRequest;
    }
  | { type: 'cancel'; jobId: number };

export interface OfflineChunkPayload {
  readonly jobId: number;
  readonly chunkIndex: number;
  readonly startFrame: number;
  readonly frames: number;
  readonly sampleRate: number;
  readonly totalFrames: number;
  readonly isFinal: boolean;
  readonly left: ArrayBuffer;
  readonly right: ArrayBuffer;
}

export type OfflineWorkerOutbound =
  | {
      type: 'progress';
      jobId: number;
      progress: number;
      renderedSec: number;
      totalSec: number;
    }
  | ({ type: 'chunk' } & OfflineChunkPayload)
  | { type: 'cancelled'; jobId: number }
  | { type: 'error'; jobId: number; message: string };

export const DEFAULT_OFFLINE_VOICING: OfflineVoicingParams = {
  hammerHardness: HAMMER_HARDNESS_DEFAULT,
  unisonDetune: UNISON_DETUNE_DEFAULT,
  stretchTuning: STRETCH_TUNING_DEFAULT,
  stringLengthScale: STRING_SCALE_DEFAULT,
  a4FreqHz: A4_FREQ_DEFAULT,
  temperamentId: TEMPERAMENT_EQUAL,
  condition: CONDITION_DEFAULT,
};

export const DEFAULT_OFFLINE_SPACE: OfflineSpaceParams = {
  bodyId: BODY_CONCERT_GRAND,
  perspective: PERSPECTIVE_PLAYER,
  lidState: LID_OPEN,
  reverbDecayS: REVERB_DECAY_DEFAULT,
  reverbMix: REVERB_MIX_DEFAULT,
};
