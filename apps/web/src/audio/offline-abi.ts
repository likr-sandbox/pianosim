/** Message contracts and parameter types for the Offline MIDI Render Web Worker. */

import type {
  BodyId,
  LidState,
  ListenerPerspective,
  PerfProfileConfig,
  TemperamentId,
} from './abi';
import type { MidiPlaybackEvent } from '../midi/smf-parser';

export type OfflineRenderStatus =
  | 'idle'
  | 'rendering'
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
}

export type OfflineWorkerInbound =
  | {
      type: 'render';
      wasmModule?: WebAssembly.Module;
      wasmBytes?: ArrayBuffer;
      request: OfflineRenderRequest;
    }
  | { type: 'cancel'; jobId: number };

export type OfflineWorkerOutbound =
  | {
      type: 'progress';
      jobId: number;
      progress: number;
      renderedSec: number;
      totalSec: number;
    }
  | {
      type: 'done';
      jobId: number;
      sampleRate: number;
      totalFrames: number;
      left: ArrayBuffer;
      right: ArrayBuffer;
    }
  | { type: 'cancelled'; jobId: number }
  | { type: 'error'; jobId: number; message: string };
