/** 3-tier PerfProfile synchronization across WasmAudioHost, AudioWorklet, and PrepWorker. */

import {
  classifyPerfChange,
  clonePerfConfig,
  encodeKnobValue,
  PERF_KNOB_ID,
  PERF_PRESET_CONFIGS,
  resolvePerfPresetFromUrl,
  type CachedWorkletState,
  type PerfKnobKey,
  type PerfPresetId,
  type PerfProfileConfig,
  type PrepInboundMessage,
  type WorkletInboundMessage,
} from './abi';
import type { TelemetryClient } from './telemetry-client';

export interface PerfSyncHooks {
  readonly isReady: () => boolean;
  readonly postToWorklet: (msg: WorkletInboundMessage) => void;
  readonly postToPrep: (msg: PrepInboundMessage) => void;
  readonly getCachedState: () => CachedWorkletState;
  readonly bumpMinSeqs: () => { minSpaceSeq: number; minVoicingSeq: number };
  readonly refreshSpaceBlob: () => void;
  readonly refreshNonDefaultVoicingIfNeeded: () => void;
}

export class PerfSyncController {
  private readonly telemetry: TelemetryClient;
  private perfPreset: PerfPresetId;
  private perfConfig: PerfProfileConfig;
  private perfSeq = 0;

  constructor(telemetry: TelemetryClient, initialPreset?: PerfPresetId) {
    this.telemetry = telemetry;
    this.perfPreset = initialPreset ?? resolvePerfPresetFromUrl();
    this.perfConfig = clonePerfConfig(PERF_PRESET_CONFIGS[this.perfPreset]);
    this.telemetry.setInitialPerf(this.perfPreset, this.perfConfig);
  }

  getPreset(): PerfPresetId {
    return this.perfPreset;
  }

  getConfig(): PerfProfileConfig {
    return this.perfConfig;
  }

  resetSeq(): void {
    this.perfSeq = 0;
  }

  setPreset(presetId: PerfPresetId, hooks: PerfSyncHooks): void {
    const nextConfig = clonePerfConfig(PERF_PRESET_CONFIGS[presetId]);
    this.applyProfile(presetId, nextConfig, hooks);
  }

  setKnob<K extends PerfKnobKey>(
    key: K,
    value: PerfProfileConfig[K],
    hooks: PerfSyncHooks,
  ): void {
    const nextConfig: PerfProfileConfig = { ...this.perfConfig, [key]: value };
    const tier = classifyPerfChange(this.perfConfig, nextConfig);
    if (tier === 'tier1' && hooks.isReady()) {
      this.perfConfig = clonePerfConfig(nextConfig);
      const seq = ++this.perfSeq;
      this.telemetry.markPerfRequested(seq, this.perfPreset, this.perfConfig);
      hooks.postToPrep({ type: 'setPerfProfile', perfConfig: this.perfConfig });
      hooks.postToWorklet({
        type: 'setPerfKnob',
        knobId: PERF_KNOB_ID[key],
        value: encodeKnobValue(key, this.perfConfig),
        presetId: this.perfPreset,
        perfConfig: this.perfConfig,
        seq,
      });
      return;
    }
    this.applyProfile(this.perfPreset, nextConfig, hooks);
  }

  applyProfile(
    presetId: PerfPresetId,
    nextConfig: PerfProfileConfig,
    hooks: PerfSyncHooks,
  ): void {
    const prev = this.perfConfig;
    const tier = classifyPerfChange(prev, nextConfig);
    this.perfPreset = presetId;
    this.perfConfig = clonePerfConfig(nextConfig);

    if (!hooks.isReady()) {
      this.telemetry.setInitialPerf(this.perfPreset, this.perfConfig);
      return;
    }

    const seq = ++this.perfSeq;
    this.telemetry.markPerfRequested(seq, this.perfPreset, this.perfConfig);

    if (tier === 'none') {
      this.telemetry.updatePerfApplied(seq, this.perfPreset, this.perfConfig);
      return;
    }

    hooks.postToPrep({ type: 'setPerfProfile', perfConfig: this.perfConfig });

    if (tier === 'tier3') {
      const { minSpaceSeq, minVoicingSeq } = hooks.bumpMinSeqs();
      hooks.postToWorklet({
        type: 'setPerfProfile',
        presetId: this.perfPreset,
        perfConfig: this.perfConfig,
        seq,
        rebuildRequired: true,
        minSpaceSeq,
        minVoicingSeq,
        cachedState: hooks.getCachedState(),
      });
      hooks.refreshNonDefaultVoicingIfNeeded();
      return;
    }

    hooks.postToWorklet({
      type: 'setPerfProfile',
      presetId: this.perfPreset,
      perfConfig: this.perfConfig,
      seq,
      rebuildRequired: false,
    });

    if (tier === 'tier2') {
      hooks.refreshSpaceBlob();
    }
  }
}
