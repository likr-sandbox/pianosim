/** Offline MIDI Render Worker controller and AudioBufferSourceNode playback engine. */

import {
  A4_FREQ_DEFAULT,
  BODY_CONCERT_GRAND,
  CONDITION_DEFAULT,
  HAMMER_HARDNESS_DEFAULT,
  LID_OPEN,
  PEDAL_SOSTENUTO,
  PEDAL_SUSTAIN,
  PEDAL_UNA_CORDA,
  PERF_PRESET_CONFIGS,
  PERF_PRESET_FULL,
  PERSPECTIVE_PLAYER,
  REVERB_DECAY_DEFAULT,
  REVERB_MIX_DEFAULT,
  STRING_SCALE_DEFAULT,
  STRETCH_TUNING_DEFAULT,
  TEMPERAMENT_EQUAL,
  UNISON_DETUNE_DEFAULT,
  type PerfProfileConfig,
  type WavFormat,
} from './abi';
import type { OfflineRenderStatus, OfflineSpaceParams, OfflineVoicingParams } from './offline-abi';
import { OfflineWorkerClient } from './offline-worker-client';
import type { WasmAudioHost } from './wasm-host';
import { createPlanarWavBlob } from './wav-recorder';
import type { ParsedSmf } from '../midi/smf-parser';

export interface OfflineRendererCallbacks {
  onNotePress?(midiKey: number, velocity: number): void;
  onNoteRelease?(midiKey: number): void;
  onPedalChange?(pedalKind: number, value7: number): void;
  onProgress?(currentSec: number, totalSec: number): void;
  onStateChange?(isPlaying: boolean): void;
  onRenderStatus?(status: OfflineRenderStatus, progress: number, error?: string | null): void;
  getVoicing?(): OfflineVoicingParams;
  getSpace?(): OfflineSpaceParams;
  getPerfConfig?(): PerfProfileConfig;
}

interface CachedOfflineRender {
  readonly cacheKey: string;
  readonly sampleRate: number;
  readonly totalFrames: number;
  readonly left: Float32Array;
  readonly right: Float32Array;
  audioBuffer: AudioBuffer | null;
}

const DEFAULT_VOICING: OfflineVoicingParams = {
  hammerHardness: HAMMER_HARDNESS_DEFAULT,
  unisonDetune: UNISON_DETUNE_DEFAULT,
  stretchTuning: STRETCH_TUNING_DEFAULT,
  stringLengthScale: STRING_SCALE_DEFAULT,
  a4FreqHz: A4_FREQ_DEFAULT,
  temperamentId: TEMPERAMENT_EQUAL,
  condition: CONDITION_DEFAULT,
};

const DEFAULT_SPACE: OfflineSpaceParams = {
  bodyId: BODY_CONCERT_GRAND,
  perspective: PERSPECTIVE_PLAYER,
  lidState: LID_OPEN,
  reverbDecayS: REVERB_DECAY_DEFAULT,
  reverbMix: REVERB_MIX_DEFAULT,
};

function defaultScoreKey(score: ParsedSmf): string {
  const first = score.events[0];
  const last = score.events[score.events.length - 1];
  return `smf:${score.events.length}:${score.durationSeconds.toFixed(4)}:${first ? first.tick : 0}:${last ? last.tick : 0}`;
}

export class OfflineRenderer {
  private readonly host: WasmAudioHost;
  private readonly workerClient = new OfflineWorkerClient();
  private callbacks: OfflineRendererCallbacks;
  private cached: CachedOfflineRender | null = null;

  private score: ParsedSmf | null = null;
  private scoreKey = '';
  private status: OfflineRenderStatus = 'idle';
  private renderProgress = 0;

  private isPlayingState = false;
  private isRenderingState = false;
  private operationEpoch = 0;

  private activeSource: AudioBufferSourceNode | null = null;
  private visualTimerId: number | null = null;
  private startAudioTime = 0;
  private startScoreTime = 0;
  private playheadSeconds = 0;
  private tempoFactor = 1.0;
  private nextEventIndex = 0;
  private readonly activeVisualNotes = new Set<number>();

  constructor(host: WasmAudioHost, callbacks: OfflineRendererCallbacks = {}) {
    this.host = host;
    this.callbacks = callbacks;
  }

  setCallbacks(callbacks: OfflineRendererCallbacks): void {
    this.callbacks = callbacks;
  }

  load(score: ParsedSmf, scoreKey?: string): void {
    this.operationEpoch++;
    this.cancelActiveRender();
    this.stopPlaybackInternal(true);
    this.score = score;
    this.scoreKey = scoreKey ?? defaultScoreKey(score);
    this.playheadSeconds = 0;
    this.nextEventIndex = 0;
    if (this.cached && this.cached.cacheKey !== this.computeCacheKey()) {
      this.cached = null;
    }
    this.setStatus(this.hasValidCache() ? 'ready' : 'idle', this.hasValidCache() ? 1 : 0);
    this.callbacks.onProgress?.(0, score.durationSeconds);
  }

  invalidateIfStale(): void {
    if (!this.cached || this.cached.cacheKey === this.computeCacheKey()) return;
    if (this.isRenderingState) this.cancelRender();
    if (!this.isPlayingState) {
      this.cached = null;
      this.setStatus('idle', 0);
    }
  }

  getScore(): ParsedSmf | null {
    return this.score;
  }

  getStatus(): OfflineRenderStatus {
    return this.status;
  }

  getProgress(): number {
    return this.renderProgress;
  }

  isPlaying(): boolean {
    return this.isPlayingState;
  }

  isRendering(): boolean {
    return this.isRenderingState;
  }

  hasValidCache(): boolean {
    return this.cached !== null && this.cached.cacheKey === this.computeCacheKey();
  }

  getCurrentTime(): number {
    return this.isPlayingState ? this.calculateCurrentScoreTime() : this.playheadSeconds;
  }

  getDuration(): number {
    return this.score?.durationSeconds ?? 0;
  }

  getTempo(): number {
    return this.tempoFactor;
  }

  async play(): Promise<void> {
    if (!this.score || this.isPlayingState || this.isRenderingState) return;
    const epoch = ++this.operationEpoch;
    await this.host.unlock();
    if (epoch !== this.operationEpoch || !this.score || this.isPlayingState) return;

    const targetKey = this.computeCacheKey();
    if (!this.cached || this.cached.cacheKey !== targetKey) {
      this.isRenderingState = true;
      this.setStatus('rendering', 0);
      try {
        const out = await this.workerClient.render(
          (jobId) => ({
            jobId,
            scoreKey: this.scoreKey,
            events: this.score!.events,
            durationSeconds: this.score!.durationSeconds,
            voicing: this.getVoicingParams(),
            space: this.getSpaceParams(),
            perfConfig: this.getOfflinePerfConfig(),
          }),
          {
            onProgress: (p) => {
              if (epoch === this.operationEpoch) this.setStatus('rendering', p);
            },
            onError: (msg) => {
              if (epoch === this.operationEpoch) this.setStatus('error', 0, msg);
            },
          },
        );
        this.isRenderingState = false;
        if (!out || epoch !== this.operationEpoch || !this.score) return;
        this.cached = {
          cacheKey: targetKey,
          sampleRate: out.sampleRate,
          totalFrames: out.totalFrames,
          left: out.left,
          right: out.right,
          audioBuffer: null,
        };
        this.setStatus('ready', 1);
      } catch (err) {
        this.isRenderingState = false;
        throw err;
      }
    }

    const ctx = this.host.getAudioContext();
    const analyser = this.host.getAnalyser();
    if (!ctx || !analyser || !this.cached) {
      throw new Error('AudioContext or AnalyserNode is unavailable.');
    }
    if (!this.cached.audioBuffer) {
      const buf = ctx.createBuffer(2, this.cached.totalFrames, this.cached.sampleRate);
      buf.copyToChannel(new Float32Array(this.cached.left), 0);
      buf.copyToChannel(new Float32Array(this.cached.right), 1);
      this.cached.audioBuffer = buf;
    }

    if (
      this.playheadSeconds >= this.score.durationSeconds ||
      this.playheadSeconds >= this.cached.audioBuffer.duration
    ) {
      this.playheadSeconds = 0;
      this.nextEventIndex = 0;
    }

    this.syncVisualStateAt(this.playheadSeconds);
    this.isPlayingState = true;
    this.setStatus('playing', 1);
    this.callbacks.onStateChange?.(true);
    this.startSourceAt(this.playheadSeconds);
    this.visualTimerId = window.setInterval(() => this.tickVisuals(), 16);
  }

  pause(): void {
    this.operationEpoch++;
    if (this.isRenderingState) {
      this.cancelActiveRender();
      this.setStatus(this.hasValidCache() ? 'ready' : 'idle', this.hasValidCache() ? 1 : 0);
      return;
    }
    if (!this.isPlayingState) return;
    this.playheadSeconds = this.calculateCurrentScoreTime();
    this.stopPlaybackInternal(false);
    this.setStatus('paused', 1);
  }

  stop(): void {
    this.operationEpoch++;
    this.cancelActiveRender();
    this.stopPlaybackInternal(true);
    this.playheadSeconds = 0;
    this.nextEventIndex = 0;
    this.callbacks.onProgress?.(0, this.score?.durationSeconds ?? 0);
    this.setStatus(this.hasValidCache() ? 'ready' : 'idle', this.hasValidCache() ? 1 : 0);
  }

  cancelRender(): void {
    this.operationEpoch++;
    if (this.isRenderingState) {
      this.cancelActiveRender();
      this.setStatus(this.hasValidCache() ? 'ready' : 'idle', this.hasValidCache() ? 1 : 0);
    }
  }

  seek(targetSeconds: number): void {
    if (!Number.isFinite(targetSeconds)) return;
    const dur = this.score?.durationSeconds ?? 0;
    const clamped = Math.max(0, Math.min(dur, targetSeconds));
    this.playheadSeconds = clamped;
    this.syncVisualStateAt(clamped);
    this.callbacks.onProgress?.(clamped, dur);
    if (this.isPlayingState && this.cached?.audioBuffer) {
      this.stopActiveSource();
      this.startSourceAt(clamped);
    }
  }

  setTempo(factor: number): void {
    if (!Number.isFinite(factor) || factor <= 0) return;
    const clamped = Math.max(0.5, Math.min(2.0, factor));
    if (Math.abs(this.tempoFactor - clamped) < 1e-4) return;
    if (this.isPlayingState) {
      const ctx = this.host.getAudioContext();
      this.startScoreTime = this.calculateCurrentScoreTime();
      this.startAudioTime = ctx ? ctx.currentTime : 0;
      if (this.activeSource) this.activeSource.playbackRate.value = clamped;
    }
    this.tempoFactor = clamped;
  }

  exportWavBlob(format: WavFormat): Blob | null {
    if (!this.hasValidCache() || !this.cached) return null;
    return createPlanarWavBlob(this.cached.left, this.cached.right, this.cached.sampleRate, format);
  }

  panic(): void {
    this.stop();
  }

  destroy(): void {
    this.operationEpoch++;
    this.workerClient.dispose();
    this.isRenderingState = false;
    this.stopPlaybackInternal(true);
    this.cached = null;
    this.score = null;
  }

  private getVoicingParams(): OfflineVoicingParams {
    return this.callbacks.getVoicing?.() ?? DEFAULT_VOICING;
  }

  private getSpaceParams(): OfflineSpaceParams {
    return this.callbacks.getSpace?.() ?? DEFAULT_SPACE;
  }

  private getOfflinePerfConfig(): PerfProfileConfig {
    return this.callbacks.getPerfConfig?.() ?? PERF_PRESET_CONFIGS[PERF_PRESET_FULL];
  }

  private computeCacheKey(): string {
    return JSON.stringify({
      scoreKey: this.scoreKey,
      voicing: this.getVoicingParams(),
      space: this.getSpaceParams(),
      perfConfig: this.getOfflinePerfConfig(),
    });
  }

  private setStatus(status: OfflineRenderStatus, progress: number, error: string | null = null): void {
    this.status = status;
    this.renderProgress = progress;
    this.callbacks.onRenderStatus?.(status, progress, error);
  }

  private cancelActiveRender(): void {
    if (this.isRenderingState || this.workerClient.isRendering()) {
      this.workerClient.cancel();
      this.isRenderingState = false;
    }
  }

  private startSourceAt(offsetSec: number): void {
    const ctx = this.host.getAudioContext();
    const analyser = this.host.getAnalyser();
    const buf = this.cached?.audioBuffer;
    if (!ctx || !analyser || !buf) return;

    const source = ctx.createBufferSource();
    source.buffer = buf;
    source.playbackRate.value = this.tempoFactor;
    source.connect(analyser);
    this.startAudioTime = ctx.currentTime;
    this.startScoreTime = offsetSec;
    this.activeSource = source;
    source.onended = () => {
      if (this.activeSource !== source) return;
      this.activeSource = null;
      this.isPlayingState = false;
      this.clearVisualTimer();
      this.playheadSeconds = 0;
      this.nextEventIndex = 0;
      this.clearVisualKeysAndPedals();
      this.callbacks.onProgress?.(0, this.score?.durationSeconds ?? 0);
      this.callbacks.onStateChange?.(false);
      this.setStatus(this.hasValidCache() ? 'ready' : 'idle', 1);
    };
    source.start(0, Math.max(0, Math.min(buf.duration, offsetSec)));
  }

  private stopActiveSource(): void {
    const source = this.activeSource;
    if (!source) return;
    this.activeSource = null;
    source.onended = null;
    try {
      source.stop();
    } catch {
      // Source may already be stopped.
    }
    source.disconnect();
  }

  private clearVisualTimer(): void {
    if (this.visualTimerId !== null) {
      window.clearInterval(this.visualTimerId);
      this.visualTimerId = null;
    }
  }

  private clearVisualKeysAndPedals(): void {
    for (const note of this.activeVisualNotes) {
      this.callbacks.onNoteRelease?.(note);
    }
    this.activeVisualNotes.clear();
    this.callbacks.onPedalChange?.(PEDAL_SUSTAIN, 0);
    this.callbacks.onPedalChange?.(PEDAL_SOSTENUTO, 0);
    this.callbacks.onPedalChange?.(PEDAL_UNA_CORDA, 0);
  }

  private stopPlaybackInternal(resetPedals: boolean): void {
    this.stopActiveSource();
    this.isPlayingState = false;
    this.clearVisualTimer();
    for (const note of this.activeVisualNotes) {
      this.callbacks.onNoteRelease?.(note);
    }
    this.activeVisualNotes.clear();
    if (resetPedals) {
      this.callbacks.onPedalChange?.(PEDAL_SUSTAIN, 0);
      this.callbacks.onPedalChange?.(PEDAL_SOSTENUTO, 0);
      this.callbacks.onPedalChange?.(PEDAL_UNA_CORDA, 0);
    }
    this.callbacks.onStateChange?.(false);
  }

  private calculateCurrentScoreTime(): number {
    if (!this.isPlayingState) return this.playheadSeconds;
    const ctx = this.host.getAudioContext();
    const audioNow = ctx ? ctx.currentTime : 0;
    return this.startScoreTime + Math.max(0, audioNow - this.startAudioTime) * this.tempoFactor;
  }

  private syncVisualStateAt(targetSec: number): void {
    for (const note of this.activeVisualNotes) {
      this.callbacks.onNoteRelease?.(note);
    }
    this.activeVisualNotes.clear();
    let cc64 = 0;
    let cc66 = 0;
    let cc67 = 0;
    let idx = 0;
    if (this.score) {
      const events = this.score.events;
      while (idx < events.length && events[idx].timeSeconds < targetSec) {
        const ev = events[idx++];
        if (ev.type === 'noteOn') this.activeVisualNotes.add(ev.midiKey);
        else if (ev.type === 'noteOff') this.activeVisualNotes.delete(ev.midiKey);
        else if (ev.type === 'cc') {
          if (ev.controller === 64) cc64 = ev.value;
          else if (ev.controller === 66) cc66 = ev.value;
          else if (ev.controller === 67) cc67 = ev.value;
        }
      }
    }
    this.nextEventIndex = idx;
    for (const note of this.activeVisualNotes) this.callbacks.onNotePress?.(note, 100);
    this.callbacks.onPedalChange?.(PEDAL_SUSTAIN, cc64);
    this.callbacks.onPedalChange?.(PEDAL_SOSTENUTO, cc66);
    this.callbacks.onPedalChange?.(PEDAL_UNA_CORDA, cc67);
  }

  private tickVisuals(): void {
    if (!this.isPlayingState || !this.score) return;
    const current = this.calculateCurrentScoreTime();
    this.playheadSeconds = current;
    const events = this.score.events;
    while (this.nextEventIndex < events.length && events[this.nextEventIndex].timeSeconds <= current) {
      const ev = events[this.nextEventIndex++];
      if (ev.type === 'noteOn') {
        this.activeVisualNotes.add(ev.midiKey);
        this.callbacks.onNotePress?.(ev.midiKey, ev.velocity);
      } else if (ev.type === 'noteOff') {
        this.activeVisualNotes.delete(ev.midiKey);
        this.callbacks.onNoteRelease?.(ev.midiKey);
      } else if (ev.type === 'cc') {
        if (ev.controller === 64) this.callbacks.onPedalChange?.(PEDAL_SUSTAIN, ev.value);
        else if (ev.controller === 66) this.callbacks.onPedalChange?.(PEDAL_SOSTENUTO, ev.value);
        else if (ev.controller === 67) this.callbacks.onPedalChange?.(PEDAL_UNA_CORDA, ev.value);
      }
    }
    const dur = this.score.durationSeconds;
    this.callbacks.onProgress?.(Math.min(current, dur), dur);
  }
}
