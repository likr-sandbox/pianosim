/** Offline MIDI Render Worker controller and progressive AudioBuffer streaming coordinator. */

import {
  PEDAL_SOSTENUTO,
  PEDAL_SUSTAIN,
  PEDAL_UNA_CORDA,
  PERF_PRESET_CONFIGS,
  PERF_PRESET_FULL,
  type PerfProfileConfig,
  type WavFormat,
} from './abi';
import {
  DEFAULT_OFFLINE_SPACE,
  DEFAULT_OFFLINE_VOICING,
  type OfflineRenderStatus,
  type OfflineSpaceParams,
  type OfflineVoicingParams,
} from './offline-abi';
import { OfflineStreamPlayer } from './offline-stream-player';
import { OfflineWorkerClient } from './offline-worker-client';
import type { WasmAudioHost } from './wasm-host';
import type { ParsedSmf } from '../midi/smf-parser';

const FIRST_CHUNK_SEC = 1.5;
const STREAM_CHUNK_SEC = 0.75;

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

function defaultScoreKey(score: ParsedSmf): string {
  const first = score.events[0];
  const last = score.events[score.events.length - 1];
  return `smf:${score.events.length}:${score.durationSeconds.toFixed(4)}:${first ? first.tick : 0}:${last ? last.tick : 0}`;
}

export class OfflineRenderer {
  private readonly host: WasmAudioHost;
  private readonly workerClient = new OfflineWorkerClient();
  private readonly streamPlayer: OfflineStreamPlayer;
  private callbacks: OfflineRendererCallbacks;

  private score: ParsedSmf | null = null;
  private scoreKey = '';
  private status: OfflineRenderStatus = 'idle';
  private renderProgress = 0;

  private isPlayingState = false;
  private isRenderingState = false;
  private renderEpoch = 0;
  private playEpoch = 0;

  private visualTimerId: number | null = null;
  private playheadSeconds = 0;
  private nextEventIndex = 0;
  private readonly activeVisualNotes = new Set<number>();

  constructor(host: WasmAudioHost, callbacks: OfflineRendererCallbacks = {}) {
    this.host = host;
    this.callbacks = callbacks;
    this.streamPlayer = new OfflineStreamPlayer(host, {
      onBufferingChange: (buffering) => {
        if (!this.isPlayingState) return;
        if (buffering) {
          this.clearVisualTimer();
          this.setStatus('buffering', this.renderProgress);
        } else {
          this.syncVisualStateAt(this.streamPlayer.getCurrentScoreTime());
          this.startVisualTimer();
          this.setStatus('playing', this.renderProgress);
        }
      },
      onEnded: () => {
        this.isPlayingState = false;
        this.clearVisualTimer();
        this.playheadSeconds = 0;
        this.nextEventIndex = 0;
        this.clearVisualKeysAndPedals();
        this.callbacks.onProgress?.(0, this.score?.durationSeconds ?? 0);
        this.callbacks.onStateChange?.(false);
        this.setStatus(this.hasValidCache() ? 'ready' : 'idle', this.hasValidCache() ? 1 : this.renderProgress);
      },
    });
  }

  setCallbacks(callbacks: OfflineRendererCallbacks): void {
    this.callbacks = callbacks;
  }

  load(score: ParsedSmf, scoreKey?: string): void {
    this.renderEpoch++;
    this.playEpoch++;
    this.cancelActiveRender();
    this.stopPlaybackInternal(true);
    this.score = score;
    this.scoreKey = scoreKey ?? defaultScoreKey(score);
    this.playheadSeconds = 0;
    this.nextEventIndex = 0;
    const targetKey = this.computeCacheKey();
    if (!this.streamPlayer.isComplete() || this.streamPlayer.getCacheKey() !== targetKey) {
      this.streamPlayer.clear();
    }
    const ready = this.hasValidCache();
    this.setStatus(ready ? 'ready' : 'idle', ready ? 1 : 0);
    this.callbacks.onProgress?.(0, score.durationSeconds);
  }

  invalidateIfStale(): void {
    const targetKey = this.computeCacheKey();
    const keyMatches = this.streamPlayer.getCacheKey() === targetKey;
    if (keyMatches && (this.streamPlayer.hasAudio() || this.isRenderingState)) return;
    if (!this.isRenderingState && !this.streamPlayer.hasAudio()) return;

    this.renderEpoch++;
    if (this.isRenderingState) {
      this.cancelActiveRender();
    }
    if (!this.isPlayingState || !this.streamPlayer.isComplete()) {
      this.playEpoch++;
      this.stopPlaybackInternal(true);
      this.streamPlayer.clear();
      this.playheadSeconds = 0;
      this.nextEventIndex = 0;
      this.callbacks.onProgress?.(0, this.score?.durationSeconds ?? 0);
      this.setStatus('idle', 0);
    } else {
      this.callbacks.onRenderStatus?.(this.status, this.renderProgress, null);
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
    return this.streamPlayer.isComplete() && this.streamPlayer.getCacheKey() === this.computeCacheKey();
  }

  getCurrentTime(): number {
    return this.isPlayingState ? this.streamPlayer.getCurrentScoreTime() : this.playheadSeconds;
  }

  getDuration(): number {
    return this.score?.durationSeconds ?? 0;
  }

  getTempo(): number {
    return this.streamPlayer.getTempo();
  }

  async play(): Promise<void> {
    if (!this.score || this.isPlayingState || this.status === 'rendering') return;
    const playEp = ++this.playEpoch;
    await this.host.unlock();
    if (playEp !== this.playEpoch || !this.score || this.isPlayingState) return;

    const targetKey = this.computeCacheKey();
    if (this.streamPlayer.hasAudio() && this.streamPlayer.getCacheKey() === targetKey) {
      this.startStreamPlayback();
      return;
    }

    const renderEp = ++this.renderEpoch;
    this.streamPlayer.clear();
    this.isRenderingState = true;
    this.setStatus('rendering', 0);

    try {
      await this.workerClient.render(
        (jobId) => ({
          jobId,
          scoreKey: this.scoreKey,
          events: this.score!.events,
          durationSeconds: this.score!.durationSeconds,
          voicing: this.getVoicingParams(),
          space: this.getSpaceParams(),
          perfConfig: this.getOfflinePerfConfig(),
          firstChunkSec: FIRST_CHUNK_SEC,
          chunkSec: STREAM_CHUNK_SEC,
        }),
        {
          onProgress: (p) => {
            if (renderEp !== this.renderEpoch) return;
            this.renderProgress = p;
            this.callbacks.onRenderStatus?.(this.status, p, null);
          },
          onChunk: (chunk) => {
            if (renderEp !== this.renderEpoch || !this.score) return;
            this.streamPlayer.appendChunk(targetKey, chunk);
            if (chunk.chunkIndex === 0 && playEp === this.playEpoch && !this.isPlayingState) {
              this.startStreamPlayback();
            }
            if (chunk.isFinal) {
              this.isRenderingState = false;
              this.renderProgress = 1;
              this.callbacks.onRenderStatus?.(this.status, 1, null);
            }
          },
          onError: (msg) => {
            if (renderEp !== this.renderEpoch) return;
            this.isRenderingState = false;
            this.stopPlaybackInternal(true);
            this.setStatus('error', 0, msg);
          },
        },
      );
    } catch (err) {
      if (renderEp === this.renderEpoch) {
        this.isRenderingState = false;
      }
      throw err;
    }
  }

  pause(): void {
    this.playEpoch++;
    if (this.status === 'rendering' && !this.streamPlayer.hasAudio()) {
      this.renderEpoch++;
      this.cancelActiveRender();
      this.setStatus('idle', 0);
      return;
    }
    if (!this.isPlayingState) return;
    this.playheadSeconds = this.streamPlayer.pause();
    this.isPlayingState = false;
    this.clearVisualTimer();
    this.clearVisualKeysOnly();
    this.callbacks.onStateChange?.(false);
    this.setStatus('paused', this.renderProgress);
  }

  stop(): void {
    this.playEpoch++;
    const hasStream =
      this.streamPlayer.hasAudio() && this.streamPlayer.getCacheKey() === this.computeCacheKey();
    if (!hasStream && this.isRenderingState) {
      this.renderEpoch++;
      this.cancelActiveRender();
    }
    this.stopPlaybackInternal(true);
    this.playheadSeconds = 0;
    this.nextEventIndex = 0;
    this.callbacks.onProgress?.(0, this.score?.durationSeconds ?? 0);
    if (this.hasValidCache() || hasStream) {
      this.setStatus('ready', this.hasValidCache() ? 1 : this.renderProgress);
    } else {
      this.setStatus('idle', 0);
    }
  }

  cancelRender(): void {
    this.renderEpoch++;
    this.playEpoch++;
    const wasRendering = this.isRenderingState || this.workerClient.isRendering();
    if (wasRendering) {
      this.cancelActiveRender();
    }
    if (!this.hasValidCache()) {
      this.stopPlaybackInternal(true);
      this.streamPlayer.clear();
      this.playheadSeconds = 0;
      this.nextEventIndex = 0;
      this.callbacks.onProgress?.(0, this.score?.durationSeconds ?? 0);
      this.setStatus('idle', 0);
    } else if (wasRendering) {
      this.setStatus(this.isPlayingState ? 'playing' : 'ready', 1);
    }
  }

  seek(targetSeconds: number): void {
    if (!Number.isFinite(targetSeconds)) return;
    const dur = this.score?.durationSeconds ?? 0;
    const clamped = Math.max(0, Math.min(dur, targetSeconds));
    this.playheadSeconds = clamped;
    this.syncVisualStateAt(clamped);
    this.callbacks.onProgress?.(clamped, dur);

    const seekState = this.streamPlayer.seek(clamped);
    if (this.isPlayingState) {
      if (seekState === 'buffering') {
        this.clearVisualTimer();
        this.setStatus('buffering', this.renderProgress);
      } else if (seekState === 'playing') {
        this.startVisualTimer();
        this.setStatus('playing', this.renderProgress);
      }
    }
  }

  setTempo(factor: number): void {
    this.streamPlayer.setTempo(factor);
  }

  exportWavBlob(format: WavFormat): Blob | null {
    if (!this.hasValidCache()) return null;
    return this.streamPlayer.exportWavBlob(format);
  }

  panic(): void {
    this.stop();
  }

  destroy(): void {
    this.renderEpoch++;
    this.playEpoch++;
    this.workerClient.dispose();
    this.isRenderingState = false;
    this.stopPlaybackInternal(true);
    this.streamPlayer.clear();
    this.score = null;
  }

  private startStreamPlayback(): void {
    if (!this.score) return;
    const totalDur = this.streamPlayer.getTotalDurationSec();
    if (
      this.playheadSeconds >= this.score.durationSeconds ||
      (totalDur > 0 && this.playheadSeconds >= totalDur)
    ) {
      this.playheadSeconds = 0;
      this.nextEventIndex = 0;
    }

    this.syncVisualStateAt(this.playheadSeconds);
    this.isPlayingState = true;
    this.callbacks.onStateChange?.(true);
    const playState = this.streamPlayer.playFrom(this.playheadSeconds);
    if (playState === 'buffering') {
      this.clearVisualTimer();
      this.setStatus('buffering', this.renderProgress);
    } else {
      this.startVisualTimer();
      this.setStatus('playing', this.renderProgress);
    }
  }

  private getVoicingParams(): OfflineVoicingParams {
    return this.callbacks.getVoicing?.() ?? DEFAULT_OFFLINE_VOICING;
  }

  private getSpaceParams(): OfflineSpaceParams {
    return this.callbacks.getSpace?.() ?? DEFAULT_OFFLINE_SPACE;
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

  private startVisualTimer(): void {
    if (this.visualTimerId === null) {
      this.visualTimerId = window.setInterval(() => this.tickVisuals(), 16);
    }
  }

  private clearVisualTimer(): void {
    if (this.visualTimerId !== null) {
      window.clearInterval(this.visualTimerId);
      this.visualTimerId = null;
    }
  }

  private clearVisualKeysOnly(): void {
    for (const note of this.activeVisualNotes) {
      this.callbacks.onNoteRelease?.(note);
    }
    this.activeVisualNotes.clear();
  }

  private clearVisualKeysAndPedals(): void {
    this.clearVisualKeysOnly();
    this.callbacks.onPedalChange?.(PEDAL_SUSTAIN, 0);
    this.callbacks.onPedalChange?.(PEDAL_SOSTENUTO, 0);
    this.callbacks.onPedalChange?.(PEDAL_UNA_CORDA, 0);
  }

  private stopPlaybackInternal(resetPedals: boolean): void {
    this.streamPlayer.stopPlayback();
    this.isPlayingState = false;
    this.clearVisualTimer();
    if (resetPedals) {
      this.clearVisualKeysAndPedals();
    } else {
      this.clearVisualKeysOnly();
    }
    this.callbacks.onStateChange?.(false);
  }

  private syncVisualStateAt(targetSec: number): void {
    this.clearVisualKeysOnly();
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
    if (!this.isPlayingState || !this.score || this.streamPlayer.isBuffering()) return;
    const current = this.streamPlayer.getCurrentScoreTime();
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
