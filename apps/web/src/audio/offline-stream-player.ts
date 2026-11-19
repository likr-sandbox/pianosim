/** Progressive chunk assembly cache and sample-accurate AudioBufferSourceNode scheduler. */

import type { WavFormat } from './abi';
import type { OfflineChunkEvent } from './offline-worker-client';
import type { WasmAudioHost } from './wasm-host';
import { createPlanarWavBlob } from './wav-recorder';

const INITIAL_LEAD_SEC = 0.02;
const REBASE_LEAD_SEC = 0.01;
const MAX_BUFFER_SLICE_SEC = 4;

export interface OfflineStreamPlayerCallbacks {
  onBufferingChange?(buffering: boolean): void;
  onEnded?(): void;
}

export class OfflineStreamPlayer {
  private readonly host: WasmAudioHost;
  private callbacks: OfflineStreamPlayerCallbacks;

  private cacheKey = '';
  private sampleRate = 48_000;
  private totalFrames = 0;
  private renderedFrames = 0;
  private complete = false;
  private left: Float32Array | null = null;
  private right: Float32Array | null = null;

  private readonly activeSources = new Set<AudioBufferSourceNode>();
  private playingState = false;
  private bufferingState = false;
  private bufferingTargetFrame = 0;

  private nextScheduleFrame = 0;
  private nextAudioTime = 0;
  private anchorAudioTime = 0;
  private anchorScoreSec = 0;
  private pausedScoreSec = 0;
  private tempoFactor = 1.0;

  constructor(host: WasmAudioHost, callbacks: OfflineStreamPlayerCallbacks = {}) {
    this.host = host;
    this.callbacks = callbacks;
  }

  setCallbacks(callbacks: OfflineStreamPlayerCallbacks): void {
    this.callbacks = callbacks;
  }

  hasAudio(): boolean {
    return this.left !== null && this.right !== null && this.renderedFrames > 0;
  }

  isComplete(): boolean {
    return this.complete && this.left !== null && this.right !== null;
  }

  getCacheKey(): string {
    return this.cacheKey;
  }

  getTotalDurationSec(): number {
    return this.sampleRate > 0 ? this.totalFrames / this.sampleRate : 0;
  }

  getRenderedFrames(): number {
    return this.renderedFrames;
  }

  getTempo(): number {
    return this.tempoFactor;
  }

  isPlaying(): boolean {
    return this.playingState;
  }

  isBuffering(): boolean {
    return this.playingState && this.bufferingState;
  }

  getCurrentScoreTime(): number {
    if (!this.playingState) return this.pausedScoreSec;
    if (this.bufferingState) return this.anchorScoreSec;
    const ctx = this.host.getAudioContext();
    const audioNow = ctx ? ctx.currentTime : 0;
    const elapsed = Math.max(0, audioNow - this.anchorAudioTime) * this.tempoFactor;
    const maxSec = this.getTotalDurationSec();
    return Math.min(maxSec, this.anchorScoreSec + elapsed);
  }

  appendChunk(cacheKey: string, chunk: OfflineChunkEvent): void {
    if (chunk.chunkIndex === 0) {
      this.stopSources();
      this.cacheKey = cacheKey;
      this.sampleRate = chunk.sampleRate;
      this.totalFrames = chunk.totalFrames;
      this.renderedFrames = 0;
      this.complete = false;
      this.left = new Float32Array(chunk.totalFrames);
      this.right = new Float32Array(chunk.totalFrames);
    }

    if (!this.left || !this.right || this.cacheKey !== cacheKey) return;
    const copyLen = Math.min(chunk.frames, this.totalFrames - chunk.startFrame);
    if (copyLen > 0) {
      this.left.set(chunk.left.subarray(0, copyLen), chunk.startFrame);
      this.right.set(chunk.right.subarray(0, copyLen), chunk.startFrame);
      this.renderedFrames = Math.max(this.renderedFrames, chunk.startFrame + copyLen);
    }
    if (chunk.isFinal) {
      this.complete = true;
    }

    if (!this.playingState) return;

    if (this.bufferingState) {
      if (this.renderedFrames > this.bufferingTargetFrame || this.complete) {
        const resumeFrame = Math.min(this.bufferingTargetFrame, Math.max(0, this.totalFrames - 1));
        this.bufferingState = false;
        this.startPlaybackFromFrame(resumeFrame, INITIAL_LEAD_SEC);
        this.callbacks.onBufferingChange?.(false);
      }
      return;
    }

    if (this.nextScheduleFrame < this.renderedFrames) {
      const ctx = this.host.getAudioContext();
      if (ctx && this.nextAudioTime < ctx.currentTime + 0.005) {
        const rebaseAt = ctx.currentTime + REBASE_LEAD_SEC;
        this.anchorAudioTime = rebaseAt;
        this.anchorScoreSec = this.nextScheduleFrame / this.sampleRate;
        this.nextAudioTime = rebaseAt;
      }
      this.scheduleAvailableRange();
    }
  }

  playFrom(scoreSec: number): 'playing' | 'buffering' {
    if (!this.hasAudio()) {
      this.playingState = true;
      this.bufferingState = true;
      this.bufferingTargetFrame = Math.max(0, Math.round(scoreSec * this.sampleRate));
      this.anchorScoreSec = Math.max(0, scoreSec);
      this.pausedScoreSec = this.anchorScoreSec;
      return 'buffering';
    }

    const targetFrame = this.clampTargetFrame(scoreSec);
    this.stopSources();
    this.playingState = true;
    if (targetFrame < this.renderedFrames || (this.complete && targetFrame <= this.totalFrames)) {
      this.bufferingState = false;
      this.startPlaybackFromFrame(targetFrame, INITIAL_LEAD_SEC);
      return 'playing';
    }

    this.bufferingState = true;
    this.bufferingTargetFrame = targetFrame;
    this.anchorScoreSec = targetFrame / this.sampleRate;
    this.pausedScoreSec = this.anchorScoreSec;
    return 'buffering';
  }

  pause(): number {
    if (!this.playingState) return this.pausedScoreSec;
    const current = this.getCurrentScoreTime();
    this.stopSources();
    this.playingState = false;
    this.bufferingState = false;
    this.pausedScoreSec = current;
    return current;
  }

  stopPlayback(): void {
    this.stopSources();
    this.playingState = false;
    this.bufferingState = false;
    this.bufferingTargetFrame = 0;
    this.nextScheduleFrame = 0;
    this.nextAudioTime = 0;
    this.anchorAudioTime = 0;
    this.anchorScoreSec = 0;
    this.pausedScoreSec = 0;
  }

  clear(): void {
    this.stopPlayback();
    this.cacheKey = '';
    this.sampleRate = 48_000;
    this.totalFrames = 0;
    this.renderedFrames = 0;
    this.complete = false;
    this.left = null;
    this.right = null;
  }

  seek(scoreSec: number): 'playing' | 'buffering' | 'idle' {
    const clampedSec = Math.max(0, scoreSec);
    if (!this.hasAudio()) {
      this.pausedScoreSec = clampedSec;
      if (this.playingState) {
        this.bufferingTargetFrame = Math.max(0, Math.round(clampedSec * this.sampleRate));
        this.anchorScoreSec = clampedSec;
        return 'buffering';
      }
      return 'idle';
    }

    const targetFrame = this.clampTargetFrame(clampedSec);
    this.pausedScoreSec = targetFrame / this.sampleRate;
    if (!this.playingState) return 'idle';

    this.stopSources();
    if (targetFrame < this.renderedFrames || (this.complete && targetFrame <= this.totalFrames)) {
      const wasBuffering = this.bufferingState;
      this.bufferingState = false;
      this.startPlaybackFromFrame(targetFrame, INITIAL_LEAD_SEC);
      if (wasBuffering) this.callbacks.onBufferingChange?.(false);
      return 'playing';
    }

    const wasBuffering = this.bufferingState;
    this.bufferingState = true;
    this.bufferingTargetFrame = targetFrame;
    this.anchorScoreSec = targetFrame / this.sampleRate;
    if (!wasBuffering) this.callbacks.onBufferingChange?.(true);
    return 'buffering';
  }

  setTempo(factor: number): void {
    if (!Number.isFinite(factor) || factor <= 0) return;
    const clamped = Math.max(0.5, Math.min(2.0, factor));
    if (Math.abs(this.tempoFactor - clamped) < 1e-4) return;

    if (this.playingState && !this.bufferingState && this.hasAudio()) {
      const currentSec = this.getCurrentScoreTime();
      const currentFrame = this.clampTargetFrame(currentSec);
      this.stopSources();
      this.tempoFactor = clamped;
      if (currentFrame < this.renderedFrames) {
        this.startPlaybackFromFrame(currentFrame, REBASE_LEAD_SEC);
      } else if (!this.complete) {
        this.bufferingState = true;
        this.bufferingTargetFrame = currentFrame;
        this.anchorScoreSec = currentFrame / this.sampleRate;
        this.callbacks.onBufferingChange?.(true);
      }
      return;
    }

    this.tempoFactor = clamped;
  }

  exportWavBlob(format: WavFormat): Blob | null {
    if (!this.isComplete() || !this.left || !this.right) return null;
    return createPlanarWavBlob(this.left, this.right, this.sampleRate, format);
  }

  private clampTargetFrame(scoreSec: number): number {
    if (this.totalFrames <= 0) return 0;
    return Math.min(
      Math.max(0, Math.round(Math.max(0, scoreSec) * this.sampleRate)),
      this.totalFrames - 1,
    );
  }

  private startPlaybackFromFrame(startFrame: number, leadSec: number): void {
    const ctx = this.host.getAudioContext();
    if (!ctx) throw new Error('AudioContext is unavailable.');
    const startTime = ctx.currentTime + leadSec;
    this.anchorAudioTime = startTime;
    this.anchorScoreSec = startFrame / this.sampleRate;
    this.nextAudioTime = startTime;
    this.nextScheduleFrame = startFrame;
    this.scheduleAvailableRange();
  }

  private scheduleAvailableRange(): void {
    const ctx = this.host.getAudioContext();
    const analyser = this.host.getAnalyser();
    if (!ctx || !analyser || !this.left || !this.right) return;

    const maxSliceFrames = Math.max(this.sampleRate, this.sampleRate * MAX_BUFFER_SLICE_SEC);
    while (this.nextScheduleFrame < this.renderedFrames) {
      const endFrame = Math.min(this.renderedFrames, this.nextScheduleFrame + maxSliceFrames);
      const len = endFrame - this.nextScheduleFrame;
      if (len <= 0) break;

      const buf = ctx.createBuffer(2, len, this.sampleRate);
      buf.copyToChannel(this.left.slice(this.nextScheduleFrame, endFrame), 0);
      buf.copyToChannel(this.right.slice(this.nextScheduleFrame, endFrame), 1);

      const source = ctx.createBufferSource();
      source.buffer = buf;
      source.playbackRate.value = this.tempoFactor;
      source.connect(analyser);
      this.activeSources.add(source);

      source.onended = () => {
        if (!this.activeSources.delete(source)) return;
        try {
          source.disconnect();
        } catch {
          // Ignore disconnect errors.
        }
        if (!this.playingState || this.bufferingState) return;
        if (this.activeSources.size === 0) {
          if (this.complete && this.nextScheduleFrame >= this.totalFrames) {
            this.playingState = false;
            this.pausedScoreSec = 0;
            this.callbacks.onEnded?.();
          } else if (!this.complete && this.nextScheduleFrame >= this.renderedFrames) {
            this.bufferingState = true;
            this.bufferingTargetFrame = Math.min(
              Math.max(0, this.totalFrames - 1),
              this.nextScheduleFrame + Math.round(0.5 * this.sampleRate),
            );
            this.anchorScoreSec = this.nextScheduleFrame / this.sampleRate;
            this.callbacks.onBufferingChange?.(true);
          }
        }
      };

      source.start(this.nextAudioTime);
      this.nextAudioTime += len / (this.sampleRate * this.tempoFactor);
      this.nextScheduleFrame = endFrame;
    }
  }

  private stopSources(): void {
    for (const source of this.activeSources) {
      source.onended = null;
      try {
        source.stop();
      } catch {
        // Source may already be stopped.
      }
      try {
        source.disconnect();
      } catch {
        // Ignore disconnect errors.
      }
    }
    this.activeSources.clear();
  }
}
