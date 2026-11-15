/** High-precision lookahead MIDI score playback scheduler synchronized to Web Audio clock. */

import {
  PEDAL_SOSTENUTO,
  PEDAL_SUSTAIN,
  PEDAL_UNA_CORDA,
  u7ToU16,
} from '../audio/abi';
import type { WasmAudioHost } from '../audio/wasm-host';
import type { ParsedSmf } from './smf-parser';

export interface MidiPlayerCallbacks {
  onNotePress?(midiKey: number, velocity: number): void;
  onNoteRelease?(midiKey: number): void;
  onPedalChange?(pedalKind: number, value7: number): void;
  onProgress?(currentSec: number, totalSec: number): void;
  onStateChange?(isPlaying: boolean): void;
}

export class MidiPlayer {
  private readonly host: WasmAudioHost;
  private readonly callbacks: MidiPlayerCallbacks;

  private score: ParsedSmf | null = null;
  private isPlayingState = false;
  private timerId: number | null = null;

  private startAudioTime = 0;
  private startScoreTime = 0;
  private playheadSeconds = 0;
  private tempoFactor = 1.0;
  private nextEventIndex = 0;
  private readonly activeNotes = new Set<number>();

  constructor(host: WasmAudioHost, callbacks: MidiPlayerCallbacks = {}) {
    this.host = host;
    this.callbacks = callbacks;
  }

  load(score: ParsedSmf): void {
    this.stop();
    this.score = score;
    this.callbacks.onProgress?.(0, score.durationSeconds);
  }

  getScore(): ParsedSmf | null {
    return this.score;
  }

  isPlaying(): boolean {
    return this.isPlayingState;
  }

  getCurrentTime(): number {
    return this.playheadSeconds;
  }

  getDuration(): number {
    return this.score?.durationSeconds ?? 0;
  }

  getTempo(): number {
    return this.tempoFactor;
  }

  async play(): Promise<void> {
    if (!this.score || this.isPlayingState) {
      return;
    }
    await this.host.unlock();
    const ctx = this.host.getAudioContext();
    const audioNow = ctx ? ctx.currentTime : 0;

    this.startAudioTime = audioNow;
    this.startScoreTime = this.playheadSeconds;
    this.isPlayingState = true;
    this.callbacks.onStateChange?.(true);

    this.timerId = window.setInterval(() => this.tick(), 15);
  }

  pause(): void {
    if (!this.isPlayingState) {
      return;
    }
    this.isPlayingState = false;
    if (this.timerId !== null) {
      window.clearInterval(this.timerId);
      this.timerId = null;
    }

    // Flush active notes with dampers engaging for natural acoustic decay
    for (const note of this.activeNotes) {
      this.host.noteOff(note, 0, 0);
      this.callbacks.onNoteRelease?.(note);
    }
    this.activeNotes.clear();

    this.callbacks.onStateChange?.(false);
  }

  stop(): void {
    this.pause();
    this.playheadSeconds = 0;
    this.nextEventIndex = 0;
    const dur = this.score?.durationSeconds ?? 0;
    this.callbacks.onProgress?.(0, dur);
  }

  seek(targetSeconds: number): void {
    const dur = this.score?.durationSeconds ?? 0;
    const clamped = Math.max(0, Math.min(dur, targetSeconds));

    // Release currently ringing voices to avoid hanging notes
    for (const note of this.activeNotes) {
      this.host.noteOff(note, 0, 0);
      this.callbacks.onNoteRelease?.(note);
    }
    this.activeNotes.clear();

    this.playheadSeconds = clamped;
    this.startScoreTime = clamped;

    const ctx = this.host.getAudioContext();
    this.startAudioTime = ctx ? ctx.currentTime : 0;

    // Fast-forward or rewind event pointer
    if (this.score) {
      const events = this.score.events;
      let idx = 0;
      while (idx < events.length && events[idx].timeSeconds < clamped) {
        idx++;
      }
      this.nextEventIndex = idx;
    }

    this.callbacks.onProgress?.(this.playheadSeconds, dur);
  }

  setTempo(factor: number): void {
    const clamped = Math.max(0.5, Math.min(2.0, factor));
    if (Math.abs(this.tempoFactor - clamped) < 1e-4) {
      return;
    }
    const currentScoreTime = this.calculateCurrentScoreTime();
    this.startScoreTime = currentScoreTime;
    const ctx = this.host.getAudioContext();
    this.startAudioTime = ctx ? ctx.currentTime : 0;
    this.tempoFactor = clamped;
  }

  panic(): void {
    this.stop();
    this.host.allNotesOff();
  }

  destroy(): void {
    this.stop();
    this.score = null;
  }

  private calculateCurrentScoreTime(): number {
    if (!this.isPlayingState) {
      return this.playheadSeconds;
    }
    const ctx = this.host.getAudioContext();
    const audioNow = ctx ? ctx.currentTime : 0;
    const elapsedAudio = Math.max(0, audioNow - this.startAudioTime);
    return this.startScoreTime + elapsedAudio * this.tempoFactor;
  }

  private tick(): void {
    if (!this.isPlayingState || !this.score) {
      return;
    }
    const currentScoreTime = this.calculateCurrentScoreTime();
    this.playheadSeconds = currentScoreTime;

    const events = this.score.events;
    while (
      this.nextEventIndex < events.length &&
      events[this.nextEventIndex].timeSeconds <= currentScoreTime
    ) {
      const ev = events[this.nextEventIndex++];
      if (ev.type === 'noteOn') {
        const velU16 = u7ToU16(ev.velocity);
        this.host.noteOn(ev.midiKey, velU16, 0);
        this.activeNotes.add(ev.midiKey);
        this.callbacks.onNotePress?.(ev.midiKey, ev.velocity);
      } else if (ev.type === 'noteOff') {
        const velU16 = u7ToU16(ev.velocity);
        this.host.noteOff(ev.midiKey, velU16, 0);
        this.activeNotes.delete(ev.midiKey);
        this.callbacks.onNoteRelease?.(ev.midiKey);
      } else if (ev.type === 'cc') {
        if (ev.controller === 64) {
          this.host.pedal(PEDAL_SUSTAIN, u7ToU16(ev.value), 0);
          this.callbacks.onPedalChange?.(PEDAL_SUSTAIN, ev.value);
        } else if (ev.controller === 66) {
          const val = ev.value >= 64 ? 65535 : 0;
          this.host.pedal(PEDAL_SOSTENUTO, val, 0);
          this.callbacks.onPedalChange?.(PEDAL_SOSTENUTO, ev.value);
        } else if (ev.controller === 67) {
          this.host.pedal(PEDAL_UNA_CORDA, u7ToU16(ev.value), 0);
          this.callbacks.onPedalChange?.(PEDAL_UNA_CORDA, ev.value);
        }
      }
    }

    this.callbacks.onProgress?.(this.playheadSeconds, this.score.durationSeconds);

    if (this.nextEventIndex >= events.length && currentScoreTime >= this.score.durationSeconds) {
      this.stop();
    }
  }
}
