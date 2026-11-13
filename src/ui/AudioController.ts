/**
 * AudioController: owns the PianoAudioEngine lifecycle (autoplay-policy unlock).
 *
 * The AudioContext is created lazily on the first user gesture (start button, key press,
 * computer key). Note events issued before the engine is ready are queued and flushed in
 * order once it is running, so the first strike is never lost or reordered.
 */

import { PianoAudioEngine } from '../audio/PianoAudioEngine';

export type AudioState = 'idle' | 'starting' | 'running' | 'error';

type PendingOp =
  | { type: 'on'; midi: number; velocity: number }
  | { type: 'off'; midi: number }
  | { type: 'pedal'; down: boolean }
  | { type: 'panic' };

const MAX_PENDING = 256;

export class AudioController {
  readonly engine: PianoAudioEngine;
  private state: AudioState = 'idle';
  private startPromise: Promise<void> | null = null;
  private pending: PendingOp[] = [];
  private listeners: Array<(s: AudioState, detail?: string) => void> = [];

  constructor(engine: PianoAudioEngine = new PianoAudioEngine()) {
    this.engine = engine;
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden && this.state === 'running') void this.engine.resume().catch(() => {});
    });
  }

  getState(): AudioState {
    return this.state;
  }

  onStateChange(fn: (s: AudioState, detail?: string) => void): void {
    this.listeners.push(fn);
  }

  private setState(s: AudioState, detail?: string): void {
    this.state = s;
    this.listeners.forEach((l) => l(s, detail));
  }

  /** Idempotent: safe to call from every gesture. */
  start(): Promise<void> {
    if (this.state === 'running') return Promise.resolve();
    if (this.startPromise) return this.startPromise;
    this.setState('starting');
    this.startPromise = (async () => {
      try {
        await this.engine.init();
        if (!this.engine.isInitialized()) throw new Error('Web Audio is not available in this browser');
        await this.engine.resume();
        this.setState('running');
        const ops = this.pending;
        this.pending = [];
        for (const op of ops) this.dispatch(op);
      } catch (err) {
        this.pending = [];
        this.startPromise = null; // allow retry
        this.setState('error', err instanceof Error ? err.message : String(err));
      }
    })();
    return this.startPromise;
  }

  private dispatch(op: PendingOp): void {
    switch (op.type) {
      case 'on':
        this.engine.noteOn(op.midi, op.velocity);
        break;
      case 'off':
        this.engine.noteOff(op.midi);
        break;
      case 'pedal':
        this.engine.setSustainPedal(op.down);
        break;
      case 'panic':
        this.engine.allNotesOff();
        break;
    }
  }

  private send(op: PendingOp): void {
    if (this.state === 'running') {
      this.dispatch(op);
      return;
    }
    if (this.state === 'error') return;
    if (this.pending.length < MAX_PENDING) this.pending.push(op);
    void this.start();
  }

  noteOn(midi: number, velocity: number): void {
    this.send({ type: 'on', midi, velocity });
  }
  noteOff(midi: number): void {
    this.send({ type: 'off', midi });
  }
  setSustainPedal(down: boolean): void {
    this.send({ type: 'pedal', down });
  }
  allNotesOff(): void {
    this.send({ type: 'panic' });
  }
}
