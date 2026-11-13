/**
 * PianoController: single source of truth for which notes are held / pedal-sustained.
 * Several input sources (pointers, computer keys) can hold the same note; the engine
 * note-off is sent only when the last source lets go. Visual state goes to the keyboard
 * component; sound goes to the audio controller.
 */

import type { AudioController } from './AudioController';
import type { PianoKeyboard } from './PianoKeyboard';

export interface NoteEvent {
  midi: number;
  velocity: number;
}

export class PianoController {
  private readonly held = new Map<number, Set<string>>();
  private readonly sustained = new Set<number>();
  private pedal = false;
  /** Notifies the UI (frequency readout etc.) about each strike. */
  onStrike?: (e: NoteEvent) => void;
  onPedalChange?: (down: boolean) => void;

  constructor(
    private readonly audio: AudioController,
    private readonly keyboard: PianoKeyboard,
  ) {}

  isPedalDown(): boolean {
    return this.pedal;
  }

  private refresh(midi: number): void {
    const isHeld = (this.held.get(midi)?.size ?? 0) > 0;
    this.keyboard.setActive(midi, isHeld, this.sustained.has(midi));
  }

  noteOn(midi: number, velocity: number, source: string): void {
    let set = this.held.get(midi);
    if (!set) this.held.set(midi, (set = new Set()));
    const wasHeld = set.size > 0;
    set.add(source);
    this.sustained.delete(midi);
    this.refresh(midi);
    if (wasHeld) return; // already sounding from another source
    this.audio.noteOn(midi, velocity);
    this.onStrike?.({ midi, velocity });
  }

  noteOff(midi: number, source: string): void {
    const set = this.held.get(midi);
    if (!set || !set.delete(source)) return;
    if (set.size > 0) return;
    this.held.delete(midi);
    if (this.pedal) this.sustained.add(midi);
    this.refresh(midi);
    this.audio.noteOff(midi);
  }

  setPedal(down: boolean): void {
    if (down === this.pedal) return;
    this.pedal = down;
    this.audio.setSustainPedal(down);
    if (!down) {
      const released = [...this.sustained];
      this.sustained.clear();
      released.forEach((m) => this.refresh(m));
    }
    this.onPedalChange?.(down);
  }

  /** Panic: silence everything and clear visual state. */
  allNotesOff(): void {
    const all = new Set([...this.held.keys(), ...this.sustained]);
    this.held.clear();
    this.sustained.clear();
    all.forEach((m) => this.refresh(m));
    this.audio.allNotesOff();
  }
}
