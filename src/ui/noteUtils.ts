/** Note helpers shared by the UI layer (MIDI numbering: 60 = C4, 69 = A4 = 440 Hz). */

export const MIDI_MIN = 21; // A0
export const MIDI_MAX = 108; // C8

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] as const;
const BLACK_PITCH_CLASSES = new Set([1, 3, 6, 8, 10]);

export function pitchClass(midi: number): number {
  return ((midi % 12) + 12) % 12;
}

export function isBlackKey(midi: number): boolean {
  return BLACK_PITCH_CLASSES.has(pitchClass(midi));
}

export function midiToNoteName(midi: number): string {
  return `${NOTE_NAMES[pitchClass(midi)]}${Math.floor(midi / 12) - 1}`;
}

export function midiToFrequency(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

export function clampVelocity(v: number): number {
  if (!Number.isFinite(v)) return 80;
  return Math.max(1, Math.min(127, Math.round(v)));
}
