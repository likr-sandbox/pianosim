/** Shared types and expressive event builders for procedural demo scores. */

import type { MidiCcEvent, MidiNoteEvent, MidiPlaybackEvent, ParsedSmf } from '../smf-parser';

export interface DemoScoreDescriptor {
  readonly id: string;
  readonly title: string;
  readonly composer: string;
  readonly era: string;
  readonly description: string;
  readonly durationSeconds: number;
  readonly generator: () => ParsedSmf;
}

export function clampMidiKey(key: number): number {
  return Math.max(21, Math.min(108, Math.round(key)));
}

export function clampVelocity(vel: number): number {
  return Math.max(1, Math.min(127, Math.round(vel)));
}

export function clampCcValue(val: number): number {
  return Math.max(0, Math.min(127, Math.round(val)));
}

/** Append a NoteOn and matching NoteOff event to the event list. */
export function addNote(
  events: MidiPlaybackEvent[],
  timeSeconds: number,
  durationSeconds: number,
  midiKey: number,
  velocity: number,
  offVelocity = 40,
  channel = 0,
  division = 480,
): void {
  const safeTime = Math.max(0, timeSeconds);
  const safeDur = Math.max(0.03, durationSeconds);
  const key = clampMidiKey(midiKey);
  const vel = clampVelocity(velocity);
  const offVel = clampVelocity(offVelocity);
  const ticksPerSecond = division * 2; // Nominal 120 BPM conversion
  const onTick = Math.round(safeTime * ticksPerSecond);
  const offTick = Math.round((safeTime + safeDur) * ticksPerSecond);

  events.push({
    type: 'noteOn',
    timeSeconds: safeTime,
    tick: onTick,
    channel,
    midiKey: key,
    velocity: vel,
  } as MidiNoteEvent);

  events.push({
    type: 'noteOff',
    timeSeconds: safeTime + safeDur,
    tick: offTick,
    channel,
    midiKey: key,
    velocity: offVel,
  } as MidiNoteEvent);
}

/** Append a micro-rolled chord with staggered onsets and top-voice emphasis. */
export function addRolledChord(
  events: MidiPlaybackEvent[],
  baseTimeSeconds: number,
  durationSeconds: number,
  keys: ReadonlyArray<number>,
  baseVelocity: number,
  rollStepSeconds = 0.014,
  topVoiceBoost = 8,
  offVelocity = 35,
): void {
  const count = keys.length;
  for (let i = 0; i < count; i++) {
    const offset = i * rollStepSeconds;
    const isTop = i === count - 1;
    const vel = isTop ? baseVelocity + topVoiceBoost : baseVelocity - (count - 1 - i) * 2;
    addNote(
      events,
      baseTimeSeconds + offset,
      Math.max(0.05, durationSeconds - offset * 0.5),
      keys[i],
      vel,
      offVelocity,
    );
  }
}

/** Append a Control Change (CC) event to the event list. */
export function addCc(
  events: MidiPlaybackEvent[],
  timeSeconds: number,
  controller: number,
  value: number,
  channel = 0,
  division = 480,
): void {
  const safeTime = Math.max(0, timeSeconds);
  const ticksPerSecond = division * 2;
  events.push({
    type: 'cc',
    timeSeconds: safeTime,
    tick: Math.round(safeTime * ticksPerSecond),
    channel,
    controller,
    value: clampCcValue(value),
  } as MidiCcEvent);
}

/** Stable chronological event sorting with deterministic tie-breaking (noteOff -> cc -> noteOn). */
export function sortEvents(events: MidiPlaybackEvent[]): void {
  events.sort((a, b) => {
    if (Math.abs(a.timeSeconds - b.timeSeconds) > 1e-6) {
      return a.timeSeconds - b.timeSeconds;
    }
    const prioA = a.type === 'noteOff' ? 0 : a.type === 'cc' ? 1 : 2;
    const prioB = b.type === 'noteOff' ? 0 : b.type === 'cc' ? 1 : 2;
    return prioA - prioB;
  });
}
