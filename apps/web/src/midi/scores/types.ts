/** Shared types and event builders for procedural demo scores. */

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
  const ticksPerSecond = division * 2; // Nominal 120 BPM conversion
  const onTick = Math.round(timeSeconds * ticksPerSecond);
  const offTick = Math.round((timeSeconds + durationSeconds) * ticksPerSecond);

  events.push({
    type: 'noteOn',
    timeSeconds,
    tick: onTick,
    channel,
    midiKey,
    velocity,
  } as MidiNoteEvent);

  events.push({
    type: 'noteOff',
    timeSeconds: timeSeconds + durationSeconds,
    tick: offTick,
    channel,
    midiKey,
    velocity: offVelocity,
  } as MidiNoteEvent);
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
  const ticksPerSecond = division * 2;
  events.push({
    type: 'cc',
    timeSeconds,
    tick: Math.round(timeSeconds * ticksPerSecond),
    channel,
    controller,
    value,
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
