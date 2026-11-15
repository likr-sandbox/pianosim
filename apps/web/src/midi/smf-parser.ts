/** Binary Standard MIDI File (SMF) parser supporting Format 0 & 1 with VLQ, running status, and tempo maps. */

export interface MidiNoteEvent {
  type: 'noteOn' | 'noteOff';
  timeSeconds: number;
  tick: number;
  channel: number;
  midiKey: number;
  velocity: number;
}

export interface MidiCcEvent {
  type: 'cc';
  timeSeconds: number;
  tick: number;
  channel: number;
  controller: number;
  value: number;
}

export type MidiPlaybackEvent = MidiNoteEvent | MidiCcEvent;

export interface ParsedSmf {
  readonly format: number;
  readonly trackCount: number;
  readonly division: number;
  readonly durationSeconds: number;
  readonly events: ReadonlyArray<MidiPlaybackEvent>;
}

interface RawEvent {
  tick: number;
  order: number;
  ev: MidiPlaybackEvent;
}

interface TempoPoint {
  tick: number;
  timeSeconds: number;
  uspqn: number;
}

class ByteReader {
  private offset = 0;
  constructor(private readonly data: Uint8Array) {}

  get remaining(): number {
    return this.data.length - this.offset;
  }

  get pos(): number {
    return this.offset;
  }

  seek(pos: number): void {
    if (pos < 0 || pos > this.data.length) {
      throw new Error('Invalid SMF: seek out of bounds');
    }
    this.offset = pos;
  }

  readU8(): number {
    if (this.offset >= this.data.length) {
      throw new Error('Invalid SMF: unexpected end of stream');
    }
    return this.data[this.offset++];
  }

  readU16(): number {
    const b0 = this.readU8();
    const b1 = this.readU8();
    return (b0 << 8) | b1;
  }

  readU32(): number {
    const b0 = this.readU8();
    const b1 = this.readU8();
    const b2 = this.readU8();
    const b3 = this.readU8();
    return ((b0 << 24) | (b1 << 16) | (b2 << 8) | b3) >>> 0;
  }

  readVlq(): number {
    let val = 0;
    for (let i = 0; i < 4; i++) {
      if (this.offset >= this.data.length) {
        throw new Error('Invalid SMF: corrupted VLQ delta time');
      }
      const b = this.data[this.offset++];
      val = (val << 7) | (b & 0x7f);
      if ((b & 0x80) === 0) {
        return val >>> 0;
      }
    }
    throw new Error('Invalid SMF: corrupted VLQ delta time (exceeds 4 bytes)');
  }

  skip(n: number): void {
    if (n < 0 || this.offset + n > this.data.length) {
      throw new Error('Invalid SMF: truncated chunk');
    }
    this.offset += n;
  }
}

/** Parses a binary Standard MIDI File (Format 0 or 1). */
export function parseSmf(buffer: ArrayBuffer | Uint8Array): ParsedSmf {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  if (bytes.length < 14) {
    throw new Error('Invalid SMF: file too small for MThd header');
  }

  const reader = new ByteReader(bytes);

  // MThd header: 0x4D 0x54 0x68 0x64
  const mthdMagic = [reader.readU8(), reader.readU8(), reader.readU8(), reader.readU8()];
  if (
    mthdMagic[0] !== 0x4d ||
    mthdMagic[1] !== 0x54 ||
    mthdMagic[2] !== 0x68 ||
    mthdMagic[3] !== 0x64
  ) {
    throw new Error('Invalid SMF: missing MThd header chunk');
  }

  const headerLen = reader.readU32();
  if (headerLen < 6) {
    throw new Error(`Invalid SMF: MThd length must be >= 6, got ${headerLen}`);
  }

  const format = reader.readU16();
  if (format !== 0 && format !== 1) {
    throw new Error(`Unsupported SMF format: ${format} (only Format 0 and 1 are supported)`);
  }

  const trackCount = reader.readU16();
  const divisionRaw = reader.readU16();
  if ((divisionRaw & 0x8000) !== 0) {
    throw new Error('Unsupported SMF: SMPTE time code division is not supported');
  }
  const division = divisionRaw === 0 ? 480 : divisionRaw;

  // Skip any extra header bytes if headerLen > 6
  if (headerLen > 6) {
    reader.skip(headerLen - 6);
  }

  const rawEvents: RawEvent[] = [];
  const tempoChanges: Array<{ tick: number; uspqn: number }> = [{ tick: 0, uspqn: 500_000 }];
  let eventCounter = 0;
  let tracksParsed = 0;

  while (tracksParsed < trackCount && reader.remaining >= 8) {
    const chunkType = [reader.readU8(), reader.readU8(), reader.readU8(), reader.readU8()];
    const isMTrk =
      chunkType[0] === 0x4d &&
      chunkType[1] === 0x54 &&
      chunkType[2] === 0x72 &&
      chunkType[3] === 0x6b;

    const chunkLen = reader.readU32();
    if (!isMTrk) {
      reader.skip(chunkLen);
      continue;
    }

    const trackStart = reader.pos;
    const trackEnd = trackStart + chunkLen;
    if (trackEnd > bytes.length) {
      throw new Error('Invalid SMF: truncated track chunk');
    }

    let currentTick = 0;
    let runningStatus = 0;

    while (reader.pos < trackEnd) {
      const delta = reader.readVlq();
      currentTick += delta;

      if (reader.pos >= trackEnd) {
        break;
      }

      let status = bytes[reader.pos];
      if ((status & 0x80) !== 0) {
        reader.readU8(); // Consume status byte
        if (status < 0xf0) {
          runningStatus = status;
        } else {
          runningStatus = 0;
        }
      } else {
        if (runningStatus === 0) {
          throw new Error('Invalid SMF: running status without previous status');
        }
        status = runningStatus;
      }

      if (status === 0xff) {
        // Meta event
        const metaType = reader.readU8();
        const metaLen = reader.readVlq();
        if (metaType === 0x51 && metaLen === 3) {
          // Set tempo
          const b0 = reader.readU8();
          const b1 = reader.readU8();
          const b2 = reader.readU8();
          const uspqn = ((b0 << 16) | (b1 << 8) | b2) >>> 0;
          tempoChanges.push({ tick: currentTick, uspqn: uspqn > 0 ? uspqn : 500_000 });
        } else {
          reader.skip(metaLen);
        }
      } else if (status === 0xf0 || status === 0xf7) {
        // SysEx
        const sysexLen = reader.readVlq();
        reader.skip(sysexLen);
      } else {
        const cmd = status & 0xf0;
        const channel = status & 0x0f;

        if (cmd === 0x80) {
          // NoteOff
          const key = reader.readU8() & 0x7f;
          const vel = reader.readU8() & 0x7f;
          rawEvents.push({
            tick: currentTick,
            order: eventCounter++,
            ev: { type: 'noteOff', timeSeconds: 0, tick: currentTick, channel, midiKey: key, velocity: vel },
          });
        } else if (cmd === 0x90) {
          // NoteOn (velocity 0 is strictly NoteOff!)
          const key = reader.readU8() & 0x7f;
          const vel = reader.readU8() & 0x7f;
          if (vel === 0) {
            rawEvents.push({
              tick: currentTick,
              order: eventCounter++,
              ev: { type: 'noteOff', timeSeconds: 0, tick: currentTick, channel, midiKey: key, velocity: 0 },
            });
          } else {
            rawEvents.push({
              tick: currentTick,
              order: eventCounter++,
              ev: { type: 'noteOn', timeSeconds: 0, tick: currentTick, channel, midiKey: key, velocity: vel },
            });
          }
        } else if (cmd === 0xa0) {
          reader.skip(2); // Polyphonic Aftertouch
        } else if (cmd === 0xb0) {
          const controller = reader.readU8() & 0x7f;
          const value = reader.readU8() & 0x7f;
          rawEvents.push({
            tick: currentTick,
            order: eventCounter++,
            ev: { type: 'cc', timeSeconds: 0, tick: currentTick, channel, controller, value },
          });
        } else if (cmd === 0xc0) {
          reader.skip(1); // Program Change
        } else if (cmd === 0xd0) {
          reader.skip(1); // Channel Pressure
        } else if (cmd === 0xe0) {
          reader.skip(2); // Pitch Wheel
        }
      }
    }

    tracksParsed++;
    reader.seek(trackEnd);
  }

  // Build tempo timeline
  tempoChanges.sort((a, b) => a.tick - b.tick);
  const tempoPoints: TempoPoint[] = [];
  let currentSec = 0;
  let lastTick = 0;
  let currentUspqn = 500_000;

  for (const tc of tempoChanges) {
    if (tc.tick > lastTick) {
      const deltaTicks = tc.tick - lastTick;
      currentSec += (deltaTicks * currentUspqn) / (division * 1_000_000);
      lastTick = tc.tick;
    }
    currentUspqn = tc.uspqn;
    tempoPoints.push({ tick: tc.tick, timeSeconds: currentSec, uspqn: currentUspqn });
  }

  function tickToSeconds(t: number): number {
    let p = tempoPoints[0];
    for (let i = 1; i < tempoPoints.length; i++) {
      if (tempoPoints[i].tick <= t) {
        p = tempoPoints[i];
      } else {
        break;
      }
    }
    const deltaTicks = t - p.tick;
    return p.timeSeconds + (deltaTicks * p.uspqn) / (division * 1_000_000);
  }

  // Assign timeSeconds to each event
  for (const item of rawEvents) {
    item.ev.timeSeconds = tickToSeconds(item.tick);
  }

  // Sort events chronologically. For identical timestamps: NoteOff (0) & CC (1) before NoteOn (2)
  rawEvents.sort((a, b) => {
    if (Math.abs(a.ev.timeSeconds - b.ev.timeSeconds) > 1e-6) {
      return a.ev.timeSeconds - b.ev.timeSeconds;
    }
    const prioA = a.ev.type === 'noteOff' ? 0 : a.ev.type === 'cc' ? 1 : 2;
    const prioB = b.ev.type === 'noteOff' ? 0 : b.ev.type === 'cc' ? 1 : 2;
    if (prioA !== prioB) {
      return prioA - prioB;
    }
    return a.order - b.order;
  });

  const finalEvents = rawEvents.map((r) => r.ev);
  const duration = finalEvents.length > 0 ? finalEvents[finalEvents.length - 1].timeSeconds + 0.5 : 0;

  return {
    format,
    trackCount,
    division,
    durationSeconds: duration,
    events: finalEvents,
  };
}
