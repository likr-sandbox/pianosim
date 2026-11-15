/** Web MIDI API manager supporting 14-bit high-res velocity (MIDI CA-031) and continuous CC pedals. */

import {
  PEDAL_SOSTENUTO,
  PEDAL_SUSTAIN,
  PEDAL_UNA_CORDA,
  u7ToU16,
} from '../audio/abi';
import type { WasmAudioHost } from '../audio/wasm-host';

export interface MidiInputDevice {
  readonly id: string;
  readonly name?: string;
  addEventListener?(type: string, listener: (ev: Event) => void): void;
  removeEventListener?(type: string, listener: (ev: Event) => void): void;
  onmidimessage?: ((ev: Event) => void) | null;
}

export interface MidiAccessInterface {
  readonly inputs: {
    forEach(cb: (input: MidiInputDevice, key?: string) => void): void;
  };
  onstatechange?: ((ev: Event) => void) | null;
}

export interface WebMidiCallbacks {
  onNotePress?(midiKey: number, velocity: number): void;
  onNoteRelease?(midiKey: number): void;
  onPedalChange?(pedalKind: number, value7: number): void;
  onDevicesChanged?(devices: Array<{ id: string; name: string }>): void;
}

export class WebMidiManager {
  private readonly host: WasmAudioHost;
  private readonly callbacks: WebMidiCallbacks;

  private midiAccess: MidiAccessInterface | null = null;
  private selectedInputId: string | null = null;
  private highResVelocityPrefix: number | null = null;
  private isInitialized = false;
  private runningStatus: number = 0;

  constructor(host: WasmAudioHost, callbacks: WebMidiCallbacks = {}) {
    this.host = host;
    this.callbacks = callbacks;
  }

  async init(): Promise<void> {
    if (this.isInitialized) {
      return;
    }
    this.isInitialized = true;
    await this.refresh();
  }

  async refresh(): Promise<void> {
    try {
      // Check for mock device first if present in test environment
      this.checkMockDevice();

      const nav = typeof navigator !== 'undefined' ? (navigator as unknown as { requestMIDIAccess?: (opts: { sysex: boolean }) => Promise<MidiAccessInterface> }) : null;
      if (nav && typeof nav.requestMIDIAccess === 'function') {
        const access = await nav.requestMIDIAccess({ sysex: false });
        this.midiAccess = access;
        access.onstatechange = () => {
          this.updateDeviceList();
        };
        this.updateDeviceList();
        this.bindAllInputs();
      }
    } catch {
      // Web MIDI unsupported or permission denied
    }
  }

  selectInput(id: string): void {
    this.selectedInputId = id === '' ? null : id;
  }

  private updateDeviceList(): void {
    if (!this.midiAccess) {
      return;
    }
    const devices: Array<{ id: string; name: string }> = [];
    const inputs = this.midiAccess.inputs;
    if (inputs && typeof inputs.forEach === 'function') {
      inputs.forEach((input: MidiInputDevice) => {
        devices.push({
          id: input.id,
          name: input.name || `MIDI Input (${input.id})`,
        });
      });
    }
    this.callbacks.onDevicesChanged?.(devices);
  }

  private bindAllInputs(): void {
    if (!this.midiAccess) {
      return;
    }
    const inputs = this.midiAccess.inputs;
    if (inputs && typeof inputs.forEach === 'function') {
      inputs.forEach((input: MidiInputDevice) => {
        this.bindInput(input);
      });
    }
  }

  private bindInput(input: MidiInputDevice): void {
    const handler = (event: Event) => {
      if (this.selectedInputId && this.selectedInputId !== input.id) {
        return;
      }
      const midiEvent = event as unknown as { data?: Uint8Array };
      if (midiEvent && midiEvent.data) {
        this.handleMidiBytes(midiEvent.data);
      }
    };
    if (typeof input.addEventListener === 'function') {
      input.addEventListener('midimessage', handler);
    }
    input.onmidimessage = handler;
  }

  private checkMockDevice(): void {
    if (typeof window === 'undefined') return;
    const mock = (window as unknown as { __MOCK_MIDI__?: MidiInputDevice }).__MOCK_MIDI__;
    if (mock) {
      this.bindInput(mock);
    }
  }

  handleMidiBytes(bytes: Uint8Array | ArrayLike<number>): void {
    if (!bytes || bytes.length === 0) return;
    let offset = 0;

    while (offset < bytes.length) {
      const b = bytes[offset];

      // 1. System Real-Time messages (0xF8..0xFF) can appear anywhere and MUST NOT alter running status
      if (b >= 0xf8) {
        offset++;
        continue;
      }

      // 2. Determine status byte vs data byte under active running status
      let status: number;
      if ((b & 0x80) !== 0) {
        // High bit is set: new status byte
        status = b;
        offset++;

        // System Common messages (0xF0..0xF7) reset running status
        if (status >= 0xf0) {
          this.runningStatus = 0;
          if (status === 0xf0) {
            // SysEx: skip data bytes until EOX (0xF7) or end of buffer
            while (offset < bytes.length && bytes[offset] !== 0xf7) {
              offset++;
            }
            if (offset < bytes.length && bytes[offset] === 0xf7) {
              offset++;
            }
          } else if (status === 0xf1 || status === 0xf3) {
            // MTC Quarter Frame or Song Select: 1 data byte
            if (offset < bytes.length && (bytes[offset] & 0x80) === 0) {
              offset++;
            }
          } else if (status === 0xf2) {
            // Song Position Pointer: 2 data bytes
            if (offset < bytes.length && (bytes[offset] & 0x80) === 0) {
              offset++;
            }
            if (offset < bytes.length && (bytes[offset] & 0x80) === 0) {
              offset++;
            }
          }
          // 0xF4..0xF7 have 0 data bytes
          continue;
        }

        // Channel voice message (0x80..0xEF): update active running status
        this.runningStatus = status;
      } else {
        // High bit is clear: data byte using running status
        if (this.runningStatus === 0) {
          // No active running status: discard orphan data byte
          offset++;
          continue;
        }
        status = this.runningStatus;
        // Do NOT increment offset; offset already points to data byte 1
      }

      // 3. Process channel voice message (offset points to first data byte)
      const cmd = status & 0xf0;

      if (cmd === 0xb0) {
        // Control Change: 2 data bytes
        if (
          offset + 1 >= bytes.length ||
          (bytes[offset] & 0x80) !== 0 ||
          (bytes[offset + 1] & 0x80) !== 0
        ) {
          if (offset < bytes.length && (bytes[offset] & 0x80) === 0) {
            offset++;
          }
          continue;
        }
        const cc = bytes[offset] & 0x7f;
        const val = bytes[offset + 1] & 0x7f;
        offset += 2;

        if (cc === 88) {
          this.highResVelocityPrefix = val;
        } else if (cc === 64) {
          const valU16 = u7ToU16(val);
          this.host.pedal(PEDAL_SUSTAIN, valU16, 0);
          this.callbacks.onPedalChange?.(PEDAL_SUSTAIN, val);
        } else if (cc === 66) {
          const valU16 = val >= 64 ? 65535 : 0;
          this.host.pedal(PEDAL_SOSTENUTO, valU16, 0);
          this.callbacks.onPedalChange?.(PEDAL_SOSTENUTO, val);
        } else if (cc === 67) {
          const valU16 = u7ToU16(val);
          this.host.pedal(PEDAL_UNA_CORDA, valU16, 0);
          this.callbacks.onPedalChange?.(PEDAL_UNA_CORDA, val);
        }
      } else if (cmd === 0x80) {
        // Note Off: 2 data bytes (key, release vel) or 1 data byte (key)
        if (offset >= bytes.length || (bytes[offset] & 0x80) !== 0) {
          continue;
        }
        const key = bytes[offset] & 0x7f;
        offset++;
        if (offset < bytes.length && (bytes[offset] & 0x80) === 0) {
          offset++; // Consume release velocity if present
        }
        this.host.noteOff(key, 0, 0);
        this.highResVelocityPrefix = null;
        this.callbacks.onNoteRelease?.(key);
      } else if (cmd === 0x90) {
        // Note On: 2 data bytes (key, velocity)
        if (
          offset + 1 >= bytes.length ||
          (bytes[offset] & 0x80) !== 0 ||
          (bytes[offset + 1] & 0x80) !== 0
        ) {
          if (offset < bytes.length && (bytes[offset] & 0x80) === 0) {
            offset++;
          }
          continue;
        }
        const key = bytes[offset] & 0x7f;
        const val7 = bytes[offset + 1] & 0x7f;
        offset += 2;

        if (val7 === 0) {
          this.host.noteOff(key, 0, 0);
          this.highResVelocityPrefix = null;
          this.callbacks.onNoteRelease?.(key);
        } else {
          let velU16: number;
          if (this.highResVelocityPrefix !== null) {
            const vel14 = (val7 << 7) | this.highResVelocityPrefix;
            velU16 = Math.round((vel14 * 65535) / 16383);
            this.highResVelocityPrefix = null;
          } else {
            velU16 = u7ToU16(val7);
          }
          this.host.noteOn(key, velU16, 0);
          this.callbacks.onNotePress?.(key, val7);
        }
      } else if (cmd === 0xc0 || cmd === 0xd0) {
        // Program Change / Channel Pressure: 1 data byte
        if (offset < bytes.length && (bytes[offset] & 0x80) === 0) {
          offset++;
        }
      } else if (cmd === 0xa0 || cmd === 0xe0) {
        // Poly Pressure / Pitch Bend: 2 data bytes
        if (offset < bytes.length && (bytes[offset] & 0x80) === 0) {
          offset++;
        }
        if (offset < bytes.length && (bytes[offset] & 0x80) === 0) {
          offset++;
        }
      } else {
        offset++;
      }
    }
  }

  destroy(): void {
    this.midiAccess = null;
    this.selectedInputId = null;
    this.highResVelocityPrefix = null;
    this.runningStatus = 0;
  }
}
