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

  handleMidiBytes(bytes: Uint8Array | number[]): void {
    if (bytes.length === 0) return;
    const status = bytes[0];
    const cmd = status & 0xf0;

    // CC 88 Prefix for High-Resolution Velocity (MIDI CA-031)
    if (cmd === 0xb0 && bytes.length >= 3) {
      const cc = bytes[1];
      const val = bytes[2];
      if (cc === 88) {
        this.highResVelocityPrefix = val & 0x7f;
        return;
      }
      if (cc === 64) {
        // Sustain / Damper pedal
        const valU16 = u7ToU16(val);
        this.host.pedal(PEDAL_SUSTAIN, valU16, 0);
        this.callbacks.onPedalChange?.(PEDAL_SUSTAIN, val);
        return;
      }
      if (cc === 66) {
        // Sostenuto pedal
        const valU16 = val >= 64 ? 65535 : 0;
        this.host.pedal(PEDAL_SOSTENUTO, valU16, 0);
        this.callbacks.onPedalChange?.(PEDAL_SOSTENUTO, val);
        return;
      }
      if (cc === 67) {
        // Una Corda pedal
        const valU16 = u7ToU16(val);
        this.host.pedal(PEDAL_UNA_CORDA, valU16, 0);
        this.callbacks.onPedalChange?.(PEDAL_UNA_CORDA, val);
        return;
      }
    }

    // NoteOff (0x80)
    if (cmd === 0x80 && bytes.length >= 2) {
      const key = bytes[1] & 0x7f;
      this.host.noteOff(key, 0, 0);
      this.highResVelocityPrefix = null;
      this.callbacks.onNoteRelease?.(key);
      return;
    }

    // NoteOn (0x90)
    if (cmd === 0x90 && bytes.length >= 3) {
      const key = bytes[1] & 0x7f;
      const val7 = bytes[2] & 0x7f;

      if (val7 === 0) {
        // Zero-velocity NoteOn is strictly NoteOff
        this.host.noteOff(key, 0, 0);
        this.highResVelocityPrefix = null;
        this.callbacks.onNoteRelease?.(key);
        return;
      }

      let velU16: number;
      if (this.highResVelocityPrefix !== null) {
        // 14-bit high resolution velocity
        const vel14 = (val7 << 7) | this.highResVelocityPrefix;
        velU16 = Math.round((vel14 * 65535) / 16383);
        this.highResVelocityPrefix = null;
      } else {
        velU16 = u7ToU16(val7);
      }

      this.host.noteOn(key, velU16, 0);
      this.callbacks.onNotePress?.(key, val7);
      return;
    }
  }

  destroy(): void {
    this.midiAccess = null;
    this.selectedInputId = null;
    this.highResVelocityPrefix = null;
  }
}
