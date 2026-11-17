/** Web MIDI API manager supporting 14-bit high-res velocity (MIDI CA-031) and continuous CC pedals. */

import {
  cc7ToU16,
  PEDAL_SOSTENUTO,
  PEDAL_SUSTAIN,
  PEDAL_UNA_CORDA,
  u7ToU16,
  WEB_AUDIO_SAMPLE_RATE,
} from '../audio/abi';
import type { WasmAudioHost } from '../audio/wasm-host';
import {
  applyTouchCurve,
  applyTouchCurveNormalized,
  DEFAULT_TOUCH_CURVE,
  type TouchCurveMode,
} from '../input/touch-curve';

const JITTER_WINDOW_MS = 3.0;
const JITTER_GAP_RESET_MS = 100.0;
const MAX_JITTER_LOOKAHEAD_MS = 60.0;

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
  onMidiActivity?(): void;
}

export class WebMidiManager {
  private readonly host: WasmAudioHost;
  private callbacks: WebMidiCallbacks;

  private midiAccess: MidiAccessInterface | null = null;
  private selectedInputId: string | null = null;
  private highResVelocityPrefix: number | null = null;
  private isInitialized = false;
  private runningStatus: number = 0;
  private touchCurve: TouchCurveMode = DEFAULT_TOUCH_CURVE;
  private lastReceivedMs: number | null = null;
  private jitterClockOffsetMs = 0;
  private readonly boundInputs = new Map<MidiInputDevice, (ev: Event) => void>();

  constructor(host: WasmAudioHost, callbacks: WebMidiCallbacks = {}) {
    this.host = host;
    this.callbacks = callbacks;
  }

  setCallbacks(callbacks: WebMidiCallbacks): void {
    this.callbacks = callbacks;
    this.callbacks.onDevicesChanged?.(this.getDevices());
  }

  setTouchCurve(mode: TouchCurveMode): void {
    this.touchCurve = mode;
  }

  getTouchCurve(): TouchCurveMode {
    return this.touchCurve;
  }

  getDevices(): Array<{ id: string; name: string }> {
    const devices: Array<{ id: string; name: string }> = [];
    const seenIds = new Set<string>();
    const inputs = this.midiAccess?.inputs;
    if (inputs && typeof inputs.forEach === 'function') {
      inputs.forEach((input: MidiInputDevice) => {
        if (!seenIds.has(input.id)) {
          seenIds.add(input.id);
          devices.push({
            id: input.id,
            name: input.name || `MIDI Input (${input.id})`,
          });
        }
      });
    }
    if (typeof window !== 'undefined') {
      const mock = (window as unknown as { __MOCK_MIDI__?: MidiInputDevice }).__MOCK_MIDI__;
      if (mock && !seenIds.has(mock.id)) {
        seenIds.add(mock.id);
        devices.push({
          id: mock.id,
          name: mock.name || `MIDI Input (${mock.id})`,
        });
      }
    }
    return devices;
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
          this.bindAllInputs();
        };
        this.bindAllInputs();
      }
      this.updateDeviceList();
    } catch {
      // Web MIDI unsupported or permission denied
    }
  }

  selectInput(id: string): void {
    this.selectedInputId = id === '' ? null : id;
  }

  private updateDeviceList(): void {
    this.callbacks.onDevicesChanged?.(this.getDevices());
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
    if (this.boundInputs.has(input)) {
      return;
    }
    const handler = (event: Event) => {
      if (this.selectedInputId && this.selectedInputId !== input.id) {
        return;
      }
      const midiEvent = event as unknown as { data?: Uint8Array; receivedTime?: number };
      if (midiEvent && midiEvent.data) {
        const receivedMs =
          typeof midiEvent.receivedTime === 'number' && Number.isFinite(midiEvent.receivedTime)
            ? midiEvent.receivedTime
            : undefined;
        this.handleMidiBytes(midiEvent.data, receivedMs);
      }
    };
    this.boundInputs.set(input, handler);
    if (typeof input.addEventListener === 'function') {
      input.addEventListener('midimessage', handler);
    } else {
      input.onmidimessage = handler;
    }
  }

  private checkMockDevice(): void {
    if (typeof window === 'undefined') return;
    const mock = (window as unknown as { __MOCK_MIDI__?: MidiInputDevice }).__MOCK_MIDI__;
    if (mock) {
      this.bindInput(mock);
    }
  }

  private computeSampleOffset(receivedTimeMs?: number): number {
    if (receivedTimeMs === undefined || !Number.isFinite(receivedTimeMs)) {
      return 0;
    }
    const nowMs =
      typeof performance !== 'undefined' && typeof performance.now === 'function'
        ? performance.now()
        : Date.now();
    const ctx = this.host.getAudioContext();
    const sampleRate = ctx ? ctx.sampleRate : WEB_AUDIO_SAMPLE_RATE;

    if (
      this.lastReceivedMs === null ||
      receivedTimeMs - this.lastReceivedMs > JITTER_GAP_RESET_MS ||
      receivedTimeMs < this.lastReceivedMs
    ) {
      this.jitterClockOffsetMs =
        Math.abs(nowMs - receivedTimeMs) > 50.0 ? nowMs - receivedTimeMs : 0;
    }
    this.lastReceivedMs = receivedTimeMs;

    const targetTimeMs = receivedTimeMs + this.jitterClockOffsetMs + JITTER_WINDOW_MS;
    const delayMs = targetTimeMs - nowMs;
    const maxSamples = Math.round((MAX_JITTER_LOOKAHEAD_MS / 1000) * sampleRate);
    return Math.max(0, Math.min(maxSamples, Math.round((delayMs * sampleRate) / 1000)));
  }

  handleMidiBytes(bytes: Uint8Array | ArrayLike<number>, receivedTimeMs?: number): void {
    if (!bytes || bytes.length === 0) return;

    if (!this.host.isReady()) {
      void this.host.unlock();
    }
    this.callbacks.onMidiActivity?.();

    const sampleOffset = this.computeSampleOffset(receivedTimeMs);
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
          const valU16 = cc7ToU16(val);
          this.host.pedal(PEDAL_SUSTAIN, valU16, sampleOffset);
          this.callbacks.onPedalChange?.(PEDAL_SUSTAIN, val);
        } else if (cc === 66) {
          const valU16 = val >= 64 ? 65535 : 0;
          this.host.pedal(PEDAL_SOSTENUTO, valU16, sampleOffset);
          this.callbacks.onPedalChange?.(PEDAL_SOSTENUTO, val);
        } else if (cc === 67) {
          const valU16 = cc7ToU16(val);
          this.host.pedal(PEDAL_UNA_CORDA, valU16, sampleOffset);
          this.callbacks.onPedalChange?.(PEDAL_UNA_CORDA, val);
        }
      } else if (cmd === 0x80) {
        // Note Off: 2 data bytes (key, release vel) or 1 data byte (key)
        if (offset >= bytes.length || (bytes[offset] & 0x80) !== 0) {
          continue;
        }
        const key = bytes[offset] & 0x7f;
        offset++;
        let relVel7 = 0;
        let hasRelVel = false;
        if (offset < bytes.length && (bytes[offset] & 0x80) === 0) {
          relVel7 = bytes[offset] & 0x7f;
          hasRelVel = true;
          offset++;
        }
        const relU16 = hasRelVel ? u7ToU16(relVel7) : 32768;
        this.host.noteOff(key, relU16, sampleOffset);
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
          this.host.noteOff(key, 32768, sampleOffset);
          this.highResVelocityPrefix = null;
          this.callbacks.onNoteRelease?.(key);
        } else {
          let velU16: number;
          let displayVel7 = val7;
          if (this.highResVelocityPrefix !== null) {
            const vel14 = (val7 << 7) | this.highResVelocityPrefix;
            const norm = applyTouchCurveNormalized(vel14 / 16383, this.touchCurve);
            velU16 = Math.max(1, Math.min(65535, Math.round(norm * 65535)));
            displayVel7 = Math.max(1, Math.min(127, Math.round(norm * 127)));
            this.highResVelocityPrefix = null;
          } else {
            displayVel7 = applyTouchCurve(val7, this.touchCurve);
            velU16 = u7ToU16(displayVel7);
          }
          this.host.noteOn(key, velU16, sampleOffset);
          this.callbacks.onNotePress?.(key, displayVel7);
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
    for (const [input, handler] of this.boundInputs) {
      if (typeof input.removeEventListener === 'function') {
        input.removeEventListener('midimessage', handler);
      } else if (input.onmidimessage === handler) {
        input.onmidimessage = null;
      }
    }
    this.boundInputs.clear();
    this.midiAccess = null;
    this.selectedInputId = null;
    this.highResVelocityPrefix = null;
    this.runningStatus = 0;
    this.lastReceivedMs = null;
    this.jitterClockOffsetMs = 0;
  }
}
