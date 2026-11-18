import type { Page } from '@playwright/test';

/** Encodes an unsigned integer as a MIDI Variable-Length Quantity (VLQ). */
function encodeVlq(val: number): number[] {
  const bytes = [val & 0x7f];
  while ((val >>= 7) > 0) {
    bytes.unshift((val & 0x7f) | 0x80);
  }
  return bytes;
}

/** Builds an in-memory Format 0 SMF binary buffer from raw events. */
export function buildMidiFile(events: Array<{ delta: number; data: number[] }>): Uint8Array {
  const trackBytes: number[] = [];
  // Set tempo 120 BPM: delta=0, 0xFF 0x51 0x03 0x07 0xA1 0x20
  trackBytes.push(0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20);
  for (const ev of events) {
    trackBytes.push(...encodeVlq(ev.delta), ...ev.data);
  }
  trackBytes.push(0x00, 0xff, 0x2f, 0x00);

  const header = [
    0x4d, 0x54, 0x68, 0x64, // 'MThd'
    0x00, 0x00, 0x00, 0x06, // length = 6
    0x00, 0x00,             // format = 0
    0x00, 0x01,             // 1 track
    0x01, 0xe0,             // 480 ticks/quarter
    0x4d, 0x54, 0x72, 0x6b, // 'MTrk'
    (trackBytes.length >> 24) & 0xff,
    (trackBytes.length >> 16) & 0xff,
    (trackBytes.length >> 8) & 0xff,
    trackBytes.length & 0xff,
    ...trackBytes,
  ];
  return new Uint8Array(header);
}

/** Injects a mock Web MIDI input device into navigator.requestMIDIAccess. */
export async function injectMockMidi(page: Page): Promise<void> {
  await page.evaluate(() => {
    class MockMIDIInput extends EventTarget {
      readonly id = 'mock-piano-in';
      readonly name = 'Mock Piano MIDI Controller';
      readonly state = 'connected';
      readonly type = 'input';
      onmidimessage: ((e: unknown) => void) | null = null;
      send(data: number[]) {
        const ev = new Event('midimessage') as Event & { data: Uint8Array; receivedTime: number };
        ev.data = new Uint8Array(data);
        ev.receivedTime = performance.now();
        this.dispatchEvent(ev);
        if (this.onmidimessage) this.onmidimessage(ev);
      }
    }
    const mock = new MockMIDIInput();
    const map = new Map<string, unknown>([[mock.id, mock]]);
    (window as unknown as { __MOCK_MIDI__: MockMIDIInput }).__MOCK_MIDI__ = mock;
    (navigator as unknown as { requestMIDIAccess: () => Promise<unknown> }).requestMIDIAccess =
      async () => ({
        inputs: map,
        outputs: new Map(),
        sysexEnabled: false,
      });
  });
}

/** Dispatches raw MIDI bytes via the mock Web MIDI subsystem. */
export async function dispatchMockMidi(page: Page, bytes: number[]): Promise<void> {
  await page.evaluate((b) => {
    const mock = (window as unknown as { __MOCK_MIDI__?: { send: (data: number[]) => void } })
      .__MOCK_MIDI__;
    mock?.send(b);
  }, bytes);
}

/** Passively records outbound worklet messages without altering real AudioWorklet execution. */
export async function tapWorkletMessages(page: Page): Promise<void> {
  await page.evaluate(() => {
    const port = window.__PIANO_DIAG__?.workletNode?.port;
    if (!port) throw new Error('Missing worklet port');
    const orig = port.postMessage.bind(port);
    const log: Array<{ type: string; releaseVelU16?: number; sampleOffset?: number }> = [];
    (window as unknown as { __MSG_LOG__: typeof log }).__MSG_LOG__ = log;
    port.postMessage = (msg: { type: string; releaseVelU16?: number; sampleOffset?: number }) => {
      log.push({ type: msg.type, releaseVelU16: msg.releaseVelU16, sampleOffset: msg.sampleOffset });
      return orig(msg);
    };
  });
}
