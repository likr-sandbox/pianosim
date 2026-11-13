/**
 * Browser-side Web Audio lifecycle telemetry (spies on the page's real AudioContext;
 * never replaces audio behaviour).
 */

import type { Page } from '@playwright/test';

/**
 * Install transparent telemetry and audio lifecycle hooks in browser page.
 */
export async function installAudioTelemetry(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const win = window as any;
    win.__pianosim = win.__pianosim || {
      events: [] as any[],
      lastNoteOn: null as any,
      lastNoteOff: null as any,
      sustainPedal: false,
      audioContextInstances: [] as AudioContext[],
      resumed: false,
    };

    // Spy on AudioContext
    if (typeof win.AudioContext !== 'undefined') {
      const OrigAudioContext = win.AudioContext;
      win.AudioContext = class extends OrigAudioContext {
        constructor(...args: any[]) {
          super(...args);
          win.__pianosim.audioContextInstances.push(this);
        }
        async resume(): Promise<void> {
          win.__pianosim.resumed = true;
          return super.resume();
        }
      };
    }
  });
}

/**
 * Retrieve audio telemetry snapshot from page.
 */
export async function getAudioTelemetry(page: Page): Promise<{
  events: any[];
  lastNoteOn: { midiNote: number; velocity: number } | null;
  lastNoteOff: { midiNote: number } | null;
  sustainPedal: boolean;
  resumed: boolean;
  instancesCount: number;
}> {
  return page.evaluate(() => {
    const p = (window as any).__pianosim;
    return {
      events: p?.events || [],
      lastNoteOn: p?.lastNoteOn || null,
      lastNoteOff: p?.lastNoteOff || null,
      sustainPedal: !!p?.sustainPedal,
      resumed: !!p?.resumed,
      instancesCount: p?.audioContextInstances?.length || 0,
    };
  });
}

/**
 * Output tap: observes what the page's real audio graph (UI -> PianoAudioEngine ->
 * AudioWorklet/ScriptProcessor running the production PianoSynthesizer) sends to the
 * AudioContext destination. A passive AnalyserNode is attached in parallel to every node
 * that connects to the destination; nothing is replaced or synthesised.
 */
export async function installAudioTap(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const win = window as any;
    const tap = {
      analysers: [] as AnalyserNode[],
      buf: null as Float32Array | null,
      lastRms: 0,
      maxRms: 0,
      absMax: 0,
      nonFinite: 0,
      polls: 0,
      sampleRate: 0,
      reset() {
        tap.maxRms = 0;
        tap.absMax = 0;
        tap.nonFinite = 0;
      },
    };
    win.__tap = tap;

    const proto = AudioNode.prototype as any;
    const origConnect = proto.connect;
    proto.connect = function (this: any, dest: any, ...rest: any[]) {
      const result = origConnect.call(this, dest, ...rest);
      if (typeof AudioDestinationNode !== 'undefined' && dest instanceof AudioDestinationNode && !this.__tapped) {
        this.__tapped = true;
        const an: AnalyserNode = this.context.createAnalyser();
        an.fftSize = 32768;
        origConnect.call(this, an);
        tap.analysers.push(an);
        tap.sampleRate = this.context.sampleRate;
        tap.buf = new Float32Array(an.fftSize);
      }
      return result;
    };

    setInterval(() => {
      const an = tap.analysers[0];
      if (!an || !tap.buf) return;
      an.getFloatTimeDomainData(tap.buf as any);
      tap.polls++;
      const n = 2048; // most recent ~43 ms
      let ss = 0;
      for (let i = tap.buf.length - n; i < tap.buf.length; i++) {
        const v = tap.buf[i];
        if (!Number.isFinite(v)) { tap.nonFinite++; continue; }
        ss += v * v;
        const a = Math.abs(v);
        if (a > tap.absMax) tap.absMax = a;
      }
      tap.lastRms = Math.sqrt(ss / n);
      if (tap.lastRms > tap.maxRms) tap.maxRms = tap.lastRms;
    }, 20);
  });
}

export interface TapStats {
  lastRms: number;
  maxRms: number;
  absMax: number;
  nonFinite: number;
  polls: number;
  sampleRate: number;
}

export async function tapReset(page: Page): Promise<void> {
  await page.evaluate(() => (window as any).__tap.reset());
}

export async function tapStats(page: Page): Promise<TapStats> {
  return page.evaluate(() => {
    const t = (window as any).__tap;
    return {
      lastRms: t.lastRms,
      maxRms: t.maxRms,
      absMax: t.absMax,
      nonFinite: t.nonFinite,
      polls: t.polls,
      sampleRate: t.sampleRate,
    };
  });
}

/** Latest 32768 samples seen at the destination, plus the context sample rate. */
export async function tapSnapshot(page: Page): Promise<{ sampleRate: number; samples: Float32Array }> {
  const r = await page.evaluate(() => {
    const t = (window as any).__tap;
    const an: AnalyserNode = t.analysers[0];
    const b = new Float32Array(an.fftSize);
    an.getFloatTimeDomainData(b as any);
    return { sampleRate: t.sampleRate as number, samples: Array.from(b) };
  });
  return { sampleRate: r.sampleRate, samples: Float32Array.from(r.samples) };
}
