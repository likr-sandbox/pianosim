import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { expect, test } from '@playwright/test';

interface StreamPlayerInstance {
  playFrom(scoreSec: number): 'playing' | 'buffering';
  seek(scoreSec: number): 'playing' | 'buffering' | 'idle';
  appendChunk(
    cacheKey: string,
    chunk: {
      jobId: number;
      chunkIndex: number;
      startFrame: number;
      frames: number;
      totalFrames: number;
      sampleRate: number;
      left: Float32Array;
      right: Float32Array;
      isFinal: boolean;
    },
  ): void;
  isBuffering(): boolean;
  isPlaying(): boolean;
}

function createHarness() {
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pianosim-stream-unit-'));
  const bundlePath = path.join(tmpDir, 'offline-stream-player.mjs');
  execFileSync(
    path.join(repoRoot, 'node_modules/.bin/esbuild'),
    [
      path.join(repoRoot, 'apps/web/src/audio/offline-stream-player.ts'),
      '--bundle',
      '--format=esm',
      '--platform=node',
      `--outfile=${bundlePath}`,
    ],
    { stdio: 'pipe' },
  );
  return {
    bundlePath,
    cleanup: () => fs.rmSync(tmpDir, { recursive: true, force: true }),
  };
}

function createMockHost() {
  const scheduled: Array<{
    len: number;
    firstSampleL: number;
    onended: (() => void) | null;
  }> = [];
  const mockHost = {
    getAudioContext: () => ({
      currentTime: 10.0,
      createBuffer: (_ch: number, len: number, _sr: number) => {
        const ch0 = new Float32Array(len);
        return {
          copyToChannel: (src: Float32Array, ch: number) => {
            if (ch === 0) ch0.set(src);
          },
          ch0,
        };
      },
      createBufferSource: () => {
        const entry = { len: 0, firstSampleL: 0, onended: null as (() => void) | null };
        const src = {
          buffer: null as { ch0: Float32Array } | null,
          playbackRate: { value: 1 },
          set onended(fn: (() => void) | null) {
            entry.onended = fn;
          },
          get onended() {
            return entry.onended;
          },
          connect: () => {},
          disconnect: () => {},
          start: () => {
            entry.len = src.buffer?.ch0.length ?? 0;
            entry.firstSampleL = src.buffer?.ch0[0] ?? 0;
            scheduled.push(entry);
          },
          stop: () => {},
        };
        return src;
      },
    }),
    getAnalyser: () => ({}),
  };
  return { mockHost, scheduled };
}

function mkRamp(start: number, frames: number): Float32Array {
  const buf = new Float32Array(frames);
  for (let i = 0; i < frames; i++) buf[i] = start + i;
  return buf;
}

test('OfflineStreamPlayer unit: underrun prebuffer resumes at exact nextScheduleFrame', async () => {
  const h = createHarness();
  try {
    const mod = (await import(pathToFileURL(h.bundlePath).href)) as {
      OfflineStreamPlayer: new (
        host: unknown,
        callbacks?: { onBufferingChange?(b: boolean): void; onEnded?(): void },
      ) => StreamPlayerInstance;
    };
    const { mockHost, scheduled } = createMockHost();
    const bufferingEvents: boolean[] = [];
    const player = new mod.OfflineStreamPlayer(mockHost, {
      onBufferingChange: (b) => bufferingEvents.push(b),
    });

    const sr = 48_000;
    const totalFrames = 144_000;
    player.playFrom(0);
    player.appendChunk('k1', {
      jobId: 1,
      chunkIndex: 0,
      startFrame: 0,
      frames: 48_000,
      totalFrames,
      sampleRate: sr,
      left: mkRamp(0, 48_000),
      right: mkRamp(0, 48_000),
      isFinal: false,
    });
    expect(scheduled.length).toBe(1);
    expect(scheduled[0].len).toBe(48_000);

    // Trigger underrun at nextScheduleFrame === 48_000 (watermark = 72_000).
    scheduled[0].onended?.();
    expect(player.isBuffering()).toBe(true);
    expect(bufferingEvents).toEqual([false, true]);

    // Chunk 1 reaches 60_000 (< 72_000 watermark): remain buffering.
    player.appendChunk('k1', {
      jobId: 1,
      chunkIndex: 1,
      startFrame: 48_000,
      frames: 12_000,
      totalFrames,
      sampleRate: sr,
      left: mkRamp(48_000, 12_000),
      right: mkRamp(48_000, 12_000),
      isFinal: false,
    });
    expect(player.isBuffering()).toBe(true);
    expect(scheduled.length).toBe(1);

    // Chunk 2 exceeds watermark (80_000 > 72_000): must resume at frame 48_000, not 72_000.
    player.appendChunk('k1', {
      jobId: 1,
      chunkIndex: 2,
      startFrame: 60_000,
      frames: 20_000,
      totalFrames,
      sampleRate: sr,
      left: mkRamp(60_000, 20_000),
      right: mkRamp(60_000, 20_000),
      isFinal: false,
    });
    expect(player.isBuffering()).toBe(false);
    expect(bufferingEvents).toEqual([false, true, false]);
    expect(scheduled.length).toBe(2);
    expect(scheduled[1].firstSampleL).toBe(48_000);
    expect(scheduled[1].len).toBe(32_000);
  } finally {
    h.cleanup();
  }
});

test('OfflineStreamPlayer unit: isFinal below watermark flushes buffering and future seek emits onEnded', async () => {
  const h = createHarness();
  try {
    const mod = (await import(pathToFileURL(h.bundlePath).href)) as {
      OfflineStreamPlayer: new (
        host: unknown,
        callbacks?: { onBufferingChange?(b: boolean): void; onEnded?(): void },
      ) => StreamPlayerInstance;
    };

    // (i) isFinal: true arriving while isBuffering() === true below bufferingWatermarkFrame.
    {
      const { mockHost, scheduled } = createMockHost();
      const bufferingEvents: boolean[] = [];
      let endedCount = 0;
      const player = new mod.OfflineStreamPlayer(mockHost, {
        onBufferingChange: (b) => bufferingEvents.push(b),
        onEnded: () => {
          endedCount++;
        },
      });
      const sr = 48_000;
      const totalFrames = 60_000;
      player.playFrom(0);
      player.appendChunk('k-final', {
        jobId: 2,
        chunkIndex: 0,
        startFrame: 0,
        frames: 20_000,
        totalFrames,
        sampleRate: sr,
        left: mkRamp(0, 20_000),
        right: mkRamp(0, 20_000),
        isFinal: false,
      });
      // Underrun at 20_000 sets bufferingWatermarkFrame = 44_000.
      scheduled[0].onended?.();
      expect(player.isBuffering()).toBe(true);

      // Final short chunk ends at 40_000 <= 44_000 watermark, yet isFinal flushes buffering.
      player.appendChunk('k-final', {
        jobId: 2,
        chunkIndex: 1,
        startFrame: 20_000,
        frames: 20_000,
        totalFrames,
        sampleRate: sr,
        left: mkRamp(20_000, 20_000),
        right: mkRamp(20_000, 20_000),
        isFinal: true,
      });
      expect(player.isBuffering()).toBe(false);
      expect(bufferingEvents).toEqual([false, true, false]);
      expect(scheduled.length).toBe(2);
      expect(scheduled[1].firstSampleL).toBe(20_000);
      expect(scheduled[1].len).toBe(20_000);
      expect(endedCount).toBe(0);
    }

    // (ii) seek() into unrendered future followed by chunk arrival and natural onEnded.
    {
      const { mockHost, scheduled } = createMockHost();
      const bufferingEvents: boolean[] = [];
      let endedCount = 0;
      const player = new mod.OfflineStreamPlayer(mockHost, {
        onBufferingChange: (b) => bufferingEvents.push(b),
        onEnded: () => {
          endedCount++;
        },
      });
      const sr = 48_000;
      const totalFrames = 144_000;
      player.playFrom(0);
      player.appendChunk('k-seek', {
        jobId: 3,
        chunkIndex: 0,
        startFrame: 0,
        frames: 48_000,
        totalFrames,
        sampleRate: sr,
        left: mkRamp(0, 48_000),
        right: mkRamp(0, 48_000),
        isFinal: false,
      });
      expect(scheduled.length).toBe(1);

      const seekState = player.seek(2.0); // frame 96_000 (> rendered 48_000)
      expect(seekState).toBe('buffering');
      expect(player.isBuffering()).toBe(true);

      // Chunk 1 reaches 96_000 (not past 96_000 watermark yet): remain buffering.
      player.appendChunk('k-seek', {
        jobId: 3,
        chunkIndex: 1,
        startFrame: 48_000,
        frames: 48_000,
        totalFrames,
        sampleRate: sr,
        left: mkRamp(48_000, 48_000),
        right: mkRamp(48_000, 48_000),
        isFinal: false,
      });
      expect(player.isBuffering()).toBe(true);

      // Chunk 2 reaches totalFrames (144_000) with isFinal: resumes at 96_000 and ends cleanly.
      player.appendChunk('k-seek', {
        jobId: 3,
        chunkIndex: 2,
        startFrame: 96_000,
        frames: 48_000,
        totalFrames,
        sampleRate: sr,
        left: mkRamp(96_000, 48_000),
        right: mkRamp(96_000, 48_000),
        isFinal: true,
      });
      expect(player.isBuffering()).toBe(false);
      expect(bufferingEvents).toEqual([false, true, false]);
      expect(scheduled.length).toBe(2);
      expect(scheduled[1].firstSampleL).toBe(96_000);
      expect(scheduled[1].len).toBe(48_000);

      scheduled[1].onended?.();
      expect(endedCount).toBe(1);
      expect(player.isPlaying()).toBe(false);
    }
  } finally {
    h.cleanup();
  }
});
