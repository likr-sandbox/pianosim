import {
  WEB_AUDIO_SAMPLE_RATE,
  WORKLET_PROCESSOR_NAME,
  type PerfPresetId,
  type PerfProfileConfig,
  type PrepInboundMessage,
  type PrepOutboundMessage,
  type WorkletInboundMessage,
  type WorkletOutboundMessage,
} from './abi';
import prepWorkerCode from './prep-worker.ts?blobworker';
import { TelemetryClient } from './telemetry-client';
import { compileWasmModule } from './wasm-loader';
import workletCode from './worklet-processor.ts?worklet';

const INIT_TIMEOUT_MS = 10_000;
const ANALYSER_FFT_SIZE = 2048;

export interface AudioPipelineHandles {
  readonly audioContext: AudioContext;
  readonly workletNode: AudioWorkletNode;
  readonly prepWorker: Worker;
  readonly analyser: AnalyserNode;
}

export interface PipelineCallbacks {
  readonly telemetry: TelemetryClient;
  readonly perfPreset: PerfPresetId;
  readonly perfConfig: PerfProfileConfig;
  readonly onSpaceDone: () => void;
  readonly onVoicingDone: () => void;
  readonly onRecChunk?: (samples: ArrayBuffer, frames: number) => void;
  readonly onRecStopped?: () => void;
}

/** Boots AudioContext (48 kHz), AudioWorklet, AnalyserNode, and Prep Web Worker with MessageChannel IPC. */
export async function bootstrapAudioPipeline(
  callbacks: PipelineCallbacks,
): Promise<AudioPipelineHandles> {
  const { telemetry } = callbacks;
  let audioContext: AudioContext | null = null;
  let workletNode: AudioWorkletNode | null = null;
  let prepWorker: Worker | null = null;
  let analyser: AnalyserNode | null = null;

  try {
    const AudioCtx =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) {
      throw new Error('Web Audio API is not supported in this browser.');
    }

    audioContext = new AudioCtx({
      latencyHint: 'interactive',
      sampleRate: WEB_AUDIO_SAMPLE_RATE,
    });
    audioContext.onstatechange = () => {
      telemetry.syncContextState();
    };
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }
    if (typeof audioContext.audioWorklet === 'undefined') {
      throw new Error('AudioWorklet is unavailable in this execution context.');
    }

    const workletBlobUrl = URL.createObjectURL(
      new Blob([workletCode], { type: 'application/javascript' }),
    );
    const workerBlobUrl = URL.createObjectURL(
      new Blob([prepWorkerCode], { type: 'application/javascript' }),
    );
    let wasmModule: WebAssembly.Module;
    try {
      const [, compiled] = await Promise.all([
        audioContext.audioWorklet.addModule(workletBlobUrl),
        compileWasmModule(),
      ]);
      wasmModule = compiled;
      prepWorker = new Worker(workerBlobUrl, { type: 'module' });
    } finally {
      URL.revokeObjectURL(workletBlobUrl);
      URL.revokeObjectURL(workerBlobUrl);
    }

    workletNode = new AudioWorkletNode(audioContext, WORKLET_PROCESSOR_NAME, {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [2],
    });

    analyser = audioContext.createAnalyser();
    analyser.fftSize = ANALYSER_FFT_SIZE;
    analyser.smoothingTimeConstant = 0;
    workletNode.connect(analyser);
    analyser.connect(audioContext.destination);

    telemetry.setAudioGraph(audioContext, analyser, workletNode);

    const channel = new MessageChannel();
    await Promise.all([
      handshakeWorklet(workletNode, wasmModule, channel.port2, telemetry, callbacks),
      handshakePrepWorker(prepWorker, wasmModule, audioContext.sampleRate, channel.port1, callbacks),
    ]);

    return { audioContext, workletNode, prepWorker, analyser };
  } catch (err) {
    if (prepWorker) prepWorker.terminate();
    if (workletNode) workletNode.disconnect();
    if (analyser) analyser.disconnect();
    if (audioContext) await audioContext.close().catch(() => undefined);
    telemetry.setAudioGraph(null, null, null);
    throw err;
  }
}

function handshakeWorklet(
  node: AudioWorkletNode,
  wasmModule: WebAssembly.Module,
  prepPort: MessagePort,
  telemetry: TelemetryClient,
  callbacks: PipelineCallbacks,
): Promise<void> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const timer = window.setTimeout(() => {
      if (!settled) {
        settled = true;
        reject(new Error('Timed out waiting for Wasm AudioWorklet initialization.'));
      }
    }, INIT_TIMEOUT_MS);

    node.onprocessorerror = (event) => {
      const msg = `AudioWorklet processor error: ${String(event)}`;
      telemetry.setStatus('error', msg);
      if (!settled) {
        settled = true;
        window.clearTimeout(timer);
        reject(new Error(msg));
      }
    };

    node.port.onmessage = (event: MessageEvent<WorkletOutboundMessage>) => {
      const msg = event.data;
      if (msg.type === 'ready') {
        if (!settled) {
          settled = true;
          window.clearTimeout(timer);
          resolve();
        }
        return;
      }
      if (msg.type === 'error') {
        telemetry.setStatus('error', msg.message);
        if (!settled) {
          settled = true;
          window.clearTimeout(timer);
          reject(new Error(msg.message));
        }
        return;
      }
      if (msg.type === 'telemetry') {
        telemetry.updateTelemetry({
          activeVoices: msg.activeVoices,
          peakL: msg.peakL,
          peakR: msg.peakR,
          rmsMono: msg.rmsMono,
          callbackUs: msg.callbackUs,
          maxCallbackUs: msg.maxCallbackUs,
          p50Us: msg.p50Us,
          p99Us: msg.p99Us,
          overrunCount: msg.overrunCount,
          strideOverruns: msg.strideOverruns,
          singleBlockOverruns: msg.singleBlockOverruns,
          stallCount: msg.stallCount,
          timingSource: msg.timingSource,
          clipped: msg.clipped,
          blockCount: msg.blockCount,
        });
        return;
      }
      if (msg.type === 'recChunk') {
        callbacks.onRecChunk?.(msg.samples, msg.frames);
        return;
      }
      if (msg.type === 'prepApplied') {
        telemetry.updatePrepApplied(msg.spaceSeq, msg.voicingSeq);
        return;
      }
      if (msg.type === 'perfApplied') {
        telemetry.updatePerfApplied(msg.perfSeq, msg.presetId, msg.perfConfig);
        return;
      }
      if (msg.type === 'recStopped') {
        callbacks.onRecStopped?.();
      }
    };

    node.port.postMessage(
      {
        type: 'init',
        wasmModule,
        prepPort,
        perfPreset: callbacks.perfPreset,
        perfConfig: callbacks.perfConfig,
      } satisfies WorkletInboundMessage,
      [prepPort],
    );
  });
}

function handshakePrepWorker(
  worker: Worker,
  wasmModule: WebAssembly.Module,
  sampleRate: number,
  workletPort: MessagePort,
  callbacks: PipelineCallbacks,
): Promise<void> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const timer = window.setTimeout(() => {
      if (!settled) {
        settled = true;
        reject(new Error('Timed out waiting for Prep Web Worker initialization.'));
      }
    }, INIT_TIMEOUT_MS);

    worker.onerror = (ev) => {
      const msg = `Prep Worker error: ${ev.message}`;
      if (!settled) {
        settled = true;
        window.clearTimeout(timer);
        reject(new Error(msg));
      }
    };

    worker.onmessage = (event: MessageEvent<PrepOutboundMessage>) => {
      const msg = event.data;
      if (msg.type === 'ready') {
        if (!settled) {
          settled = true;
          window.clearTimeout(timer);
          resolve();
        }
        return;
      }
      if (msg.type === 'spaceDone') {
        callbacks.onSpaceDone();
        return;
      }
      if (msg.type === 'voicingDone') {
        callbacks.onVoicingDone();
        return;
      }
      if (msg.type === 'error') {
        if (!settled) {
          settled = true;
          window.clearTimeout(timer);
          reject(new Error(msg.message));
        }
      }
    };

    worker.postMessage(
      {
        type: 'init',
        wasmModule,
        sampleRate,
        workletPort,
        perfConfig: callbacks.perfConfig,
      } satisfies PrepInboundMessage,
      [workletPort],
    );
  });
}
