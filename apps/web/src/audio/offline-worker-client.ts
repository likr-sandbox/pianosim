/** Worker lifecycle and job cancellation wrapper for offline-render-worker.ts. */

import type {
  OfflineRenderRequest,
  OfflineWorkerInbound,
  OfflineWorkerOutbound,
} from './offline-abi';
import offlineWorkerCode from './offline-render-worker.ts?blobworker';
import { compileWasmModule } from './wasm-loader';

export interface OfflineChunkEvent {
  readonly jobId: number;
  readonly chunkIndex: number;
  readonly startFrame: number;
  readonly frames: number;
  readonly sampleRate: number;
  readonly totalFrames: number;
  readonly isFinal: boolean;
  readonly left: Float32Array;
  readonly right: Float32Array;
}

export interface OfflineWorkerClientHooks {
  readonly onProgress: (progress: number) => void;
  readonly onChunk: (chunk: OfflineChunkEvent) => void;
  readonly onError: (message: string) => void;
}

export class OfflineWorkerClient {
  private worker: Worker | null = null;
  private wasmModulePromise: Promise<WebAssembly.Module> | null = null;
  private nextJobId = 1;
  private activeJobId: number | null = null;
  private activeResolve: ((completed: boolean) => void) | null = null;
  private activeReject: ((err: Error) => void) | null = null;
  private workerHandlers: OfflineWorkerClientHooks | null = null;

  isRendering(): boolean {
    return this.activeJobId !== null;
  }

  async render(
    buildRequest: (jobId: number) => OfflineRenderRequest,
    hooks: OfflineWorkerClientHooks,
  ): Promise<boolean> {
    this.cancel();
    const jobId = this.nextJobId++;
    const worker = this.ensureWorker(hooks);
    this.activeJobId = jobId;
    this.workerHandlers = hooks;

    const wasmModule = await this.getWasmModule();
    if (this.activeJobId !== jobId) {
      return false;
    }

    const request = buildRequest(jobId);
    return new Promise<boolean>((resolve, reject) => {
      if (this.activeJobId !== jobId) {
        resolve(false);
        return;
      }
      this.activeResolve = resolve;
      this.activeReject = reject;
      const msg: OfflineWorkerInbound = { type: 'render', wasmModule, request };
      worker.postMessage(msg);
    });
  }

  cancel(): void {
    const jobId = this.activeJobId;
    if (jobId === null) return;
    this.worker?.postMessage({ type: 'cancel', jobId } satisfies OfflineWorkerInbound);
    const resolve = this.activeResolve;
    this.clearActiveJob();
    resolve?.(false);
  }

  dispose(): void {
    this.cancel();
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
  }

  private clearActiveJob(): void {
    this.activeJobId = null;
    this.activeResolve = null;
    this.activeReject = null;
  }

  private getWasmModule(): Promise<WebAssembly.Module> {
    if (!this.wasmModulePromise) {
      this.wasmModulePromise = compileWasmModule();
    }
    return this.wasmModulePromise;
  }

  private ensureWorker(hooks: OfflineWorkerClientHooks): Worker {
    this.workerHandlers = hooks;
    if (this.worker) return this.worker;
    const url = URL.createObjectURL(new Blob([offlineWorkerCode], { type: 'application/javascript' }));
    let worker: Worker;
    try {
      worker = new Worker(url, { type: 'module' });
    } finally {
      URL.revokeObjectURL(url);
    }
    worker.onmessage = (ev: MessageEvent<OfflineWorkerOutbound>) => {
      const msg = ev.data;
      if (msg.jobId !== this.activeJobId) return;
      if (msg.type === 'progress') {
        this.workerHandlers?.onProgress(msg.progress);
        return;
      }
      if (msg.type === 'cancelled') {
        const resolve = this.activeResolve;
        this.clearActiveJob();
        resolve?.(false);
        return;
      }
      if (msg.type === 'error') {
        const reject = this.activeReject;
        this.clearActiveJob();
        this.workerHandlers?.onError(msg.message);
        reject?.(new Error(msg.message));
        return;
      }
      if (msg.type === 'chunk') {
        const chunk: OfflineChunkEvent = {
          jobId: msg.jobId,
          chunkIndex: msg.chunkIndex,
          startFrame: msg.startFrame,
          frames: msg.frames,
          sampleRate: msg.sampleRate,
          totalFrames: msg.totalFrames,
          isFinal: msg.isFinal,
          left: new Float32Array(msg.left),
          right: new Float32Array(msg.right),
        };
        if (msg.isFinal) {
          const resolve = this.activeResolve;
          this.clearActiveJob();
          this.workerHandlers?.onChunk(chunk);
          resolve?.(true);
        } else {
          this.workerHandlers?.onChunk(chunk);
        }
      }
    };
    worker.onerror = (ev) => {
      const message = `Offline worker error: ${ev.message}`;
      const reject = this.activeReject;
      this.clearActiveJob();
      this.workerHandlers?.onError(message);
      reject?.(new Error(message));
    };
    this.worker = worker;
    return worker;
  }
}
