/** Worker lifecycle and job cancellation wrapper for offline-render-worker.ts. */

import type {
  OfflineRenderRequest,
  OfflineWorkerInbound,
  OfflineWorkerOutbound,
} from './offline-abi';
import offlineWorkerCode from './offline-render-worker.ts?blobworker';
import { compileWasmModule } from './wasm-loader';

export interface OfflineRenderOutput {
  readonly sampleRate: number;
  readonly totalFrames: number;
  readonly left: Float32Array;
  readonly right: Float32Array;
}

export interface OfflineWorkerClientHooks {
  readonly onProgress: (progress: number) => void;
  readonly onError: (message: string) => void;
}

export class OfflineWorkerClient {
  private worker: Worker | null = null;
  private wasmModulePromise: Promise<WebAssembly.Module> | null = null;
  private nextJobId = 1;
  private activeJobId: number | null = null;
  private activeResolve: ((result: OfflineRenderOutput | null) => void) | null = null;
  private activeReject: ((err: Error) => void) | null = null;

  isRendering(): boolean {
    return this.activeJobId !== null;
  }

  async render(
    buildRequest: (jobId: number) => OfflineRenderRequest,
    hooks: OfflineWorkerClientHooks,
  ): Promise<OfflineRenderOutput | null> {
    this.cancel();
    const jobId = this.nextJobId++;
    const worker = this.ensureWorker(hooks);
    const wasmModule = await this.getWasmModule();
    const request = buildRequest(jobId);

    return new Promise<OfflineRenderOutput | null>((resolve, reject) => {
      this.activeJobId = jobId;
      this.activeResolve = resolve;
      this.activeReject = reject;
      this.workerHandlers = hooks;
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
    resolve?.(null);
  }

  dispose(): void {
    this.cancel();
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
  }

  private workerHandlers: OfflineWorkerClientHooks | null = null;

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
        resolve?.(null);
        return;
      }
      if (msg.type === 'error') {
        const reject = this.activeReject;
        this.clearActiveJob();
        this.workerHandlers?.onError(msg.message);
        reject?.(new Error(msg.message));
        return;
      }
      if (msg.type === 'done') {
        const resolve = this.activeResolve;
        this.clearActiveJob();
        resolve?.({
          sampleRate: msg.sampleRate,
          totalFrames: msg.totalFrames,
          left: new Float32Array(msg.left),
          right: new Float32Array(msg.right),
        });
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
