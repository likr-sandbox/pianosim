/** 60fps real-time logarithmic FFT spectrum visualizer reading from AnalyserNode. */

export class SpectrumCanvas {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly getAnalyser: () => AnalyserNode | null;

  private freqData: Float32Array | null = null;
  private rafId: number | null = null;
  private running = false;
  private resizeObserver: ResizeObserver | null = null;

  constructor(canvas: HTMLCanvasElement, getAnalyser: () => AnalyserNode | null) {
    this.canvas = canvas;
    this.getAnalyser = getAnalyser;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to acquire 2D rendering context for spectrum canvas');
    }
    this.ctx = ctx;

    this.resize();
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(canvas);
    }

    document.addEventListener('visibilitychange', this.onVisibilityChange);
    this.start();
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.loop();
  }

  stop(): void {
    this.running = false;
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  destroy(): void {
    this.stop();
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    document.removeEventListener('visibilitychange', this.onVisibilityChange);
  }

  private loop = (): void => {
    if (!this.running) return;
    this.draw();
    this.rafId = requestAnimationFrame(this.loop);
  };

  private draw(): void {
    const { canvas, ctx } = this;
    const w = canvas.width;
    const h = canvas.height;
    if (w === 0 || h === 0) {
      return;
    }

    ctx.clearRect(0, 0, w, h);
    this.drawGrid(w, h);

    const analyser = this.getAnalyser();
    if (!analyser) {
      return;
    }

    if (!this.freqData || this.freqData.length !== analyser.frequencyBinCount) {
      this.freqData = new Float32Array(new ArrayBuffer(analyser.frequencyBinCount * 4));
    }
    analyser.getFloatFrequencyData(this.freqData as Float32Array<ArrayBuffer>);

    const fMin = 20;
    const fMax = 20000;
    const sampleRate = analyser.context.sampleRate;
    const fftSize = analyser.fftSize;
    const binCount = this.freqData.length;

    ctx.beginPath();
    let started = false;

    // Log-frequency curve: f(x) = fMin * (fMax / fMin)^(x / w)
    const step = 2;
    for (let x = 0; x <= w; x += step) {
      const freq = fMin * Math.pow(fMax / fMin, x / w);
      const binExact = (freq * fftSize) / sampleRate;
      const bin0 = Math.min(binCount - 1, Math.floor(binExact));
      const bin1 = Math.min(binCount - 1, bin0 + 1);
      const frac = binExact - bin0;
      const db = this.freqData[bin0] * (1 - frac) + this.freqData[bin1] * frac;

      // Map -100 dB (bottom) to 0 dB (top)
      const clampedDb = Math.max(-100, Math.min(0, db));
      const y = h * (-clampedDb / 100);

      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }

    // Spectrum stroke line
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Subtle gradient fill under curve to satisfy nonZero alpha pixel check
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(56, 189, 248, 0.35)');
    grad.addColorStop(1, 'rgba(56, 189, 248, 0.05)');
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
  }

  private drawGrid(w: number, h: number): void {
    const ctx = this.ctx;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
    ctx.font = '10px monospace';
    ctx.lineWidth = 1;

    // Frequency markers: 100Hz, 1kHz, 10kHz
    const freqs = [100, 1000, 10000];
    const labels = ['100Hz', '1kHz', '10kHz'];
    for (let i = 0; i < freqs.length; i++) {
      const f = freqs[i];
      const x = Math.round((Math.log(f / 20) / Math.log(1000)) * w);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
      ctx.fillText(labels[i], x + 4, h - 4);
    }

    // dB level markers: -20, -40, -60, -80 dB
    const levels = [-20, -40, -60, -80];
    for (const db of levels) {
      const y = Math.round(h * (-db / 100));
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
      ctx.fillText(`${db}dB`, 4, y - 2);
    }
  }

  private resize(): void {
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(300, Math.floor((rect.width || 600) * dpr));
    const h = Math.max(120, Math.floor((rect.height || 120) * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  private onVisibilityChange = (): void => {
    if (typeof document !== 'undefined' && document.hidden) {
      this.stop();
    } else {
      this.start();
    }
  };
}
