/** Static HTML skeleton for PianoSim Web Studio shell. */

export const STUDIO_SHELL_HTML = `
  <main class="studio-shell">
    <header class="studio-header">
      <div class="header-titles">
        <p class="eyebrow">Milestone 7 &middot; Pure-Rust Wasm SIMD128 Physical Engine</p>
        <h1 class="studio-title">PianoSim Web Studio</h1>
      </div>
      <div class="header-actions">
        <button id="btn-unlock" data-testid="unlock-audio" type="button" class="btn btn-primary">
          Unlock Audio
        </button>
        <button id="btn-a4-trigger" data-testid="trigger-a4" type="button" class="btn btn-accent val-mono">
          A4 = 440.0 Hz
        </button>
        <button id="btn-panic" data-testid="panic-btn" type="button" class="btn btn-ghost">
          All Notes Off
        </button>
      </div>
    </header>

    <p id="error-banner" data-testid="error-banner" class="error-banner" hidden></p>

    <!-- Real-time Diagnostics HUD & Spectrum Canvas Slot -->
    <section class="viz-section" aria-label="Audio Visualizer and Telemetry HUD">
      <div id="hud-mount" class="hud-slot"></div>
      <div class="canvas-card">
        <canvas
          id="spectrum-canvas"
          data-testid="spectrum-canvas"
          class="spectrum-canvas"
          width="600"
          height="120"
        ></canvas>
      </div>
    </section>

    <!-- Velocity & Performance Control Bar -->
    <section class="performance-controls" aria-label="Input Velocity and Pedals">
      <label class="velocity-control" for="velocity-slider">
        <span class="control-label">Strike Velocity</span>
        <input
          id="velocity-slider"
          data-testid="velocity-slider"
          type="range"
          min="1"
          max="127"
          value="100"
          class="velocity-slider"
        />
        <span id="velocity-value" class="val-mono">100</span>
      </label>
      <div id="touch-curve-mount" class="touch-curve-slot"></div>
      <div id="pedal-mount" class="pedal-slot"></div>
    </section>

    <!-- Responsive 88-Key Piano Keyboard -->
    <section class="keyboard-stage" aria-label="88-Key Concert Grand Piano Keyboard">
      <div id="keyboard-mount" class="keyboard-mount"></div>
    </section>

    <!-- MIDI Transport & Scoring -->
    <div id="transport-mount" class="transport-slot"></div>

    <!-- Acoustic Parameter Presets Bar -->
    <div id="preset-mount" class="preset-slot"></div>

    <!-- Real-Time Performance Profile Panel -->
    <div id="perf-mount" class="perf-slot"></div>

    <!-- Voicing & Space Panels -->
    <div class="studio-panels-row">
      <div id="voicing-mount" class="panel-slot"></div>
      <div id="space-mount" class="panel-slot"></div>
    </div>
  </main>
`;
