import './styles/base.css';
import './styles/keyboard.css';
import './styles/hud.css';
import './styles/panels.css';
import './styles/presets.css';
import { OfflineRenderer } from './audio/offline-renderer';
import { TelemetryClient } from './audio/telemetry-client';
import { WasmAudioHost } from './audio/wasm-host';
import { WebMidiManager } from './midi/web-midi';
import { AppShell } from './ui/app-shell';

const root = document.getElementById('app');
if (!root) {
  throw new Error('Missing #app mount container');
}

const telemetry = new TelemetryClient();
const host = new WasmAudioHost(telemetry);
const midiManager = new WebMidiManager(host);
const offlineRenderer = new OfflineRenderer(host);

const shell = new AppShell(root, host, telemetry, midiManager, offlineRenderer);

// Initialize Web MIDI subsystem
void midiManager.init();

window.addEventListener('beforeunload', () => {
  shell.destroy();
  offlineRenderer.destroy();
  midiManager.destroy();
  void host.dispose();
});
