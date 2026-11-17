import './styles/base.css';
import './styles/keyboard.css';
import './styles/hud.css';
import './styles/panels.css';
import './styles/presets.css';
import { TelemetryClient } from './audio/telemetry-client';
import { WasmAudioHost } from './audio/wasm-host';
import { MidiPlayer } from './midi/midi-player';
import { WebMidiManager } from './midi/web-midi';
import { AppShell } from './ui/app-shell';

const root = document.getElementById('app');
if (!root) {
  throw new Error('Missing #app mount container');
}

const telemetry = new TelemetryClient();
const host = new WasmAudioHost(telemetry);
const midiManager = new WebMidiManager(host);
const midiPlayer = new MidiPlayer(host);

const shell = new AppShell(root, host, telemetry, midiManager, midiPlayer);

// Initialize Web MIDI subsystem
void midiManager.init();

window.addEventListener('beforeunload', () => {
  shell.destroy();
  midiPlayer.destroy();
  midiManager.destroy();
  void host.dispose();
});
