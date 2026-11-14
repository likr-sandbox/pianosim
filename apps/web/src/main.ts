import './styles/base.css';
import { TelemetryClient } from './audio/telemetry-client';
import { WasmAudioHost } from './audio/wasm-host';
import { AppShell } from './ui/app-shell';

const root = document.getElementById('app');
if (!root) {
  throw new Error('Missing #app mount container');
}

const telemetry = new TelemetryClient();
const host = new WasmAudioHost(telemetry);
const shell = new AppShell(root, host, telemetry);

window.addEventListener('beforeunload', () => {
  shell.destroy();
  void host.dispose();
});
