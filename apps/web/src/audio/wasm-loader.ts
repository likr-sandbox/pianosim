const WASM_MAGIC = [0x00, 0x61, 0x73, 0x6d] as const;

export function assertValidWasmBytes(bytes: ArrayBuffer): void {
  if (bytes.byteLength < 8) {
    throw new Error('Fetched piano_wasm.wasm is empty or truncated.');
  }
  const header = new Uint8Array(bytes, 0, 4);
  for (let i = 0; i < 4; i++) {
    if (header[i] !== WASM_MAGIC[i]) {
      throw new Error(
        'Invalid Wasm binary magic header (expected compiled public/piano_wasm.wasm).',
      );
    }
  }
}

export async function compileWasmModule(): Promise<WebAssembly.Module> {
  const url = `${import.meta.env.BASE_URL}piano_wasm.wasm`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
  }
  if (typeof WebAssembly.compileStreaming === 'function') {
    try {
      return await WebAssembly.compileStreaming(response.clone());
    } catch {
      // Fallback when static dev server serves non-application/wasm MIME header
    }
  }
  const bytes = await response.arrayBuffer();
  assertValidWasmBytes(bytes);
  return WebAssembly.compile(bytes);
}
