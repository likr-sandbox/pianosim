import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "/pianosim/",
  build: {
    target: "esnext",
    outDir: "dist",
    assetsDir: "assets",
    sourcemap: true,
  },
  server: {
    port: 5173,
    open: false,
  },
  test: {
    environment: "node",
    include: ["tests/acoustic/**/*.test.ts", "tests/unit/**/*.test.ts"],
    exclude: ["tests/e2e/**", "node_modules/**", "dist/**"],
  },
});
