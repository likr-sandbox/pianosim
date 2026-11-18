---
name: workflow-retrospective
description: >-
  Audits slow multi-agent sessions, swarms, and test/CI pipelines using transcript
  logs and repository timing profiles to produce prioritized P0-P2 speedups.
  Use when asked to investigate task duration, /teamwork or swarm bottlenecks,
  CI/E2E slowness, or redundant verification overhead.
---

# Workflow Retrospective & Bottleneck Audit

Run this read-only diagnostic before proposing or implementing build, test, or multi-agent workflow optimizations in `pianosim`.

## 1. Map the Execution Timeline (`transcript.jsonl`)
Inspect the parent and subagent `transcript.jsonl` files under `~/.gemini/antigravity-cli/brain/` to measure critical-path wall time, subagent overlap, long step gaps, and truncation:

```bash
python3 - <<'PY'
import json, pathlib, datetime as dt

base = pathlib.Path.home() / ".gemini/antigravity-cli/brain"
for p in sorted(base.glob("*/.system_generated/logs/transcript.jsonl")):
    lines = p.read_text(errors="replace").splitlines()
    if not lines:
        continue
    rows = [json.loads(l) for l in lines if l.strip()]
    t0 = dt.datetime.fromisoformat(rows[0]["created_at"].replace("Z", "+00:00"))
    t1 = dt.datetime.fromisoformat(rows[-1]["created_at"].replace("Z", "+00:00"))
    gaps = []
    for a, b in zip(rows, rows[1:]):
        da = dt.datetime.fromisoformat(a["created_at"].replace("Z", "+00:00"))
        db = dt.datetime.fromisoformat(b["created_at"].replace("Z", "+00:00"))
        sec = (db - da).total_seconds()
        if sec >= 120:
            gaps.append((round(sec), b.get("step_index"), b.get("type")))
    subs = [
        tc.get("arguments", {}).get("Role")
        for r in rows
        for tc in (r.get("tool_calls") or [])
        if tc.get("name") == "invoke_subagent"
    ]
    trunc = sum(1 for r in rows if r.get("truncated_fields"))
    print(f"{p.parent.parent.parent.name}: steps={len(rows)} dur={(t1-t0)} subs={len(subs)} trunc={trunc} top_gaps={sorted(gaps, reverse=True)[:5]}")
PY
```

Identify:
- **Rework**: stale base branches vs `origin/main`, manual `edit -> test` loops over >3 coupled parameters instead of a single sweep, or post-push CI parity failures.
- **Lock / CPU Contention**: overlapping `workspace:"inherit"` tasks or concurrent `cargo` / Playwright runs fighting over `target/` or shared host CPU (`uptime`, `ps`).
- **Truncation / Context Bloat**: oversized subagent reports (`> 20 KiB`) causing truncation or compaction overhead.

## 2. Audit Build, Test & Duplicate Verification Cost
- **Rust Profiles**: Verify `[profile.dev]` and `[profile.test]` in `Cargo.toml` remain aligned (`opt-level = 2, debug = 1`) so `piano-cli` and `cargo test --workspace` share a single incremental graph.
- **CI Caching & Gating**: Check `.github/workflows/{ci,deploy}.yml` for `Swatinem/rust-cache@v2`, PR `concurrency`, and `deploy.yml` `workflow_run` gating on `CI` success.
- **Tiered Verification (`docs/SWARM_WORKFLOW.md`)**: Count redundant full-gate executions over identical git trees or Tier 0 (`docs/**`, `.agents/**`) / Tier 1 (`apps/web/e2e/**`) diffs.

## 3. Audit E2E Synchronization & Execution Lanes
- Run `export PATH="$HOME/.cargo/bin:$PATH" && cargo run -p piano-cli -- check-lines` to verify zero `waitForTimeout` calls in `apps/web/e2e/` and `<= 500` line limits.
- Check `apps/web/playwright.config.ts` to ensure `functional` runs in parallel over `vite preview` while `perf-stability` runs sequentially (`workers: 1`), and initialization `expect.poll` timeouts match `unlockAudioEngine` (`15_000 ms`).

## 4. Produce a Prioritized P0–P2 Matrix
Summarize verified findings (citing only directly observed timings and counts) as a P0 / P1 / P2 table ranked by critical-path minutes saved vs implementation risk.
