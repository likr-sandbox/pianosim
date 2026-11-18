# pianosim プロジェクト知識ベース & 開発規約 (`AGENTS.md`)

本ファイルは全エージェントが共有するプロジェクトの正規ルールおよび実践知（Living Knowledge Base）である。常に最新化し **500行以下・24,000バイト未満** を維持すること。

## 1. 必読ドキュメント
- システム設計・4ステージ物理モデル・FTO規約: `docs/ARCHITECTURE.md`
- 段階的マイルストーン（M1〜M8）・10クレート分割マップ: `docs/ROADMAP.md`
- 非採用要件（D1〜D8）と設計誤り時のフォールバック: `docs/DECISIONS_AND_FALLBACKS.md`
- マルチエージェント運用・Tier 別排他検証・スイープ/Git規約: `docs/SWARM_WORKFLOW.md`

## 2. 絶対遵守の実装・設計不変条件
1. **1ファイル500行上限**: すべてのソースコード（`.rs`, `.ts`, `.js`, `.css`）および設計文書は **500行以下**（目標120〜350行）とする（CI の `piano-cli check-lines` で強制）。
2. **リアルタイムDSP制約**: `Engine::init(sample_rate)` 以降のオーディオスレッド処理（`render_block` / `AudioWorklet::process`）内では**ヒープ確保ゼロ・ロック取得ゼロ・システムコールゼロ・パニック禁止**を厳守する。
3. **Web実行アーキテクチャ & Stage 2駆動方針**: `COOP/COEP` 不要で静的ホスト互換を維持しつつ、オーディオ演算は単一スレッド `AudioWorkletGlobalScope` + `wasm32-unknown-unknown`（標準 `+simd128` 単一バイナリ、`+relaxed-simd` / `fmaf` 不使用・`0 imports`）、重い事前計算（`set_space` IR/FFT・調律変更）は専用 **Prep Web Worker + `Transferable ArrayBuffer`** へ分離する（`hammer_hardness` 単体は Voicing Dirty Mask で即時反映）。Web `AudioContext` は **`48 kHz` 固定**（`96 kHz` はオフラインCLI検証専用）。全88鍵・230本・約2.6万モードを維持し、ビット一致ゼロスキップ・物理エネルギー地平線縮退（`-115〜-120 dB`）・サステイン減衰済み鍵の受動共鳴降格・融合SAV（`-55 dB` 平滑休眠）を適用する。
4. **純Rust自己完結**: 弦・ハンマー定数および構造音響ボディモード／IRはすべて `piano-body-gen` 等のRustオフライン物理シミュレーションで生成する（Python/JAX・録音波形不使用）。
5. **知財クリーンルーム（FTO）7原則**:
   - `OpenPiano` 等の **AGPL-3.0 コード参照・流用禁止**（パーミッシブライセンス依存のみ）。
   - 解析的分散式 $f_j = jF_0\sqrt{1+Bj^2}$ を直接評価（RBF/SVM/NN/Taylor補間・事前畳み込み励振テーブル禁止）。
   - ダンパーは幾何学的クリアランス入力の片側ヒステリシス接触要素（粘性係数テーブル動的書換禁止）。
   - 打弦弦・開放弦の単一縮約ブリッジ双方向連成（並列エフェクト化・倍音関係 `if-then` ダンプ判定禁止）。
   - 8ゾーン物理位置射影（グループ集約＋後段微小遅延禁止）および棚板衝撃の第9構造体ゾーン直接入力（2系統個別減衰制御禁止）。

## 3. Swarm・Worktree・Tier 別検証・Git 運用ルール
- **着手前 `git fetch` 必須**: 各探索・Wave 開始時は `git fetch origin && git rev-list --left-right --count HEAD...origin/main` で先行コミットを確認する。
- **所有権・ブランチ**: `author` は `workspace:"share"` で起動し、`git checkout -B m<N>/<worker>-<topic> main` で作業する（同ウェーブで同クレート／同ファイル非共有）。未コミット残存・`main` 未統合状態での `kill` は厳禁。
- **Tier 0〜3 差分別ゲート & Step A/B 排他検証**: 5人同時 `inherit` による Cargo ロック・CPU 競合を避け、Step A（`Reviewer`＋差分/FTO専任 `Auditor`）と Step B（無競合時 `Challenger` 単独ベンチ）に分離する。`docs/**`（Tier 0）や `apps/web/e2e/**`（Tier 1）では Rust 全テスト・`bench` を再実行せず、同一ソースツリーの再検証も省略する。詳細は `docs/SWARM_WORKFLOW.md` 参照。
- **スイープ & タイムアウト**: 3回以上の定数調整は手動編集ループを禁じ1回のプログラムでレンジ掃引する。ワンオフ Node/ブラウザ検証は `timeout -k 5s 180s` ＋ `finally` クリーンアップを必須とする。
- **マイルストーン単位 `push`・Single-Writer**: 途中 `push` 禁止。`.agents/AGENTS.md` 更新は `integrator` のみが行い、タグ打刻＋`push` 後は GitHub Actions `CI` 成功と Pages HTTP 200 の両方を必須確認する（イベント駆動報告・20 KiB 以内）。

## 4. 環境・ツールチェーン・高速ビルド/E2E 知見
- **Rustツールチェーン & `Cargo.toml` プロファイル**: ローカル `rustc` (`1.97.1`), `cargo`, `wasm32-unknown-unknown` は `$HOME/.cargo/bin`（コマンド先頭に `export PATH="$HOME/.cargo/bin:$PATH"`）。CI（`.github/workflows/{ci,deploy}.yml`）は `dtolnay/rust-toolchain@master` (`toolchain: "1.97.1"`)＋`Swatinem/rust-cache@v2` を使用。`Cargo.toml` の `[profile.dev]` を `opt-level = 2, debug = 1` に統一（`test` が自動継承）し、`piano-cli` と `cargo test` の二重コンパイルを避けつつ `cargo test --workspace` を **12.7分 → ~23〜31秒** へ短縮した（`debug_assertions` 有効維持、`challenge4` 重量統合ゲートのみ `#[cfg(not(debug_assertions))]` で `scenarios` へ分離）。
- **`.github/workflows/deploy.yml` CI 連動**: `deploy.yml` は `workflow_run`（`workflows: ["CI"], completed`, `main` push 成功時のみ）で発火し、赤 CI での誤公開を防ぐ。
- **`rustfmt` 1.97.1・`clippy`・`check-lines`**: 10段超 Horner 式は `const` 配列ループ使用、`clippy::neg_cmp_op_on_partial_ord` 対策で `x <= 0.0 || x.is_nan()` を使用、`check-lines` は必ず `cargo fmt` 後に実行。`check_lines.rs` は `apps/web/e2e/` 内の `waitForTimeout`（壁時計 sleep）も静的エラーとして禁止する。
- **Playwright E2E（2プロジェクト並列・`vite preview`・決定論的同期）**:
  - Loopback IPv4 SYN ハング回避のため `apps/web/playwright.config.ts` は `PIANOSIM_PORT` (`5188`) ＋ `webServer.wait: { stdout: /Local:\s+http/ }` を維持。`functional`（並列 `workers: 3`, CI `2`）と `perf-stability`（`dependencies: ['functional']`, `workers: 1`）に分離し、`npm run build:web && npm run preview`（CI 時は既存ビルド再利用）で配信して **71/71 E2E を `4.0分 → ~2.3〜2.7分`** に短縮。
  - Prep Worker / `BlobStreamer` 適用完了を `abi.ts` / `telemetry-client.ts` の `requestedSpaceSeq`/`appliedSpaceSeq`/`requestedVoicingSeq`/`appliedVoicingSeq` として `window.__PIANO_DIAG__` に公開し、`audio-diag.ts` の `waitForPrepIdle` / `waitAudioBlocks` / `waitForAudioRms` / `waitForSilence` で全105箇所の `waitForTimeout` を撤廃。
- **`AudioWorkletGlobalScope` + Prep Worker + Wasm 境界**: `0 imports` 標準 `+simd128` 単一 `.wasm`（`+relaxed-simd` / `extended-const` / `fmaf` 禁止）＋ Prep Worker + `Transferable ArrayBuffer`。`vite.config.ts` の `?worklet`/`?blobworker` は `esbuild.build` で自己完結バンドル生成。Web `AudioContext` は `48 kHz` 固定。
- **決定論的テストとベンチマーク分離**: 時間制約は `piano-cli bench` / `npm run bench:web` へ分離（直前 `sleep 3`、`taskset -c 0` 不使用）。GitHub CLI 未導入（Pages: `https://likr-sandbox.github.io/pianosim/`、`base: /pianosim/`）。

## 5. 音響DSP・検証上の知見（蓄積ログ）
- **クロス CPU `.pbdy` パリティ (`multi_body_parity.rs`)**: `glibc` `libm` IFUNC 差分で8モードの固有ベクトル符号が反転し得るため、生バイト比較ではなく符号不変・物理量不変（±1結合反転・`keybed` 非反転・`verify_pole_cancellation <= 1e-4 dB`）で検証する（`GLIBC_TUNABLES="glibc.cpu.hwcaps=-FMA,-AVX2,-AVX512F"` 下 `piano-body-gen` `14/14` PASS）。
- **Wave 4 (`II-6, IV-1..IV-3`)**: 低音 `k=0..16` `3×6` ファントム（`12×24`, `wasm32` `inline(never)`）、A4 `392.0..466.2 Hz`・古典4音律・`SplitMix64` `Condition`（既定値ビット一致）、デモ6曲ヒューマナイズ・Y座標ベロシティ・24b TPDF/32f WAV（`WorkletRecorderPool`）、joint LS `f0_estimator.rs`、Stage 4 `challenge3` 統一、`challenge4_wave4_gates.rs`（ISO-226/IACC/`ff` IMD）。
- **Wave 3 (`III-1..III-4`)**: `HallFdn`（`hall_fdn.rs` `490`行, `<400Hz` バイノーラル IACC シェルビング, 既定 `decay=1.6, mix=0.18`）、3種 `.pbdy`（`concert_grand` `1,372` / `upright_studio` `1,305` / `vintage_saloon` `1,342` モード, `cancel_err=0.0000 dB`）、`15×20` Ritz（sum-factorization + Householder QL）+ `>3.5kHz` ベルベットテール、0遅延 $C^2$ `StereoLimiter`（`limiter.rs`）。
- **Wave 2 (`II-1..II-5`)**: 有限ハンマー $\Phi_w(j)$ + $\beta(k)$ + Hermite `AdaaPowerLut` + 低音 Zone `0..1` 回復、高音 `6 kHz` WDF 再接続 + `C5+` デチューンテーパ、`damping_law.rs` 3機構 $T_{60}(k, j)$、調律連動 `DuplexBank` (`+20 dB`) + `<1e-20` フラッシュ、幾何ダンパー剥離（`prefix` のみ, `MAX_PEEL_SLOTS=24`）・離鍵 Zone 8・FIR `prev_sub_forces` 保持。`contact_tests.rs` `498`行・`presets.rs`/`hall_fdn.rs` `490`行のため追加時要分割。
- **Wave 1 (`I-1..I-5`) & M1〜M8 基盤**: `BlobStreamer` 共有 `staging` 直列化＋`checked_add` ガード、`u7ToU16`/`cc7ToU16`＋`gamma=2.15`＋`Touch Curve`、`EventQueue` carryover＋`MidiPlayer` 60ms先読み、`WebMidiManager`＋連続 CC67＋Una Corda一本化、共有24点 $p$-グリッド `PGridLuts` (`384 KB`)。Stage 1〜4 基盤（`no_std` 5クレート、Exact ZOH `<=0.00257 cents`、460 V/H 片側ダンパー、Drift-regulated SAV + `8×` 局所アクション、8ゾーン+NUPOLS 0遅延 `<= -130.28 dB`、Prep Worker 8ms二重バッファ）。

## 6. 直近マイルストーン履歴
- **Workflow Opt (2026-11-18)**: `[profile.dev] opt-level = 2, debug = 1`・`rust-cache@v2`・`deploy.yml` CI 連動・E2E `waitForTimeout` 105箇所全廃 & `check_lines.rs` 静的禁止・`waitForPrepIdle` 導入・`functional`/`perf-stability` 2レーン化（`cargo test` `12.7分 → ~23〜31s`, E2E `71/71` `4.0分 → ~2.3〜2.7m`）、`SWARM_WORKFLOW.md` Tier 0〜3・Step A/B・`git fetch`・スイープ/`timeout` 規約化。
- **Wave 1〜4 + CI Parity Fix (`wave1..wave4`, `6c76de8`, 2026-11-18)**: 全18品質施策およびクロス CPU `.pbdy` パリティ修正を完了・公開（`272/272` Rust, `71/71` E2E, `27,807` `0xfd`, `19/19` scenarios, Pages `HTTP/2 200`）。
- **M1〜M8 + Demo/Presets (2026-11-14..17)**: `v0.1.0-ts-dwg` アーカイブ後、10クレート純Rustエンジン・4ステージ物理モデル・Webデモ6曲/プリセットを構築。
