# pianosim プロジェクト知識ベース & 開発規約 (`AGENTS.md`)

本ファイルは全エージェントが共有するプロジェクトの正規ルールおよび実践知（Living Knowledge Base）である。常に最新の状態に保ち、陳腐化した項目は書き換えて **500行以下・24,000バイト未満** を維持すること。

## 1. 必読ドキュメント
- システム設計・4ステージ物理モデル・FTO規約: `docs/ARCHITECTURE.md`
- 段階的マイルストーン（M1〜M8）・10クレート分割マップ: `docs/ROADMAP.md`
- 非採用要件（D1〜D8）と設計誤り時のフォールバック: `docs/DECISIONS_AND_FALLBACKS.md`
- Swarmマルチエージェント開発・`share` worktree・Git運用: `docs/SWARM_WORKFLOW.md`

## 2. 絶対遵守の実装・設計不変条件
1. **1ファイル500行上限**: すべてのソースコード（`.rs`, `.ts`, `.js`, `.css`）および設計文書は **500行以下**（目標120〜350行）とする。肥大化の兆候が見えた段階で責務ごとにモジュール分割する（CIの `piano-cli check-lines` で強制）。
2. **リアルタイムDSP制約**: `Engine::init(sample_rate)` 以降のオーディオスレッド処理（`render_block` / `AudioWorklet::process`）内では**ヒープ確保ゼロ・ロック取得ゼロ・システムコールゼロ・パニック禁止**を厳守する。
3. **Web実行アーキテクチャ**: 単一スレッド `AudioWorkletGlobalScope` + `wasm32-unknown-unknown`（`+simd128` / `f32x4`）で実行し、静的ホスト互換性（`COOP/COEP` 不要）を維持する。品質ティアや休止弦スリープ間引きは入れず、全88鍵・230本・約2.6万モードを毎サンプル常時SIMD駆動する（性能限界検知時は `docs/DECISIONS_AND_FALLBACKS.md` §D1 を参照）。
4. **純Rust自己完結**: 弦・ハンマー定数および構造音響ボディモード／IRはすべて `piano-body-gen` 等のRustオフライン物理シミュレーションで生成する（Python/JAX・録音波形は使用しない）。
5. **知財クリーンルーム（FTO）7原則**:
   - `OpenPiano` 等の **AGPL-3.0 コード参照・流用禁止**（パーミッシブライセンス依存のみ）。
   - 解析的分散式 $f_j = jF_0\sqrt{1+Bj^2}$ を直接評価（RBF/SVM/NN/Taylor補間や事前畳み込み励振テーブル禁止）。
   - ダンパーは幾何学的クリアランス入力の片側ヒステリシス接触要素（粘性係数テーブル動的書換禁止）。
   - 打弦弦・開放弦の単一縮約ブリッジ双方向連成（並列エフェクト化や倍音関係 `if-then` ダンプ判定禁止）。
   - 8ゾーン物理位置射影（グループ集約＋後段微小遅延禁止）および棚板衝撃の第9構造体ゾーン直接入力（2系統個別減衰制御禁止）。

## 3. Swarm・`workspace:"share"` Worktree・Git運用ルール
- **所有権分離**: 並列稼働する `author` ワーカーは全員 `workspace:"share"` で起動し、同一ウェーブ内で同一クレート／同一ファイルを複数人で触らない。
- **ブランチ運用**: `author` は `share` worktree 内で必ず `git checkout -B m<N>/<worker>-<topic> main` を切って作業する。次ウェーブの作業開始時も最新 `main` から切り直す。
- **コミット粒度**: 論理単位（1モジュール＋テスト等）ごとに Conventional Commits（`feat(crate): ...`, `fix(...): ...`）でコミットし、全コミットでビルド成功＆500行上限を維持する。
- **マイルストーン単位 `push`**: **マイルストーン途中の `git push` は厳禁**。`verifier`（`workspace:"inherit"`、編集禁止）による全検収ゲート合格後に `integrator` が `main` へ履歴を保って統合し、`.agents/AGENTS.md` を更新コミットした上で、**マイルストーン完了ごとに1回のみ**タグ打刻＋`git push origin main --tags` と公開URL確認を行う。
- **`kill` 禁止条件**: 未コミット変更が残っている間、および `main` 統合・統合後テスト完了前の `author` への `swarm{action:"kill"}` は厳禁（worktree消失防止）。
- **知識更新（競合防止）**: 並列 `author` は `.agents/AGENTS.md` を直接編集せず、ボード報告の `Learnings:` 欄に記載する。`integrator` のみが `main` 上で本ファイルを最新化・剪定する。

## 4. 環境・ツールチェーン固有の知見（蓄積ログ）
- **Rustツールチェーン & CI固定**: ローカル `rustc` (`1.97.1`), `cargo`, `wasm32-unknown-unknown` は `$HOME/.cargo/bin` にあり、実行時は必ず `export PATH="$HOME/.cargo/bin:$PATH"` を先頭に付ける。GitHub Actions（`.github/workflows/{ci,deploy}.yml`）は `@stable` のドリフトを防ぐため `dtolnay/rust-toolchain@master` + `toolchain: "1.97.1"` で固定する（`@1.97.1` タグは不可、ローカル `rust-toolchain.toml` は不使用）。
- **`rustfmt` 1.97.1 と行数検査**: 10段超の深いネストの右結合Horner多項式式は `rustfmt` がCPU 100%でハングするため `const` 配列ループ（`eval_horner` 等）で評価する。また `cargo fmt` の自動折り返しで行数が増えるため、`piano-cli check-lines` は必ず `cargo fmt` 後に実行する。
- **ポート `127.0.0.1:5173` 回避 & `share` Worktree Web準備**: 未バインドLoopback TCP SYNが約130秒ハングするため、`apps/web/playwright.config.ts` では `PIANOSIM_PORT`（例: `5188`, `--strictPort`）と `webServer.wait: { stdout: /Local:\\s+http/ }` を併用する（E2E約4.4s完走、Chromiumは `--autoplay-policy=no-user-gesture-required --no-sandbox`）。`workspace:share` worktree には `node_modules/` が無いため、Webビルド/E2E前に `npm ci --prefer-offline` を実行する。
- **`AudioWorkletGlobalScope` + Wasm 境界**: インポート0個の素の `wasm32-unknown-unknown` C-ABI `.wasm`（`WebAssembly.instantiate(bytes, {})`）と Vite `transformWithEsbuild`（`?worklet` Blob URL）で、`window` / `TextDecoder` / `/@vite/env` クラッシュを回避し `dev`・`preview`・`/pianosim/` で同一動作させる。
- **CI静的・バイナリ検査ゲート**: `.github/workflows/ci.yml` で `cargo fmt --check`、`piano-cli check-lines`、`cargo clippy --workspace -- -D warnings`、`cargo clippy --package piano-wasm --target wasm32-unknown-unknown -- -D warnings`、`cargo test --workspace`、`npm run build`、`piano-cli check-wasm-simd apps/web/public/piano_wasm.wasm`、`npm run test:e2e` を必須実行する。
- **GitHub CLIとPages公開**: `gh` コマンドは未インストール。リモート `origin = git@github.com:likr-sandbox/pianosim.git`、GitHub Pages（`https://likr-sandbox.github.io/pianosim/`、`base: /pianosim/`）は GitHub Actions 公開が有効。確認は `curl`・GitHub API・公開URLへのHeadless Playwrightで行う。

## 5. 音響DSP・検証上の知見（蓄積ログ）
- **旧v1実装の保全**: 旧TS製DWG実装（`c75cf57`）はGitタグ `v0.1.0-ts-dwg` にアーカイブ済み（比較時は `git worktree add /tmp/pianosim-v1 v0.1.0-ts-dwg`）。
- **ゼロアロケーションと `no_std` コア**: `[T]::sort_by` はヒープ確保の可能性があるため固定長 `[SampleEvent; 512]` キューはインプレース挿入ソートを使う。`piano-simd` / `piano-spec` / `piano-strings` は `#![cfg_attr(not(test), no_std)]`（`alloc` / 純Rust `f64` 初期化数学）とし、外部クレート依存ゼロを維持する。
- **Exact Coupled-Form Incremental ZOH**: `f32` Direct-form `2*r*cos(theta)` は低音（A0 @ 96 kHz で ~15.8 cents誤差）で `2.0` 近傍のULP劣化により破綻する。`d = 1 - r*cos(Omega*Ts)`, `s = r*sin(Omega*Ts)` を `f64` で算出して `(u, v)` 回転（`du = d*u + s*v`, `dv = s*u - d*v`）で更新すると 44.1/48/96 kHz 全域で `<= 0.00257 cents` を達成できる。ダンパー補間も `damp = 1 - hold_01` を用いて `(d_hold, s_hold)` 起点でアフィン補間し、`hold=1` 時の24bit仮数部を保存する。
- **8サンプルレジスタタイリング & D1実測ベースライン**: 全約25.5k〜26.2kモードを毎サンプル無条件駆動したまま `TILE=8` レジスタ常駐化により SoA メモリ帯域を8分の1に削減し、ブロック末尾でのみ `< 1e-20` 非正規化数フラッシュを行う（`piano-simd::F32x4` に `Neg` 実装なし）。M2実測: ネイティブ `128 @ 48 kHz` `p50=1002.5 us / p99=1385.9 us (52.0%)`、Chromium Wasm SIMD128 `128 @ 48 kHz` バッチ平均 `1078.0 us (40.4%) / p99=1168.8 us`（0アンダーラン、D1閾値 70% = `1866.7 us` に余裕）。
- **音域・高域・分散キャリブレーション**: 音域依存打弦ロールオフ（`slope = 0.12 + 1.80*x`）により A4 基音が第2倍音を +5.9 dB 上回る。C8 第1部分音は剛性分散により `partial_frequency_hz(key_tuned_f0_hz, B, 1)`（~4358 Hz）へシフトするため、高音ピッチ検証は分散適用後の理論部分音と比較する。C6〜C8は鍵長連動パルス（`~0.20*T0`）と攻撃RMS・T60を自動検証する。
- **ステレオ・Wasm SIMD・E2E検査の誠実性**: L/R 出力バッファは絶対にエイリアスさせない。`check-wasm-simd` は Code section（ID 10）の命令をデコードして実 `0xfd` オペコードを数える。Playwright E2E は合成モック・`test.skip` 禁止で実Wasm `AudioWorklet` + 受動 `AnalyserNode` を検証する。

## 6. 本ファイルの更新履歴（直近マイルストーン要約）
- **M2 (2026-11-15)**: 88鍵・230弦物理スケーリング／Railsback調律／D4 Voicing枠（`piano-spec`）、約25.5k〜26.2kモード連続SIMD ZOH弦バンク（`piano-strings`）、`PianoEngine` 88鍵統合、`piano-cli bench` を実装。Coupled-form ZOH高精度化、`TILE=8` タイリング、M2ネイティブ/Chromium Wasm D1実測、打弦ロールオフ、CI Rust `1.97.1` 固定知見を統合。
- **M1 (2026-11-15)**: `v0.1.0-ts-dwg` (`c75cf57`) へ旧v1保全後に10クレート純Rustモノレポ＋`piano-simd`＋C-ABI `piano-wasm`＋`piano-cli`＋Vanilla TS `AudioWorklet`＋CI/Pagesを確立。
- **Pre-M1 (2026-11-14)**: 初版作成（設計仕様 `ARCHITECTURE.md` / `ROADMAP.md` / `DECISIONS_AND_FALLBACKS.md` / `SWARM_WORKFLOW.md`）。
