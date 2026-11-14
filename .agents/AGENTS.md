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
- **RustツールチェーンのPATH**: `rustc` (`1.97.1`), `cargo`, `wasm-pack` (`0.15.0`), `wasm-bindgen`, `rustup`（`wasm32-unknown-unknown` 導入済み）は `$HOME/.cargo/bin` にあり、非ログインシェルのデフォルト `PATH` には含まれない。コマンド実行時は必ず `export PATH="$HOME/.cargo/bin:$PATH"` を先頭に付けること。
- **ポート `127.0.0.1:5173` のブラックホール問題**: 本実行環境では `127.0.0.1:5173` 宛てのTCP SYNが応答せずハングする。ViteサーバおよびPlaywright `webServer` 設定は `PIANOSIM_PORT`（例: `5188`）と `--strictPort` で5173以外のポートへ逃がせるようにすること。Headless Chromium起動には `--autoplay-policy=no-user-gesture-required` と `--no-sandbox` を付与する。
- **GitHub CLIとPages公開**: 環境に `gh` コマンドは未インストール。リモートは `origin = git@github.com:likr-sandbox/pianosim.git`、GitHub Pages（`https://likr-sandbox.github.io/pianosim/`）は `Settings -> Pages -> Source: GitHub Actions` が有効化済み。Viteの `base` は `/pianosim/` とし、デプロイ確認は `curl` と公開URLへのHeadless Playwrightで行う。

## 5. 音響DSP・検証上の知見（蓄積ログ）
- **旧v1実装の保全**: M1で旧TS製DWG実装（`c75cf57`）を一掃する直前に、Gitタグ `v0.1.0-ts-dwg` を付与して比較用に残すこと。
- **ステレオ出力バッファのエイリアス防止**: テストやWasm/TS境界でモノラルミックスダウンを取得する際、L/Rに同一配列を渡すと信号加算や破壊が起きる。Rustの借用規則に加え、TS側・テストヘルパ側でも必ず別個の `Float32Array` 領域を確保する。
- **高音域（C7〜C8）の接触時間・減衰落とし穴**: 高音弦の1周期（C8 ≈ `0.24 ms`）に対しハンマー接触パルスが長すぎると基本波がスペクトルヌルに落ち、1次LPF型の単純なループ減衰は高音弦の基本波を数十msで消滅させる。Stage 1の局所384 kHz接触計算とStage 2のZOHレゾネータでは、A0〜C4だけでなく **C6〜C8の攻撃RMS・スペクトル・T60減衰** を必ず自動検証対象に含める。
- **E2Eテストの誠実性**: Playwright E2Eで合成モック音源を使うとリグレッションを見逃す。必ず実Wasm `AudioWorklet` パイプラインの出力（`AudioContext` 終端タップまたはWorkletテレメトリ）を検証し、空のUIスキップガード（`test.skip`）を入れない。

## 6. 本ファイルの更新履歴（直近マイルストーン要約）
- **Pre-M1 (2026-11-14)**: 初版作成。v1実装・`/grill-me` 設計決定（`ARCHITECTURE.md`, `ROADMAP.md`, `DECISIONS_AND_FALLBACKS.md`, `SWARM_WORKFLOW.md`）および環境・DSP知見を登録。
