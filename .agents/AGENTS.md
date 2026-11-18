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
3. **Web実行アーキテクチャ & Stage 2駆動方針**: `COOP/COEP` 不要で静的ホスト互換を維持しつつ、オーディオ演算は単一スレッド `AudioWorkletGlobalScope` + `wasm32-unknown-unknown`（標準 `+simd128` 単一バイナリ、`+relaxed-simd` / `fmaf` 不使用・`0 imports`）、重い事前計算（`set_space` IR/FFT・調律変更）は専用 **Prep Web Worker + `Transferable ArrayBuffer`** へ分離する（`hammer_hardness` 単体は Voicing Dirty Mask で即時反映）。Web `AudioContext` は **`48 kHz` 固定**（`96 kHz` はオフラインCLI検証専用）。品質ティア分割は入れず全88鍵・230本・約2.6万モードの物理定義を維持しつつ、ビット完全一致ゼロスキップ、物理エネルギー地平線縮退（`-115〜-120 dB`、FTO #4/#7準拠）、サステイン踏下中の減衰済み鍵の受動共鳴降格、および融合SAV（`-55 dB` 平滑休眠）を適用する。
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
- **`rustfmt` 1.97.1・`clippy` と行数検査**: 10段超の深いネストの右結合Horner多項式式は `rustfmt` がCPU 100%でハングするため `const` 配列ループ（`eval_horner` 等）で評価する。`clippy::neg_cmp_op_on_partial_ord` は `!(x > 0.0)` を拒否するため NaN安全ガードは `x <= 0.0 || x.is_nan()` を用いる。`cargo fmt` の自動折り返しで行数が増えるため、`piano-cli check-lines` は必ず `cargo fmt` 後に実行する。
- **ポート `127.0.0.1:5173` 回避 & `share` Worktree Web準備**: 未バインドLoopback TCP SYNが約130秒ハングするため、`apps/web/playwright.config.ts` では `PIANOSIM_PORT`（例: `5188`, `--strictPort`）と `webServer.wait: { stdout: /Local:\\s+http/ }` を併用する（E2E約4.4s完走、Chromiumは `--autoplay-policy=no-user-gesture-required --no-sandbox`）。`workspace:share` worktree には `node_modules/` が無いため、Webビルド/E2E前に `npm ci --prefer-offline` を実行する。
- **`AudioWorkletGlobalScope` + Prep Worker + Wasm 境界**: インポート0個の標準 `+simd128` 単一 `.wasm`（`+relaxed-simd` / `extended-const` / `fmaf` 禁止、`WebAssembly.compileStreaming` 共有）を用い、Prep Web Worker + `Transferable ArrayBuffer` で空間/調律事前計算をオフスレッド化する。`apps/web/vite.config.ts` の `?worklet` / `?blobworker` は `esbuild.build({ bundle: true, metafile: true, format: 'esm', target: 'es2022', platform: 'browser' })`＋`addWatchFile` で外部 `import` ゼロの自己完結バンドルを生成し、`dev`・`build`・`/pianosim/` を同一動作させる。Web版 `AudioContext` は `48 kHz` 固定。
- **CI静的・バイナリ検査ゲート**: `.github/workflows/ci.yml` で `cargo fmt --check`、`piano-cli check-lines`、`cargo clippy --workspace -- -D warnings`、`cargo clippy --package piano-wasm --target wasm32-unknown-unknown -- -D warnings`、`cargo test --workspace`、`npm run build`、`piano-cli check-wasm-simd apps/web/public/piano_wasm.wasm`（`0 imports` / `+simd128` 有 / `relaxed-simd` 無）、`npm run test:e2e` を必須実行する。
- **決定論的テストと専用ベンチマーク分離（`piano-cli bench` & `npm run bench:web`）**: 実行時間・ジッター制約を自動テスト（`cargo test` / Playwright `test:e2e`）から完全排除し、テストは機能・音響・数値的決定性のみを検証。時間制約は専用ベンチマーク（ネイティブ: `cargo run --release -p piano-cli -- bench`、Web: `npm run bench:web`）に完全分離して検証する。テスト直後のベンチマーク実行前は `sleep 3` でクールダウンし、カーネルIRQスパイクを避けるため `taskset -c 0` は使用しない。
- **GitHub CLIとPages公開**: `gh` コマンドは未インストール。リモート `origin = git@github.com:likr-sandbox/pianosim.git`、GitHub Pages（`https://likr-sandbox.github.io/pianosim/`、`base: /pianosim/`）は GitHub Actions 公開が有効。確認は `curl`・GitHub API・公開URLへのHeadless Playwrightで行う。

## 5. 音響DSP・検証上の知見（蓄積ログ）
- **Wave 4（II-6, IV-1〜IV-3 音楽表現・客観品質ゲート完成 & 実測ベースライン）**:
  - **II-6 低音 `ff` ファントム `3×6` 拡張**: `k = 0..16` を横12（3 vecs）× 縦24（6 vecs）へ拡張し、`k >= 16` は `n_pol <= 3` を維持。`bridge_sav.rs` / `sav_tile_3x6.rs` / `phantom_3x6.rs` で `#[cfg_attr(target_arch="wasm32", inline(never))]` により Wasm ホットループのレジスタ圧迫と `64@48kHz` RT超過を回避。
  - **IV-1 コンサートピッチ・古典音律・`Condition` マクロ**: `voicing.rs` / `temperament.rs` / `condition.rs` に `concert_pitch_hz`（`392.0..466.2 Hz`）、4古典音律（`Equal`, `WerckmeisterIII`, `KirnbergerIII`, `Vallotti`）の分散前適用、および決定論的 `SplitMix64` `Condition` マクロ（`NoteVoicingOffset[88]`）を実装し、`440.0 Hz + Equal + condition=0.0` でビット完全一致を維持。
  - **IV-2 ヒューマナイズ・Y座標ベロシティ・WAV出力**: 内蔵6曲スコアをヒューマナイズし、`pointer-glissando.ts` で鍵盤Y座標ベロシティ対応（`clientX/Y===0` 合成イベント時は `undefined` 保持＋スライダー中心変調）。`piano-cli`（`wav_writer.rs`）に 24-bit TPDF / 32-bit float WAV 出力、Web にゼロ確保16スロット `WorkletRecorderPool` 録音書き出しを追加。
  - **IV-3 統一Stage 4ゲート・`f0_estimator.rs`・決定論的E2E**: 分散考慮・同時調和最小二乗（joint LS）の `f0_estimator.rs` で A0 40 ms 短窓第2倍音漏れを解消。`challenge3_body_resonance.rs` を Stage 4 `render_block` 比較へ統一し、`challenge4_wave4_gates.rs` に ISO-226 連続性・バイノーラル IACC・`ff` IMD ゲートを追加。`pedals-voicing.spec.ts` は `captureStrikeAttack` / `captureHoldReleaseWindow` と `waitAudioBlocks` 同期で `waitForTimeout` をゼロ化。
  - **Wave 4 検証ベースライン**: `269/269` Rust debug / `270/270` release、`71/71` Playwright E2E、`piano_wasm.wasm` `27,807` `0xfd`（`0` imports / `0` relaxed-simd）、`bench`・`scenarios`・`bench:web` 全合格。
- **Wave 3（III-1〜III-4 空間音響・マルチボディ・出力段完成 & 実測ベースライン）**:
  - **III-1 `HallFdn` (`hall_fdn.rs`, `490`行)**: Zero-Alloc 0遅延 `F32x4`×2 6タップ初期反射＋8×8 Hadamard FDN（`+10.4..13.4 us` delta @ 128 frames）、`< 400 Hz` Mid/Side バイノーラル IACC シェルビング（`+2.5 / -18.0 dB`、IACC `<350Hz=0.94..0.99` vs `>1.2kHz=-0.08..-0.13`）＋拡散EQ。`EngineCore::new` 既定値は `decay_s=1.6, mix=0.18`（`RadiatorBank::from_preset*` は `m6_gate2` 直接畳み込み等価のため `mix=0.0` 維持）。Web `SpacePanel` にボディ／リバーブスライダを追加（`wasm-host.ts` から `pipeline-bootstrap.ts` を分離）。
  - **III-2 マルチボディ `.pbdy` 極零点同期**: 純Rust生成 `concert_grand_v1.pbdy`（`1,372` モード, `62,372` B）、`upright_studio_v1.pbdy`（`1,305` モード, `59,357` B）、`vintage_saloon_v1.pbdy`（`1,342` モード, `61,022` B）。`AdmittanceSsm::from_reduced_model` + `BodyModelCache` + `SpaceCrossfadeGuard` で Stage 3 & Stage 4 を同期（`cancel_err=0.0000 dB`、`44.1/48/96 kHz` 全帯域 WDF 正実性維持）。
  - **III-3 `15×20` Ritz & 高域ベルベット拡散テール**: `15×20` Ritz 基底（`300` 基底）／`48×64` 求積格子（`3,072` 点）を 2D テンソル積 sum-factorization（`plate_assembly.rs`）+ Householder QL（`linalg_eigen.rs`）+ `OnceLock` テストキャッシュで計算（`piano-body-gen` debug ~`7–23 s`）。`ir_synth.rs` に `(zone, ch)` 無相関 `> 3.5 kHz`（`SplitMix64`, `3.8 kHz` 4次HPF）ベルベット拡散テールを追加（`synthesize_irs p50=17.0 ms < 40 ms`、`44.1/48/96 kHz <= 0.222 dB`）。
  - **III-4 `StereoLimiter` (`limiter.rs`)**: 0遅延ステレオリンク `max(|L|, |R|)`（`0.2 ms` att / `35 ms, 180 ms` 2段 rel）、`<= 0.91` ユニティ通過、検出器上限 `4.0`、有理 $C^2$ シーリング、`RadiatorBank::reset()` でリセット。
  - **Wave 3 検証ベースライン & 要注意ファイル**: `242/242` Rust テスト、`65/65` Playwright E2E、`piano_wasm.wasm` `26,595` `0xfd` オペコード（`0` imports / `0` relaxed-simd）、`bench`（`RadiatorBank p99=98.7 us`, `64@48kHz p99=645.0 us` — 競合下マージン `21.7 us` に留意）、`scenarios` 12/12、`bench:web`（`0` stalls）。`presets.rs` と `hall_fdn.rs` が `490/500` 行のため追加時は要分割。
- **Wave 2（II-1〜II-5 Stage 1〜3 連続性完成 & 実測ベースライン）**:
  - **II-1 低音域改善 & 有限ハンマー**: 有限ハンマー接触幅による解析的空間窓 $\Phi_w(j)$ + $\beta(k)$（FTO #2）と `AdaaPowerLut` $C^1$ 3次Hermite `(f1_table, f_table)` を実装。低音 `max_j=20..24`（`n_pol<=6`）、Zone `0..1` 再校正により `A0..C2` 対 `C4` を `+0.98 dB` へ回復（`m7_stage4_88key` `mf72` 全88鍵スプレッド `19.95 dB`、Challenger 1 `mf64` 全88鍵は `C8=-48.44` vs `G5=-28.18` で `20.26 dB`、Wave 4 `IV-3` 校正時の留意点）。`contact_tests.rs` は `498` 行（追加時は要分割）。
  - **II-2 高音WDF再接続 & デチューンテーパ**: 高音 `cutoff_hz=6,000 Hz`（`max_j=4`）へ引き上げ `A#7..C8` を `n_pol=1` で Stage 3 WDF 再接続、`C5+` ユニゾンデチューン・テーパにより `C8` を `0.18..0.32 cents`（`A4=466 Hz` 時も `< 0.9 Hz` うなり）へ抑制。
  - **II-3 3機構弦減衰則**: `damping_law.rs` へ空気粘性・熱弾性・`k < 26` 銅巻線界面摩擦の3機構 $T_{60}(k, j)$ を導入し `.max(1.85/.45/.25)` クランプを撤廃（`C8` $T_{60} \approx 1.66\text{ s}$）。
  - **II-4 調律連動 `DuplexBank` & 非正規化数フラッシュ**: `DUPLEX_MODE_LAYOUT` 長さ比で `EngineCore::new`・`apply_precomputed_voicing`・`DIRTY_TUNING` 時に追従（倍音 `if-then` なし、`DUPLEX_AUDIO_GAIN=3.0` `+20 dB`）＋Stage 3 `< 1e-20` フラッシュ実装（`HertzBridgeContact::x` は $O(x^2)$ 代数減衰のため実打弦直後の `is_quiescent()` は深休止まで偽のまま残るが非可聴）。
  - **II-5 ダンパー剥離・離鍵棚板・FIR連続性**: 幾何学的クリアランスゼロ交差で `0.93..1.95 mN` $C^2$ パルスを `prefix` モードのみへ注入（`tail_nonzero=false`, `vel>64`, `MAX_PEEL_SLOTS=24`; `pedals.rs:303` 2ms未満ペダルフラッタ用ガード候補あり）、`NoteOff` Zone 8 独立 `KeybedBank` スロット、`HammerOversampleState::launch()` の `prev_sub_forces` 保持を実装。`212/212` Rust・`62/62` E2E・`26,236` `0xfd`。
- **Wave 1（I-1〜I-5 制御・境界修正）**: `BlobStreamer` (`blob-streamer.ts`) による `pendingVoicing`/`pendingSpace` 共有 `staging` 直列化と `checked_add` ガード、`u7ToU16` 非圧縮化＋線形 `cc7ToU16`＋`gamma=2.15`＋`Touch Curve`、`EventQueue` Zero-Alloc 持ち越し＋`MidiPlayer` 60ms先読み、`WebMidiManager` 双方向結線＋連続CC67＋Una Corda一本化、共有24点 $p$-グリッド `PGridLuts` (`384 KB`)＋Hermite `adaa1_lerp`（`190/190` Rust・`62/62` E2E）。
- **M7〜M8 音響・低遅延アーキテクチャ基盤**:
  - **Stage 2〜4 駆動**: 単一弦SAV結合＋`DampedSymp`・閉ダンパー予圧平衡・ゼロスキップ・`-115〜-120 dB` 地平線縮退・サステイン受動共鳴降格・SAV `-55 dB` 平滑休眠；Stage 3 `solve_phi` 事前計算SSM移動度・Sherman-Morrison バイパス・`BridgeBank`・解析 `DuplexBank`・3ペダル創発・`ZoneRunCache`；Stage 4 Zones 0..7/Zone 8 SI校正＋NUPOLS（4フレームSIMD FIR head＋4x千鳥FFT、0遅延、誤差 <= -130.28 dB）＋`piano-body-gen`（`concert_grand_v1.pbdy`）＋極零点相殺・Zone 9棚板衝撃・20msボイシング平滑化・8ms二乗余弦クロスフェード。
  - **非同期 Prep Worker & 時刻脱量子化**: `Transferable ArrayBuffer`＋8ms二重バッファ切替、`hammer_hardness` Voicing Dirty Mask 即時反映。AudioWorklet 32ブロック平均（31.25 µs分解能）と HUD Peak/RMS (dBFS)・dB SPL（-18 dBFS ≈ 82 dB SPL）表示。
- **旧v1実装・`no_std`・Zero-Alloc・ステレオ/E2E誠実性**: `v0.1.0-ts-dwg`（`c75cf57`）アーカイブ済み。`[SampleEvent; 512]` はインプレース挿入ソート。`piano-simd`/`spec`/`strings`/`action`/`bridge` は `#![cfg_attr(not(test), no_std)]`（外部クレートゼロ）。L/R 出力バッファ非エイリアス、`check-wasm-simd` Code section 実 `0xfd` 計数、Playwright E2E はモック/`test.skip` 禁止。
- **Exact Coupled-Form ZOH & $\gamma_k$-不変SIデュアルポート**: `coupled_pole_f32` は `s_f32` 先量子化＋`s_ratio^2 / (1 + cos_mag)` 求解で回転誤差 `<= 0.00257 cents`・SAV保存 `< 1e-5`。$\tilde{u}_j=\gamma_k \dot{q}_j, \tilde{v}_j=\gamma_k \Omega_j q_j$ と `alpha_vel` で厳密SIポート量と $\gamma_k$ 不変性を維持。
- **直交V/H 460スロット片側ダンパー（FTO #3） & 因果ブリッジ**: 垂直(`2*s`)/水平(`2*s+1`)パッド分離＋`long_vecs` 除外で零空間解消（250msで `> 49 dB` 減衰）、`ModalSoa` 極は非減衰保持。ブリッジ射影は `C_VERT=1.0`/`C_HORIZ=0.18`。横符号 `(-1)^(j-1)`＋傾斜＋`LongitudinalBank`（`5.2 kHz` ニー/`14.5 kHz` カットオフ）で $\tau_L < \tau_T$ 因果プリカーサーを実現。
- **Drift-regulated SAV + 局所 `8×` アクション + 高域校正**: `SavPairQueue` (`SUB=128`) + Cayley直交ステップ（`LAMBDA_REG_DEFAULT=400.0`）＋`0..52` 低中音テーパ。1次Timoshenko＋Stulov＋Borin K/1次ADAA＋16タップFIR＋O(1) `PedalBank` で全25.5kモード駆動（末尾 `< 1e-20` フラッシュ）。C8 第1部分音は分散適用値 `partial_frequency_hz(key_tuned_f0_hz, B, 1)`（~4358 Hz）と比較。
- **Webデモスコア・物理プリセット・コンサートピッチA4**: `apps/web/src/midi/scores/`（6曲・全88鍵 [21,108]）と `PresetBar` 4種双方向同期。基準ピッチ `A4`（Wave 1 で `415.0〜466.0 Hz` 導入、Wave 4 `IV-1` で `392.0〜466.2 Hz` ＋4古典音律・`Condition` へ拡張）は Mersenne則 $T_0 \propto F_0^2$ で張力・$B$・ストレッチ・`DuplexBank` 64モードを追従（`440.0 Hz` + `Equal` + `condition=0.0` 時ビット完全一致、6変数 `VoicingSmoother`）。

## 6. 本ファイルの更新履歴（直近マイルストーン要約）
- **Wave 4 (`wave4`, 2026-11-18)**: `II-6`, `IV-1〜IV-3` を統合（低音 `3×6` ファントム & `inline(never)` 隔離、`392.0..466.2 Hz` A4・4古典音律・`SplitMix64` `Condition`、デモ6曲ヒューマナイズ・Y座標ベロシティ・24b TPDF/32f WAV & `WorkletRecorderPool`、`f0_estimator.rs` 修正・Stage 4 `challenge3` 統一・ISO-226/IACC/`ff` IMD ゲート・E2E `waitAudioBlocks` 同期）。`269/269` Rust・`71/71` E2E・`27,807` `0xfd`・`bench`/`scenarios`/`bench:web` 全合格。
- **Wave 3 (`wave3`, 2026-11-18)**: 空間音響・複数ボディ・出力段 `III-1..III-4` を統合（`HallFdn` 6タップER+8×8 Hadamard FDN & IACCシェルビング、3種 `.pbdy` 極零点同期 & `SpaceCrossfadeGuard`、`15×20` Ritz sum-factorization + `>3.5kHz` ベルベットテール、0遅延 `StereoLimiter`）。`242/242` Rust・`65/65` E2E・`26,595` `0xfd`・`bench`/`scenarios`/`bench:web` 全合格。
- **Wave 2 (`wave2`, 2026-11-18)**: Stage 1〜3 音色・音域連続性 `II-1..II-5` を統合（有限ハンマー $\Phi_w(j)$ + `AdaaPowerLut` Hermite + Zone `0..1` 校正 & `.pbdy` 再ベイクで `A0..C2` 回復、高音 `6 kHz` WDF 再接続 + `C5+` デチューンテーパ、3機構 $T_{60}(k, j)$、調律連動 `DuplexBank` `+20 dB` + `< 1e-20` フラッシュ、幾何ダンパー剥離・離鍵 Zone 8・FIR 連続保持）。`212/212` Rust・`62/62` E2E・`26,236` `0xfd`・`bench`/`scenarios` 全合格。
- **Wave 1 (`wave1`, 2026-11-18)**: P0制御・境界修正 `I-1〜I-5` ＋コンサートピッチ `A4`（`415.0〜466.0 Hz`）統合（`BlobStreamer` 直列化、`u7ToU16`/`cc7ToU16`＋`gamma=2.15`＋`Touch Curve`、`EventQueue` carryover＋`MidiPlayer` 60ms先読み、`WebMidiManager`・連続CC67・重複減衰一本化、共有24点 $p$-グリッド `384 KB` LUT）。`190/190` Rust・`62/62` E2E 合格。
- **Web Demo Scores & Parameter Presets (2026-11-17)**: クラシック6曲デモスコア（`apps/web/src/midi/scores/`）と4種物理プリセット（`PresetBar`）実装、E2E 54/54合格。
- **M8 (2026-11-16)**: Balasubramaniam 複合負荷＆64サンプル@48 kHzゲート・3大課題 `piano-cli scenarios` 全パス、全184テスト・E2E 40/40合格、Pages公開。
- **M7 (2026-11-16)**: 低遅延再設計（Stage 2〜4 最適化、Prep Worker＋8msクロスフェード、Voicing Dirty Mask、`piano-cli bench`/`bench:web` 分離）、全172テスト・E2E 40/40合格。
- **M4〜M6 (2026-11-15)**: Stage 2 非線形弦（`coupled_pole_f32`, `LongitudinalBank`, SAV+ファントム）、Stage 3 WDFブリッジ（`BridgeBank`, `DuplexBank`, 3ペダル創発）、Stage 4 放射器（`piano-body-gen`, NUPOLS 0遅延畳み込み）統合。
- **M1〜M3 (2026-11-15)**: `v0.1.0-ts-dwg` 保全後に10クレート純Rustモノレポ・88鍵SIMD ZOH・局所 `8×` Stage 1 アクション構築。
- **Pre-M1 (2026-11-14)**: 初版作成（`ARCHITECTURE.md` / `ROADMAP.md` / `DECISIONS_AND_FALLBACKS.md` / `SWARM_WORKFLOW.md`）。
