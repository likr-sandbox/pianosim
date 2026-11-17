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
- **Wave 1（I-1〜I-5 制御・境界修正 & 実測ベースライン）**:
  - **I-1 `BlobStreamer` 共有 `staging` 直列化**: `worklet-processor.ts` から `blob-streamer.ts` を分離し、`pendingVoicing`/`pendingSpace` 転送中（`off > 0`）は他方と後続同種ブロブをキュー待機させて上書きを防止（`vcLen=309,372` 約101ms、`spLen=884,416` 約288ms @ 8,192 floats/2.67ms block、切替完了 `~300–420 ms` のため E2E は `450–550 ms` settle 待機）。`WasmContext::stage_space_chunk` に `checked_add` 境界ガードを追加。
  - **I-2 ペダル線形化 & ベロシティ再校正**: `abi.ts` で `u7ToU16` の `14000` 下限圧縮を撤廃（`u7ToU16(1)=516`）しペダル専用線形 `cc7ToU16` を分離。`main_velocity_curve` を `gamma=2.15`（`0.075 + 0.86*u + 3.665*u^2.15`：`MIDI 1=0.082 m/s > v_esc`, `mf=u7_to_u16(64)=33_025` が `1.35 m/s`, `MIDI 100=51_602` が `2.94 m/s`, `MIDI 127=4.60 m/s`）へ再校正し、Web に `Touch Curve`（`light 0.65 / natural 1.0 / heavy 1.65`）を追加（`m3_tests`/`m5_tests` の旧 `mf` 代理値 `51_602`/`48_000` は `33_025` へ更新）。
  - **I-3 マルチブロック先読み & Carryover**: `EventQueue::partition_for_block` / `advance_block` と `WasmContext::render` により `sample_offset >= n` イベントを Zero-Alloc で次ブロックへ持ち越し（`engine_core.rs:154-157` 単体契約を保持）、`all_notes_off` で将来イベントも破棄。`MidiPlayer` は `60 ms` 先読み＋`seek()` 時 CC64/66/67 復元を実装。
  - **I-4 `WebMidiManager` 双方向結線 & Una Corda 一本化**: `boundInputs` 単一リスナ重複排除、`0x80` 離鍵ベロシティ伝達（`0x90 vel=0 -> 32768`）、自動 `unlock()`、`3 ms`（`144` samples @ 48 kHz）ジッタ吸収窓、連続 CC67 Una Corda（`[0.0, 1.0]`）を実装。`piano-action` の重複減衰（`lib.rs` 速度乗算と `oversample.rs` `shoulder_soft` 剛性乗算）を撤廃し `una_corda_felt_mod`＋`strike_weights`＋`eps_eff` 上限 `0.80` へ一本化。
  - **I-5 共有 $p$-グリッド LUT & 動的 `exponent_p`**: 88鍵個別 LUT（`1,408 KB`）を `ActionBank` 共有 24点 $p \in [2.10, 3.50]$ グリッド `PGridLuts`（`384 KB`、`-1,024 KB`、初期化 ~3.6x 高速化）＋ `adaa1_lerp` / `solve_dimless_lerp` へ置換し `< 1 us` で動的 `exponent_p` 更新を実現。`PGridLuts::lookup` は `+inf` が上限側へクランプされるよう `exponent_p.is_nan() || exponent_p <= P_GRID_MIN` で判定する。
  - **Wave 1 構造・検証ベースライン**: `param_smooth.rs` -> `space_guard.rs`、アクションテスト（`pedals_tests.rs`/`shank_timoshenko_tests.rs`）、Web `shell-template.ts` を分割。`apps/web/e2e/midi-playback.spec.ts` は `490` 行（500行上限近接のため追加時は分割必須）。`piano_wasm.wasm` `14,668` `0xfd` オペコード（`1,172` `v128` locals, `0` imports）、全 `186/186` Rust テスト・`59/59` E2E・`scenarios` 12/12・D1 ベンチ合格。
- **M7〜M8 音響・低遅延アーキテクチャ基盤**:
  - **Stage 2〜4 駆動**: 単一弦SAV結合＋`DampedSymp`・閉ダンパー予圧平衡・ゼロスキップ・`-115〜-120 dB` 地平線縮退・サステイン受動共鳴降格・SAV `-55 dB` 平滑休眠；Stage 3 `solve_phi` 事前計算SSM移動度・接触ゼロ時 Sherman-Morrison バイパス・`BridgeBank`・解析 `DuplexBank`・3ペダル創発・`reduce_sum_4`/`ZoneRunCache`/`render_segment`；Stage 4 Zones 0..7（×300〜350）および Zone 8（×0.025）SI音圧校正＋NUPOLS（4フレームSIMD FIR head＋4x千鳥分割FFT、0追加遅延、RMS誤差 <= -130.28 dB）＋`piano-body-gen`（Ritz/統計約1,500モード、`concert_grand_v1.pbdy` 57.6 KB）＋極零点相殺・Zone 9棚板衝撃・20msボイシング平滑化・8ms二乗余弦クロスフェード。
  - **非同期 Prep Worker & 時刻脱量子化**: `Transferable ArrayBuffer`＋8ms二重バッファ切替、`hammer_hardness` Voicing Dirty Mask 即時反映。AudioWorklet 1ms `Date.now()` 量子化を32ブロックストライド平均（31.25 µs分解能）で解消し、HUD で全ブロック Peak/RMS (dBFS) と校正 dB SPL（-18 dBFS ≈ 82 dB SPL）を表示。
- **旧v1実装の保全**: 旧TS製DWG実装（`c75cf57`）はGitタグ `v0.1.0-ts-dwg` にアーカイブ済み（比較時は `git worktree add /tmp/pianosim-v1 v0.1.0-ts-dwg`）。
- **ゼロアロケーションと `no_std` コア**: `[T]::sort_by` はヒープ確保の可能性があるため固定長 `[SampleEvent; 512]` キューはインプレース挿入ソートを使う。`piano-simd` / `piano-spec` / `piano-strings` / `piano-action` / `piano-bridge` は `#![cfg_attr(not(test), no_std)]`（`alloc` / 純Rust `f64` 初期化数学）とし、外部クレート依存ゼロを維持する。
- **Exact Coupled-Form ZOH & $\gamma_k$-不変SIデュアルポート**: `f32` Direct-form `2*r*cos(theta)` は低音（A0 @ 96 kHz で ~15.8 cents誤差）で破綻するため、`coupled_pole_f32` は `s_f32` を先に量子化し $(1-d)^2 + s_{\text{f32}}^2 = r_{\text{hold}}^2$ を `s_ratio^2 / (1 + cos_mag)` で解く（無損失時誤差 `<= 5e-10` @ 44.1/48/96 kHz、ZOH回転誤差 `<= 0.00257 cents`、`f32` 状態での `< 1e-5` SAVハミルトニアン保存を保証）。ZOH状態 $\tilde{u}_j=\gamma_k \dot{q}_j, \tilde{v}_j=\gamma_k \Omega_j q_j$ と $b_j=\gamma_k \frac{2T_s}{\mu L}\phi_j$, $a_j=\frac{\phi_j}{\Omega_j\gamma_k}$, `alpha_vel` ($\alpha_{\text{vel}}=\frac{\mu L}{2T_s\gamma_k^2}$) により追加配列なしで厳密SIポート量と $\gamma_k$ 不変性を実現する。
- **直交V/H 460スロット片側ダンパー（FTO #3） & V/Hブリッジ射影**: 垂直（`2*s`）と水平（`2*s+1`）のダンパーパッド（`DAMPER_PAD_SLOTS = 460`）を分離し、`long_vecs` を `b_damp`/`a_damp` から除外することで `V + 0.85 H == 0`（`HORIZ_DAMP_SCALE = 0.85`）のランク1零空間を解消し（`250 ms` で `> 49 dB` 減衰）、`ModalSoa` の極（`d_coeff`, `s_coeff`）は恒久的に非減衰のまま保持する。一方ブリッジ射影は `modal_soa` の `C_VERT = 1.0` / `C_HORIZ = 0.18`（`step_prefix_run_k` は `C_HORIZ / C_VERT = 0.18`）であり、水平ブリッジ連成を `0.78` 等と誤認しないこと。
- **因果ブリッジ & 縦振動プリカーサー（`piano-strings`）**: 横モード空間符号 `(-1)^(j-1)` + ブリッジ傾斜（`long_bridge_weight`）+ 打弦点局所縦振動プリカーサー（`LongitudinalBank`、4次 `5.2 kHz` ニー / `14.5 kHz` カットオフ、`t60_longitudinal` 基準 `0.15 s`）により、`tau_T` より先に `tau_L` で因果的プリカーサーがブリッジへ到達する。
- **Drift-regulated SAV + Bank–Sujbert ファントム & 音域テーパ**: 統合二乗和Kirchhoff＋ファントムエネルギー $E_{\text{nl}} \ge 0$（`SavPairQueue` `SUB=128` バッチ + `CHUNK=128` ZOH + `F32x4` レーンスライド畳み込み）をCayley直交 Risse–Hélie–Bilbao ステップ（`LAMBDA_REG_DEFAULT = 400.0`）で解き、`nonlinear_register_weight` 低中音テーパ（`0..LONGITUDINAL_MAX_KEY_IDX=52`）で高音・リバトゥート不変性を維持する。
- **局所 `8×` アクション・定常SIMD駆動**: 1次Timoshenkoシャンク＋Stulov羊毛記憶＋Borin K法/1次ADAA中点法＋16タップBlackman-sinc FIR＋`PedalBank::settling` O(1) クリアランスで全約25.5kモードを常時駆動する（末尾で `< 1e-20` 非正規化数フラッシュ、`F32x4` に `Neg` なし）。
- **音域・高域・分散キャリブレーション**: 音域依存打弦ロールオフ（`slope = 0.12 + 1.80*x`）により A4 基音が第2倍音を +5.9 dB 上回る。C8 第1部分音は剛性分散により `partial_frequency_hz(key_tuned_f0_hz, B, 1)`（~4358 Hz）へシフトするため、高音ピッチ検証は分散適用後の理論部分音と比較する。C6〜C8は鍵長連動パルス（`~0.20*T0`）と攻撃RMS・T60を自動検証する。
- **ステレオ・Wasm SIMD・E2E検査の誠実性**: L/R 出力バッファは絶対にエイリアスさせない。`check-wasm-simd` は Code section（ID 10）の命令をデコードして実 `0xfd` オペコードを数える。Playwright E2E は合成モック・`test.skip` 禁止で実Wasm `AudioWorklet` + 受動 `AnalyserNode` を検証する。
- **Webデモスコア & 物理プリセット（Web UI/MIDI）**: `ParsedSmf` 生成器を `apps/web/src/midi/scores/` に分割し全ファイル500行上限を維持（全イベント 88鍵 [21, 108] 厳守）。`PresetBar` は `host.setVoicing` / `host.setSpace`・スライダー操作時 `custom` 自動移行を双方向同期。

## 6. 本ファイルの更新履歴（直近マイルストーン要約）
- **Wave 1 (`wave1`, 2026-11-18)**: P0制御・境界修正 `I-1〜I-5` を実装・統合（`esbuild` `?worklet`/`?blobworker` バンドル、共有 `staging` `BlobStreamer` 直列化＋境界ガード、`u7ToU16` 非圧縮化＆線形 `cc7ToU16`＋`gamma=2.15` 速度曲線＋`Touch Curve`、`EventQueue` マルチブロック carryover＋`MidiPlayer` 60ms先読み＆`seek` ペダル復元、`WebMidiManager` 双方向結線・3msジッタ窓・`0x80` 離鍵速度・連続CC67 Una Corda・重複減衰一本化、共有24点 $p$-グリッド `384 KB` LUT＋動的 `exponent_p`）。全 `186/186` Rust テスト・`59/59` E2E・`scenarios` 12/12・D1ベンチ合格。
- **Web Demo Scores & Parameter Presets (2026-11-17)**: クラシック6曲デモスコア（`apps/web/src/midi/scores/`）と4種物理プリセット（`PresetBar`）を実装、E2E 54/54合格。
- **M8 (2026-11-16)**: Balasubramaniam 複合負荷＆64サンプル@48 kHzゲート・3大課題 `piano-cli scenarios` 全パス、全184テスト・E2E 40/40合格、Pages公開。
- **M7 (2026-11-16)**: 低遅延再設計（Stage 2 SAV/DampedSymp、Stage 3 SSM事前計算/SMバイパス、Stage 4 SI音圧校正、Prep Worker＋8msクロスフェード、Voicing Dirty Mask、AudioWorklet時刻脱量子化、`piano-cli bench`/`bench:web` 分離）、全172テスト・E2E 40/40合格。
- **M4〜M6 (2026-11-15)**: Stage 2 非線形弦（`coupled_pole_f32`, `LongitudinalBank`, SAV+ファントム）、Stage 3 WDFブリッジ（`BridgeBank`, `DuplexBank`, 3ペダル創発）、Stage 4 放射器（`piano-body-gen` 1,500モードベイク, NUPOLS 0遅延畳み込み, 二重共振解消, ボイシング平滑化）を統合。
- **M1〜M3 (2026-11-15)**: `v0.1.0-ts-dwg` 保全後に10クレート純Rustモノレポ・88鍵SIMD ZOH・局所 `8×` Stage 1 アクション（Timoshenko/Stulov/Borin K+ADAA1/V-H 460ダンパー）を構築。
- **Pre-M1 (2026-11-14)**: 初版作成（設計仕様 `ARCHITECTURE.md` / `ROADMAP.md` / `DECISIONS_AND_FALLBACKS.md` / `SWARM_WORKFLOW.md`）。
