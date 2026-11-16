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
- **`rustfmt` 1.97.1 と行数検査**: 10段超の深いネストの右結合Horner多項式式は `rustfmt` がCPU 100%でハングするため `const` 配列ループ（`eval_horner` 等）で評価する。また `cargo fmt` の自動折り返しで行数が増えるため、`piano-cli check-lines` は必ず `cargo fmt` 後に実行する。
- **ポート `127.0.0.1:5173` 回避 & `share` Worktree Web準備**: 未バインドLoopback TCP SYNが約130秒ハングするため、`apps/web/playwright.config.ts` では `PIANOSIM_PORT`（例: `5188`, `--strictPort`）と `webServer.wait: { stdout: /Local:\\s+http/ }` を併用する（E2E約4.4s完走、Chromiumは `--autoplay-policy=no-user-gesture-required --no-sandbox`）。`workspace:share` worktree には `node_modules/` が無いため、Webビルド/E2E前に `npm ci --prefer-offline` を実行する。
- **`AudioWorkletGlobalScope` + Prep Worker + Wasm 境界**: インポート0個の標準 `+simd128` 単一 `.wasm`（`+relaxed-simd` / `extended-const` / `fmaf` 禁止、`WebAssembly.compileStreaming` 共有）を用い、Prep Web Worker + `Transferable ArrayBuffer` で空間/調律事前計算をオフスレッド化する。Vite `transformWithEsbuild` で `dev`・`preview`・`/pianosim/` を同一動作させる。Web版 `AudioContext` は `48 kHz` 固定。
- **CI静的・バイナリ検査ゲート**: `.github/workflows/ci.yml` で `cargo fmt --check`、`piano-cli check-lines`、`cargo clippy --workspace -- -D warnings`、`cargo clippy --package piano-wasm --target wasm32-unknown-unknown -- -D warnings`、`cargo test --workspace`、`npm run build`、`piano-cli check-wasm-simd apps/web/public/piano_wasm.wasm`（`0 imports` / `+simd128` 有 / `relaxed-simd` 無）、`npm run test:e2e` を必須実行する。
- **決定論的テストと専用ベンチマーク分離（`piano-cli bench` & `npm run bench:web`）**: 実行時間・ジッター制約を自動テスト（`cargo test` / Playwright `test:e2e`）から完全排除し、テストは機能・音響・数値的決定性のみを検証。時間制約は専用ベンチマーク（ネイティブ: `cargo run --release -p piano-cli -- bench`、Web: `npm run bench:web`）に完全分離して検証する。テスト直後のベンチマーク実行前は `sleep 3` でクールダウンし、カーネルIRQスパイクを避けるため `taskset -c 0` は使用しない。
- **GitHub CLIとPages公開**: `gh` コマンドは未インストール。リモート `origin = git@github.com:likr-sandbox/pianosim.git`、GitHub Pages（`https://likr-sandbox.github.io/pianosim/`、`base: /pianosim/`）は GitHub Actions 公開が有効。確認は `curl`・GitHub API・公開URLへのHeadless Playwrightで行う。

## 5. 音響DSP・検証上の知見（蓄積ログ）
- **M7（アーキテクチャ再設計・低遅延最適化・実測ベースライン）**:
  - **Stage 2 弦駆動**: 単一弦SAV結合＋チャンク化線形共鳴減衰（`DampedSymp`）、閉ダンパー予圧平衡、厳密ゼロスキップ、物理エネルギー地平線縮退（`-115〜-120 dB`）、サステイン中受動共鳴降格、SAV平滑休眠（`-55 dB`）を統合。
  - **Stage 3 ブリッジ**: 事前計算SSMゾーン移動度（`solve_phi`）、接触ゼロ時の8×8 Sherman-Morrisonバイパス、ステレオ分割複素数FMA。
  - **Stage 4 放射器 & 音圧校正**: Zones 0..7（×300〜350）および Zone 8（×0.025）の物理SI音圧校正、400倍ハック撤廃、DuplexBank結線、低音バッフル改善。
  - **非同期 Prep Web Worker & Voicing Dirty Mask**: 空間IR/FFT事前計算をオフスレッド化し `Transferable ArrayBuffer` ＋ 8ms二重バッファ二乗余弦クロスフェードで無瞬断切り替え。`hammer_hardness` は Voicing Dirty Mask によりオーディオスレッド即時反映（< 1 µs）。
  - **AudioWorklet 時刻脱量子化 & テレメトリ**: `AudioWorkletGlobalScope` の 1ms `Date.now()` 量子化による 2,000 µs 偽オーバーランを解消。32ブロックストライド平均（分解能 31.25 µs）で honest な D1 判定を実施。HUD に全ブロック走査 Peak/RMS (dBFS) および校正 dB SPL（-18 dBFS ≈ 82 dB SPL）を表示。
  - **M7実測ベースライン**: `piano_wasm.wasm` `14,652` `0xfd` オペコード（`1,169` `v128` locals, `0` imports）。ネイティブ `bench`（`128 @ 48 kHz`）12和音アルペジオ `p99=810.0 us`、20ボイスサステイン `p99=920.0 us`（D1上限 `1,866.7 us` に対し余裕度50%超）、RadiatorBank `p99 < 160.0 us`（上限 650 µs）、IR合成 `p50 ≈ 13 ms`（上限 40 ms）。Chromium `bench:web` 定常 `p50=680.0 us / p99=750.0 us`（0 stalls）。
- **旧v1実装の保全**: 旧TS製DWG実装（`c75cf57`）はGitタグ `v0.1.0-ts-dwg` にアーカイブ済み（比較時は `git worktree add /tmp/pianosim-v1 v0.1.0-ts-dwg`）。
- **ゼロアロケーションと `no_std` コア**: `[T]::sort_by` はヒープ確保の可能性があるため固定長 `[SampleEvent; 512]` キューはインプレース挿入ソートを使う。`piano-simd` / `piano-spec` / `piano-strings` / `piano-action` / `piano-bridge` は `#![cfg_attr(not(test), no_std)]`（`alloc` / 純Rust `f64` 初期化数学）とし、外部クレート依存ゼロを維持する。
- **Exact Coupled-Form ZOH & $\gamma_k$-不変SIデュアルポート**: `f32` Direct-form `2*r*cos(theta)` は低音（A0 @ 96 kHz で ~15.8 cents誤差）で破綻するため、`coupled_pole_f32` は `s_f32` を先に量子化し $(1-d)^2 + s_{\text{f32}}^2 = r_{\text{hold}}^2$ を `s_ratio^2 / (1 + cos_mag)` で解く（無損失時誤差 `<= 5e-10` @ 44.1/48/96 kHz、ZOH回転誤差 `<= 0.00257 cents`、`f32` 状態での `< 1e-5` SAVハミルトニアン保存を保証）。ZOH状態 $\tilde{u}_j=\gamma_k \dot{q}_j, \tilde{v}_j=\gamma_k \Omega_j q_j$ と $b_j=\gamma_k \frac{2T_s}{\mu L}\phi_j$, $a_j=\frac{\phi_j}{\Omega_j\gamma_k}$, `alpha_vel` ($\alpha_{\text{vel}}=\frac{\mu L}{2T_s\gamma_k^2}$) により追加配列なしで厳密SIポート量と $\gamma_k$ 不変性を実現する。
- **直交V/H 460スロット片側ダンパー（FTO #3） & V/Hブリッジ射影**: 垂直（`2*s`）と水平（`2*s+1`）のダンパーパッド（`DAMPER_PAD_SLOTS = 460`）を分離し、`long_vecs` を `b_damp`/`a_damp` から除外することで `V + 0.85 H == 0`（`HORIZ_DAMP_SCALE = 0.85`）のランク1零空間を解消し（`250 ms` で `> 49 dB` 減衰）、`ModalSoa` の極（`d_coeff`, `s_coeff`）は恒久的に非減衰のまま保持する。一方ブリッジ射影は `modal_soa` の `C_VERT = 1.0` / `C_HORIZ = 0.18`（`step_prefix_run_k` は `C_HORIZ / C_VERT = 0.18`）であり、水平ブリッジ連成を `0.78` 等と誤認しないこと。
- **因果ブリッジ & 縦振動プリカーサー（`piano-strings`）**: 横モード空間符号 `(-1)^(j-1)` + ブリッジ傾斜（`long_bridge_weight`）+ 打弦点局所縦振動プリカーサー（`LongitudinalBank`、4次 `5.2 kHz` ニー / `14.5 kHz` カットオフ、`t60_longitudinal` 基準 `0.15 s`）により、`tau_T` より先に `tau_L` で因果的プリカーサーがブリッジへ到達する。
- **Drift-regulated SAV + Bank–Sujbert ファントム & 音域テーパ**: 統合二乗和Kirchhoff＋ファントムエネルギー $E_{\text{nl}} \ge 0$（`SavPairQueue` `SUB=128` バッチ + `CHUNK=128` ZOH + `F32x4` レーンスライド畳み込み）をCayley直交 Risse–Hélie–Bilbao ステップ（`LAMBDA_REG_DEFAULT = 400.0`）で解き、`nonlinear_register_weight` 低中音テーパ（`0..LONGITUDINAL_MAX_KEY_IDX=52`）で高音・リバトゥート不変性を維持する。
- **局所 `8×` アクション・定常SIMD駆動・M4ベースライン**: 1次Timoshenkoシャンク＋Stulov羊毛記憶＋`BorinKLut`/`AdaaState`（`AdaaPowerLut`）中点法＋16タップBlackman-sinc FIR＋`PedalBank::settling` O(1) クリアランスで全約25.5kモードを常時駆動する（末尾で `< 1e-20` 非正規化数フラッシュ、`F32x4` に `Neg` なし）。M4実測: `piano_wasm.wasm` `9,035` `0xfd` オペコード（`0` imports）、ネイティブ `128 @ 48 kHz` `p50=1170.0 us (43.9%) / p99=1408.7 us (52.8%)`、Chromium `AudioWorklet` `mean=1280.9 us (48.03%), p99=1393.7 us (52.27%), 0 underruns`、デバッグテスト `piano-engine` 約 `208s` / `piano-strings` 約 `211s`（`--release` 約 `3.2s` / `3.6s`）。
- **M6（Stage 4 放射器・ボディ生成・ボイシング・実測ベースライン）**: `piano-body-gen`（Layer 1A Ritz/統計約1,500モード・8ゾーン響板＋リム・棚板・共鳴腔連成、決定論的ベイク `concert_grand_v1.pbdy` 57.6 KB）、Stage 4 NUPOLS（4フレームSIMD FIR head＋4x千鳥分割FFT、0サンプル追加遅延、直接畳み込み比RMS誤差 -130.28 dB以下）、`PianoEngine` 直列接続（Stage 3 複素アドミタンスとの極零点相殺による二重共振解消、Zone 9棚板衝撃直接加算、20ms指数ボイシング平滑化、8ms二重バッファ二乗余弦空間/大屋根クロスフェード、C-ABI `piano_set_voicing`/`piano_set_space`）を統合。M6実測: `piano_wasm.wasm` `13,044` `0xfd` オペコード（`720` `v128` locals, `0` imports）、ネイティブ `128 @ 48 kHz` `p50=1170.5 us (43.9%) / p99=1728.9 us (64.8%)`、ワークスペース単体・音響結合テスト 158/158+ 合格、Playwright E2E 4層38テスト基盤確立。
- **M5（Stage 3 WDFブリッジ・SIMD最適化・実測ベースライン）**: `BridgeBank`（8ゾーン複素モーダル・アドミタンス Exact Pole-Matched ZOH + 瞬間WDF散乱 $R_{\text{tot},z}^{-1}$ + 8ゾーンSIMD散乱/集約 `ZoneRunCache`）、`DuplexBank`（倍音関係 `if-then` 判定なしの解析モデル）、3ペダル創発（Sostenuto / Una Corda打弦シフト / 連続Half-Pedal）を実装。さらに `F32x4` `unpack_lo`/`unpack_hi`・レーンシャッフル＋`reduce_sum_4`（4弦並列WDFプレフィックス合算）＋3ベクトル展開 `accum_soa_range_in`＋`render_segment` サンプル精度 mid-segment イベント処理を導入。`piano-cli bench` 実行規約として `taskset -c 0` はカーネルIRQ割り込みスパイクのため禁止し、テスト直後は `sleep 3` でクールダウンする。M5実測: `piano_wasm.wasm` `10,361` `0xfd` オペコード（`576` `v128` locals, `0` imports）、ネイティブ `128 @ 48 kHz` `p50=1169.7 us (43.9%) / p99=1746.4 us (65.5%)`、Chromium Wasm `AudioWorklet` `128 @ 48 kHz`（`1,152` callbacks）`mean=1733.3 us (65.00%), p50=1709.4 us (64.10%), p99=1846.9 us (69.26%), 0 underruns`（D1閾値 `1,866.7 us` まで残り `~20 us` の余裕のため M6 予算計画に留意）、未最適化デバッグ `cargo test --workspace` は `piano-engine 461.37s / piano-strings 234.81s / piano-wasm 17.31s`（`--release` は `12.32s / 3.83s / 0.28s`）。
- **音域・高域・分散キャリブレーション**: 音域依存打弦ロールオフ（`slope = 0.12 + 1.80*x`）により A4 基音が第2倍音を +5.9 dB 上回る。C8 第1部分音は剛性分散により `partial_frequency_hz(key_tuned_f0_hz, B, 1)`（~4358 Hz）へシフトするため、高音ピッチ検証は分散適用後の理論部分音と比較する。C6〜C8は鍵長連動パルス（`~0.20*T0`）と攻撃RMS・T60を自動検証する。
- **ステレオ・Wasm SIMD・E2E検査の誠実性**: L/R 出力バッファは絶対にエイリアスさせない。`check-wasm-simd` は Code section（ID 10）の命令をデコードして実 `0xfd` オペコードを数える。Playwright E2E は合成モック・`test.skip` 禁止で実Wasm `AudioWorklet` + 受動 `AnalyserNode` を検証する。

## 6. 本ファイルの更新履歴（直近マイルストーン要約）
- **M8 (2026-11-16)**: 実負荷ストレステスト・3大聴覚課題ベンチマーク・最終最適化＆本番公開を達成。Balasubramaniam（DAFx 2026）複合負荷ベンチマーク（10-12和音＋32分音符アルペジオ＋連続ペダル＋ライブボイシング）および64サンプル@48 kHzリアルタイムゲート（p99 <= 666.7 us）をクリア。3大課題客観的音響検証スイート（`piano-cli scenarios`、縦波プリカーサー先行到達・打弦チャープ・Stage 4 0遅延FIR、12 Hz連打Stulov羊毛メモリ変動、2段階減衰・Lehtonen背景共鳴利得・棚板衝撃力単調性）を自動化し全パス。`bench_rt.rs` を246行（< 300行）へモジュール分割。全184単体・結合テスト合格、Playwright E2E 40/40合格、GitHub Pages本番デプロイ。
- **M7 (2026-11-16)**: アーキテクチャ根本再設計＆Web低遅延化（Stage 2 SAV/DampedSymp最適化、Stage 3 SSMゾーン移動度事前計算/Sherman-Morrisonバイパス、Stage 4 SI音圧校正/Duplex結線、非同期Prep Web Worker＋8msクロスフェード、Voicing Dirty Mask、AudioWorklet 1ms Date.now()脱量子化テレメトリ、決定論的テストと専用ベンチマーク分離 `piano-cli bench` & `npm run bench:web`）を達成。全172単体/結合テスト合格、Playwright E2E 40/40合格、ネイティブ・Web共に全D1レイテンシ制約（< 70% budget）をクリア。
- **M6 (2026-11-15)**: Stage 4 放射器・ボディ生成・ボイシング統合（`piano-body-gen` 1,500モードベイク 57.6 KB、NUPOLS 0遅延8ゾーン千鳥畳み込み誤差 <= -130.28 dB、二重共振解消、Zone 9棚板衝撃、ボイシング平滑化・空間クロスフェード、C-ABI `piano_set_voicing`/`piano_set_space`、4層38テストPlaywright E2E基盤確立）を達成し、全158+テスト合格および M6 ネイティブ D1 ベースライン（p99=1,728.9 us）を達成。
- **M5 (2026-11-15)**: Stage 3 複素モーダルWDF双方向ブリッジ（`BridgeBank` + 8ゾーン `ZoneRunCache`）、解析デュプレックス（`DuplexBank`）、3ペダル創発（Sostenuto / Una Corda / 連続Half-Pedal）、L1dプレフィックス＋`reduce_sum_4`＋3ベクトル `accum_soa_range_in` SIMD最適化を統合し、ワークスペース `112/112` テスト合格および M5 ネイティブ／Chromium D1 ベースラインを達成。
- **M4 (2026-11-15)**: Stage 2 非線形弦（単位円整合 `coupled_pole_f32`、因果ブリッジ＋縦振動プリカーサー `LongitudinalBank`、Bank–Sujbert ファントム畳み込み、Cayley直交・`LAMBDA_REG_DEFAULT=400.0` SAV、音域テーパ）を `piano-strings` に実装し、`PianoEngine` 閉ループ M4 Gate 1〜3 および M4 ネイティブ/Chromium D1 ベースラインを達成。
- **M3 (2026-11-15)**: 局所 `8×`（`384 kHz`）Stage 1 アクション（ペダル `PedalBank`、`settling` O(1) クリアランス最適化、Timoshenkoシャンク、Stulov羊毛ヒステリシス、Borin K法＋1次ADAA、460スロット直交V/H片側ダンパー `piano-action`）と $\gamma_k$ 不変SIデュアルポート連成（`piano-strings`）を `PianoEngine` へ統合し M3 Gate 1〜3 を達成。
- **M2 (2026-11-15)**: 88鍵・230弦物理スケーリング／Railsback調律／D4 Voicing枠（`piano-spec`）、約25.5k〜26.2kモード連続SIMD ZOH弦バンク（`piano-strings`）、`PianoEngine` 88鍵統合、`piano-cli bench` を実装。Coupled-form ZOH高精度化、`TILE=8` タイリング、打弦ロールオフ、CI Rust `1.97.1` 固定知見を統合。
- **M1 (2026-11-15)**: `v0.1.0-ts-dwg` (`c75cf57`) へ旧v1保全後に10クレート純Rustモノレポ＋`piano-simd`＋C-ABI `piano-wasm`＋`piano-cli`＋Vanilla TS `AudioWorklet`＋CI/Pagesを確立。
- **Pre-M1 (2026-11-14)**: 初版作成（設計仕様 `ARCHITECTURE.md` / `ROADMAP.md` / `DECISIONS_AND_FALLBACKS.md` / `SWARM_WORKFLOW.md`）。
