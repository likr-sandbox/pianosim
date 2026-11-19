# ピアノ物理モデリング音源 アーキテクチャ設計仕様 (`ARCHITECTURE.md`)

## 1. 設計基本方針と開発制約

本ドキュメントは、純Rustモノレポ構成の物理モデリング・コンサートグランドピアノ音源（`pianosim`）のアーキテクチャを定義する。

1. **完全クリーンルーム設計（FTO準拠）**
   - 商用知財の自由実施領域（FTO）を確保するため、AGPL-3.0ライセンス実装のコード参照・流用を禁止し、採用する外部依存はパーミッシブライセンス（`MIT` / `Apache-2.0` / `BSD` 等）および標準ライブラリのみに限定する（詳細は第7節）。
2. **純Rustモノレポ自己完結（オフライン物理生成含む）**
   - 10クレートの Cargo workspace（うち `piano-simd`, `piano-spec`, `piano-action`, `piano-strings`, `piano-bridge`, `piano-radiator` の6クレートは `#![cfg_attr(not(test), no_std)]` 対応）で構成する。
   - Python/JAX による実機録音からの逆問題同定や録音波形サンプルは一切使用せず、全88鍵・230本の弦／ハンマー物理定数および3種のボディ構造音響データ（`.pbdy`）は **Rust製オフライン物理ソルバ（`crates/piano-body-gen`）** から手続き生成する。
3. **Web / ネイティブ共通のゼロアロケーション・リアルタイムコア**
   - 初期化（`PianoEngine::new` / `new_with_voicing` / `new_with_profile`）以降のオーディオコールバック（`render_block` / `AudioWorkletProcessor::process`）内では、**ヒープ確保ゼロ・ロック取得ゼロ・システムコールゼロ・パニック禁止**（`panic = "abort"`）を厳守する。
   - サンプル正確（sample-accurate）なイベント分割と MIDI 2.0 高分解能（16ビットベロシティ `u16`・連続ペダル開度 `0..=65535`）を標準入力境界（`SampleEvent`）とし、Web アプリケーション（`apps/web`）および検証・ベンチマーク・オフライン書き出し用ネイティブ CLI（`piano-cli`）へ同一の物理コアを提供する。
4. **Web 2系統実行モデル（`PerfProfile` 19ノブ・Case B 逐次オフライン MIDI・48 kHz 固定）**
   - GitHub Pages 等の静的ホスティングでも `COOP/COEP` ヘッダ不要で動作する **単一スレッド `AudioWorkletGlobalScope` + `wasm32-unknown-unknown`（標準 `+simd128`、`+relaxed-simd` / `+extended-const` / `fmaf` 不使用・`0 imports`）** を採用する。Web 版 `AudioContext` は **`48 kHz` 固定**（`128` サンプル量子）で駆動し、`44.1 / 96 kHz` のマルチレート検証はネイティブ CLI（`piano-cli`）およびテストで担保する。
   - 計算品質と実時間予算のトレードオフを全4ステージ横断の **19パラメータ POD `PerfProfile`（`Full` / `Balanced` / `Realtime`、`piano-spec::perf_profile`）** として統一管理し、Web 版は用途に応じた2系統実行を採る：
     - **オフライン MIDI レンダリング（`Full` プロファイル）**: 専用 **`offline-render-worker.ts`** が正準およびアップロード `.mid` ファイルを `Full` 物理品質（全約2.6万モード・`8x` アクション・10ブリッジ極・0.5秒 9ゾーン IR）で生成し、初回 `1.5 s` ＋後続 `0.75 s` チャンクで `OfflineStreamPlayer` へ逐次ストリーミング再生および `24-bit` TPDF / `32-bit` float WAV 書き出しを行う。
     - **ライブ鍵盤 / Web MIDI 試奏（`AudioWorklet`, `48 kHz`）**: 既定プロファイルは `Balanced`（`DEFAULT_LIVE_PERF_PRESET`）とし、`PerfPanel` または `?perf=full|balanced|realtime` から `Full` / `Balanced` / `Realtime` および19個別ノブを切替可能とする。① ビット完全一致ゼロスキップ、② モード次数別物理エネルギー地平線縮退（低次 `1..=8` `low_order_energy_horizon_db` vs 高次 `>8` `energy_horizon_db`、FTO #4/#7準拠）、③ サステイン・半踏時の減衰済み鍵受動共鳴降格、④ 融合 SAV 更新＋平滑休眠により実時間予算内へ収める。
   - 空間（ボディ・視点・大屋根）の IR／NUPOLS 事前計算および調律・弦スケール再計算は **専用 Prep Web Worker**（`SpacePrepContext` / `VoicingPrepContext`）へオフロードし、`BlobStreamer` が **`2,048` float（8 KB）チャンク** を `piano_space_chunk_ptr` 経由でスタンバイ側放射器バッファへ直接書き込み、`16` 分割スタンバイ遅延クリア（`clear_runtime_state_slice`）と `Transferable ArrayBuffer` 経由の 8 ms 二乗余弦クロスフェードで適用する。`VoicingBlob` は `VoicingCommitState`（`8` 鍵/ブロック × `11` ブロック）で漸進コミットし、`hammer_hardness` 単体変更（Voicing Dirty Mask）および `HallFdn` リバーブ（`reverb_decay_s`, `reverb_mix`）は Worklet 内で即座に反映してオーディオスレッド停止 `0 ms` を保証する（`appliedSpaceSeq` / `appliedVoicingSeq` で完了追跡）。
5. **1ファイル500行上限ルール**
   - 全ソースコード（`.rs`, `.ts`, `.js`, `.css`）および設計文書（`.md`）は **1ファイル500行以下** を絶対規約とし、`piano-cli check-lines` によって CI 上で機械的に検査する。
6. **代替方針の記録**
   - 採用を見送った設計選択肢（D1〜D8）および拡張境界（`NoteVoicingOffset[88]`・`PIANOBOD` バイナリ・`PianoEngine::render_block` 等）は [DECISIONS_AND_FALLBACKS.md](DECISIONS_AND_FALLBACKS.md)、クレート構成とマイルストーン実績は [ROADMAP.md](ROADMAP.md) にまとめる。

---

## 2. 既存物理モデル音源の3大聴覚課題への構造的アプローチ

| 既存モデルの課題 | 物理的発生要因 | 本アーキテクチャでの構造的解決策 |
|---|---|---|
| **1. 強打アタック初期の高域チャープ遊離感** | 単純ウェーブガイドや独立励振モデルでは、打撃パルスと響板・ボディの立ち上がりが時間・位相的に分離して聞こえる | 弦のブリッジ傾斜角連成による**縦波プリカーサー**先行到達に加え、圧縮深さ依存の有限ハンマー幅空間窓 $\Phi_w(j)$ と打弦位置比 $\beta(k)$ を導入し、さらに Stage 4 の**ゼロ追加遅延 SIMD FIR（`16/32/64` タップ／ブロック FFT 選択可）＋ Staggered MIMO NUPOLS** でボディ初期反射と単一時間軸上に統合する |
| **2. 中高音域の高速連打時のプラスチック的均一感** | メモリレスな打弦モデルおよび全鍵均一の理想境界により、トリルや連打（ribattuto）の有機的揺らぎが失われる | 2時定数 **Stulov ヒステリシス緩和フィルタ**（`TAU_FAST_S = 50 us`）の状態を離鍵・再打弦間も保持し、接触中アグラフ短弦反射（`AgraffeDelayLine`）と `SplitMix64` 決定論的ユニゾン偏差・経年変化 `Condition`（`0.0..=1.0`）を付与する |
| **3. 長時間演奏時の線の細さ・胴鳴り（身体性）不足** | 2次元平板のみの近似では、木製リム・内部空気室・鋳鉄フレームの低中域質量感や棚板の打鍵・離鍵衝撃が欠落する | クラウン初期曲率単極子・全2D面積分・3D空気室・駒ピン線積分＆ロッキング結合を備えた3種の `.pbdy`（`ConcertGrand` / `UprightStudio` / `VintageSaloon`）をオフライン生成し、Stage 4 に **3D 初期反射＋`4/8/16` ライン Hadamard `HallFdn`** を統合。打鍵・離鍵時の**棚板衝撃（keybed thump）**を第9ゾーンへ直接入力する |

---

## 3. 全体2層アーキテクチャ構成

```
[第1層A: ビルド時オフライン物理生成 (crates/piano-body-gen, crates/piano-spec)]
  88鍵弦スケーリング則 + 15×20 Ritz 直交異方性響板(クラウン単極子/リブ/リム/3D空気室/鉄骨)
  ──▶ 3種 .pbdy バイナリ (concert_grand / upright_studio / vintage_saloon)
      各 10 縮約ブリッジ極 + 極除去済み放射構造モード群 (1,287〜1,338 モード, <= 65 KB)
                                │
[第1層B: 非同期事前計算 & 逐次オフラインMIDI生成 (piano-radiator, Prep Worker, Offline Render Worker)]
  Web: 48 kHz 固定 (CLIは 44.1 / 48 / 96 kHz) + PerfProfile (Full / Balanced / Realtime, 19ノブ)
  ├──▶ Prep Worker: 9ゾーン×L/R 適応IR (<=0.5s) + SpaceBlob v2 (SPACE_BLOB_MAGIC_V2) / VoicingBlob
  │    ──▶ 2,048 float チャンク直書き (piano_space_chunk_ptr) / 11ブロック漸進コミット
  └──▶ offline-render-worker.ts: .mid ファイルを Full 品質で逐次レンダリング (1.5s + 0.75s) & WAV出力
                                │
                                ▼
[第2層: リアルタイムDSPコア (Zero Heap Alloc / No Locks / AudioWorklet 48 kHz ライブ試奏)]
  MIDI 1.0/2.0 イベント (SampleEvent: 16bit ベロシティ, 連続 CC64/66/67)
   ├──▶ Stage 1 (piano-action): 2/4/8x Timoshenko + AgraffeDelayLine + Stulov + Poly/LUT ADAA + 464-SoA間引きダンパー + Keybed
   ├──▶ Stage 2 (piano-strings): 最大~26,184 モード SoA (ZOH + 動的Φ_w/β + 3機構T60 + 疎/3x6ファントム + 次数別地平線 + 融合SAV)
   ├──▶ Stage 3 (piano-bridge): 4..=10極縮約ブリッジ 8x8 WDF 双方向連成 + 任意Hertz非線形 + 64モード DuplexBank + 3ペダル創発
   └──▶ Stage 4 (piano-radiator): 9ゾーン MIMO (16/32/64 FIR or BlockFFT + ステレオ集約可変NUPOLS) + 4/8/16ライン HallFdn + C^2 Limiter
```

---

## 4. 第1層：オフライン物理生成と非同期IR／ボイシング合成

### 4.1 弦スケーリング・調律・3種ボディ固有値解析（`piano-spec`, `piano-body-gen`）
1. **88鍵・230本弦スケーリングと調律拡張**:
   - 低音単弦8鍵＋複弦18鍵＋中高音トリコード62鍵＝計230本に対し、$L(k), \mu(k), T_0(k), ESK^2(k)$ とインハーモニシティ $B(k) \approx 0.00036 \to 0.045$ を定義する。
   - 部分音周波数は $f_j = jF_0\sqrt{1+Bj^2}$ を直接評価し、自動 Railsback ストレッチ（スケーリング係数可変）、`C5+` ユニゾンデチューンテーパ（C8 で隣弦うなり `< 0.9 Hz` に抑制）、コンサートピッチ `A4`（`392.0..466.2 Hz`、既定 `440.0 Hz`）、A4基準の古典4音律（`Equal`, `WerckmeisterIII`, `KirnbergerIII`, `Vallotti`）、および `SplitMix64` による決定論的経年変化 `Condition`（`0.0..=1.0`）を提供する。
   - 新品状態（`condition = 0.0`）でも同相ロックを防ぐため、基底定数として微小ユニゾン偏差（`0.2–0.5 cents`）および鍵・弦ごとの V/H ブリッジピン角（`bridge_pin_angle_rad`）を組み込む。
2. **3種 `.pbdy` ボディ生成（`piano-body-gen`）**:
   - `15×20` 基底の Rayleigh-Ritz 直交異方性響板ソルバ（sum-factorization ＋ Householder QL）により、クラウン初期曲率の面内膜応力（単極子体積変位）と全2D面積分 $\iint w_r\,dA$、3次元空気室の Helmholtz 結合、駒ピン列沿いの局所線積分＋ロッキング（ねじれ角）結合、リブ・木製リム境界・鋳鉄フレーム・大屋根連成を解き、~16.5 kHz までの統計的 Rayleigh モードで補完する。
   - 3種類の物理ボディ（`concert_grand_v1.pbdy`: **`1,338`** モード・`60,842 B`、`upright_studio_v1.pbdy`: **`1,287`** モード・`58,547 B`、`vintage_saloon_v1.pbdy`: **`1,297`** モード・`58,997 B`）を `PIANOBOD`（`BodyPreset`、IEEE CRC-32 付き、各 `< 65 KB`）として `piano-radiator/assets/` に同梱する。
3. **10縮約ブリッジ極・文献モビリティ校正・厳密極除去（Pole Cancellation）**:
   - 駆動点モビリティが実測文献値域（$Z_0 \cdot \operatorname{Re}\{Y_{\text{bridge}}\} \in [10^{-4}, 10^{-3}]\text{ m/(N}\cdot\text{s)}$）に一致するよう校正された **10個の縮約ブリッジ極**（`PRESET_REDUCED_POLES = 10`）および `y_inf` を Stage 3 用に抽出し、放射スペクトル傾斜（約 `-3〜-4 dB/oct`）を保ちつつ Stage 4 放射モード留数から台形則離散逆フィルタで10極寄与を差し引く（`verify_pole_cancellation <= 1e-4 dB`、符号不変パリティ検証済み）。

### 4.2 適応 IR 合成・`SpaceBlob` v2・Prep / Offline Worker（`piano-radiator`, `piano-strings`）
- `piano-radiator::ir_synth` はホスト実レート（Web 48 kHz 固定、CLI 44.1/48/96 kHz）、3ボディ、`4` 聴取視点（`Player`, `CloseStereo`, `ConcertHall`, `BinauralHrtf`：Woodworth ITD + Brown-Duda ILD + 耳介 cues）、`3` 大屋根状態（`Closed`, `Half`, `Open`）および `PerfProfile` の `ir_duration_secs`（`0.05..0.50 s`）・`spatial_tail_split_tap`・`fir_head_taps`・`fir_block_fft` に応じて 9ゾーン×L/R（18経路）IR と `3.8 kHz` HPF ベルベット拡散テールを合成し、**`SpaceBlob` v2（`SPACE_BLOB_MAGIC_V2`）** として事前 FFT スペクトルを梱包する。
- Web 実行時は Prep Web Worker が `SpacePrepContext` / `VoicingPrepContext` を非同期生成し、`BlobStreamer` が `2,048` float チャンクで `piano_space_chunk_ptr` へ直接書き込んで `16` 分割スタンバイクリア＋8 ms クロスフェード（空間）および `11` ブロック `VoicingCommitState`（調律）で無停止適用する。並行して、`offline-render-worker.ts` は `.mid` ファイルの `Full` 品質逐次レンダリングと WAV 書き出しを担う。

---

## 5. 第2層：4ステージ・リアルタイムDSPパイプライン

### 5.1 Stage 1：可変マルチレート・多項式 ADAA・`AgraffeDelayLine`・片側ダンパー（`crates/piano-action`）
- **可変 `2/4/8x` 局所アクション ＋ `AgraffeDelayLine` ＋ 軟化 Stulov フェルト**: 接触中の鍵のみ `action_oversample`（`2/4/8x`、既定 `Full = 8x`, `Balanced/Realtime = 4x`）で 1次 Timoshenko 柔軟シャンク（`shank_flex_enabled` で集中定数切替可）・Stulov 2時定数羊毛フェルト（`TAU_FAST_S = 50 us`, `EPS_FAST_BASE = 0.34`, 低中域 $p \in [2.7, 3.2]$）を解き、接触ループ内の短弦アグラフ反射遅延線 **`AgraffeDelayLine`**（`agraffe_delay.rs`）によって多峰性打弦力波形 $F_h(t)$ を創発させる。間引き FIR は `prev_sub_forces` を保持して離脱クリックを防ぐ。
- **共有 $p$-グリッド LUT ＆ ゼロ LUT 多項式 ADAA（`PolyPowerApprox`）**: 従来の共有 24 点 $p$-グリッド区分エルミート `AdaaPowerLut`（`PGridLuts`, 384 KB）に加え、`polynomial_adaa = true` 時は `piano-simd::poly_adaa::PolyPowerApprox`（6次 Bernstein-monomial 近似＋Newton K-method `k_method_poly.rs`）でハードウェア gather 非搭載 SIMD 環境のキャッシュ圧迫を回避する。
- **片側ヒステリシス接触ダンパー（`464`-SoA＋制御レート間引き）＆ `KeybedBank`**: `0..70` 鍵×2偏波の `460` 独立スロットを `464`（`116 x 4`）64B整列 SoA（`DamperPadSoa`, `damper_pad_simd.rs`）で保持する。`PedalBank`（`pedals_clearance.rs`）が `damper_control_decim`（`1|2|4|8|16`）周期の線形クリアランス／剥離補間を行い、本番の活性弦経路では `step_damper_pair`（`control_decim > 1` 時は `step_damper_sample_decim`）で制御点のヒステリシス状態をキャッシュしつつ毎サンプルの片側幾何クリアランス接触（FTO #3 準拠）を評価する（`damper_pad_simd.rs` は一括 SIMD `step_damper_batch_simd` も提供する）。打鍵・離鍵衝撃は `KeybedBank` として Zone 8 へ直接入力する。

### 5.2 Stage 2：最大2.6万モード ZOH＋動的空間窓＋次数別地平線縮退＋融合 SAV（`crates/piano-strings`）
- **64B整列 SoA ＆ `PerfProfile` モード制御**: 最大 **約26,184 モード**（230本×2偏波＋縦モード）に対し、ビット一致ゼロスキップ、水平偏波高域カット（`horiz_max_hz`）、初期化時高次死蔵モード刈込（`dead_mode_prune_db`）、**モード次数別エネルギー地平線**（低次 `1..=8` `low_order_energy_horizon_db` vs 高次 `>8` `energy_horizon_db`、自弦内完結で Sostenuto 余韻と計算削減を両立）、サステイン・半踏時の減衰済み鍵受動共鳴制限（`passive_res_max_hz`）、および `sav_dormancy_db` / `sav_max_key_idx` 休眠を適用する。
- **動的空間窓・3機構 $T_{60}$・高音 WDF 再接続**: ハンマー圧縮深さに応じた実効接触幅 $\Phi_w(j)$（`contact_phi.rs`）と `damper_phi`（静的空間幅＋`2.6 kHz` 粘弾性ロールオフ＋位相不変半踏テール）で励振・制音を整形する。`damping_law.rs` は `THERMOELASTIC_COEFF`（`2.05e-9`）を緩めて高域減衰の主役をブリッジ放射損失へ移し、高音 `6 kHz` 前置帯域は WDF 境界へ再接続する。
- **ドリフト制御 SAV ＆ 疎／`3×6` ファントム連成**: Risse, Hélie & Bilbao（DAFx 2025）型のパワーバランス保存 SAV で `ff` 連打ドリフトを `0.5 cents` 未満に抑え、低音 `k = 0..16` の `3×6` ファントムと `F2..G4` 縦モード連成を再配分しつつ、`bass_phantom_sparse = true` 時は主要共振ペアのみの `2×2` 疎カーネル（`phantom_sparse.rs`）へ切替可能とする。

### 5.3 Stage 3：可変極縮約ブリッジ WDF 連成＆デュープレックス（`crates/piano-bridge`）
- 全230弦終端と Stage 2 プレフィックス剥離（`MAX_PEEL_SLOTS = 24`）を、**`active_bridge_poles`（`4..=10`、不活性高域極を実効 `y_inf` へ畳み込み `admittance_smooth.rs` で滑らか射影）** の縮約響板 `AdmittanceSsm` ＋ `8×8` WDF ジャンクション＋任意 SAV Hertz 接触（`hertz_nonlinear` で線形コンプライアンス切替可）＋調律連動 64 モード `DuplexBank`（Zones 2..8、`< 1e-20` フラッシュ付）へ双方向結合し、二段階減衰・無音押鍵・3ペダル創発および診断用 `BridgeBank::set_fixed_end` 切替を提供する。

### 5.4 Stage 4：可変タップ FIR / ブロック FFT・ステレオ集約 NUPOLS・`HallFdn`（`crates/piano-radiator`）
- **9ゾーン MIMO 放射**: 8ブリッジゾーンと Zone 8（棚板）を、`fir_head_taps`（`16/32/64`、不足分はサブティア FFT 補完）または `fir_block_fft` 有効時の **`BlockFftHead`**（32サンプル・2パーティション FFT、`block_fft_head.rs`）＋ 4x 千鳥 NUPOLS で畳み込む。`spatial_tail_split_tap` 指定時は分割タップ以降の遅いパーティションを因果 RMS 整合で L/R 2ch へ統合（`nupols_collapse.rs`）し、逆 FFT コストを大幅削減する。
- **空間残響 `HallFdn` ＆ $C^2$ `StereoLimiter`**: 後段に 6 タップ 3D ER＋`hall_fdn_lines`（`4/8/16`）直交 Hadamard `HallFdn`（`< 400 Hz` IACC シェルビング、既定 `decay = 1.6 s`, `mix = 0.18`）と 0遅延 $C^2$ `StereoLimiter`（`LINEAR_THRESHOLD = 0.91`、`|\text{out}| \le 1.0`）を通す。

---

## 6. SIMD 抽象化・Web 実行境界・ネイティブ CLI

1. **ポータブル SIMD (`piano-simd`)**: `F32x4` が `wasm32`（標準 `+simd128`、非FMA対で `fmaf` 回避）・`x86_64`・`aarch64`・スカラを統一し、`AlignedBuf`（64B整列）、`Radix2Fft`、`RealFft`、`SplitComplexBuf`、エルミート `AdaaPowerLut`、および `PolyPowerApprox` を提供する。
2. **Web Wasm + Worker + Studio UI (`piano-wasm`, `apps/web`)**:
   - `vite.config.ts` の `?worklet` / `?blobworker` プラグインが自己完結 Blob スクリプトを生成し、単一 `.wasm`（`0 imports`, `+simd128`）を `AudioWorklet`・Prep Web Worker・`offline-render-worker.ts` で共有する。C-ABI（`exports.rs`, `perf_exports.rs`）はゼロスパイク転送（`piano_space_chunk_ptr`, `piano_stage_body_space_chunk`, `piano_commit_body_space`, `piano_is_voicing_applying`）と 19 ノブ `PerfProfile`（`knob_id 0..=18`）を公開する。
   - **ライブ演奏・録音系**: `WebMidiManager`（14/16bit ベロシティ・連続 CC64/66/67）、`PointerGlissando`（Y座標ベロシティ）、`TouchCurve`（`Light`/`Natural`/`Heavy`）、`PerfPanel`（3プリセット・19ノブ・`?perf=`）、`WorkletRecorderPool`（事前確保16スロットピンポン転送＋24-bit TPDF / 32-bit float WAV 書き出し）および全ブロック走査テレメトリ HUD（`window.__PIANO_DIAG__`）を備える。
   - **オフライン MIDI 再生系（Case B）**: `apps/web/public/demos/*.mid` の正準 Format-0 `.mid` 6曲およびユーザー投入 `.mid` を `smf-parser.ts` で解析し、`OfflineRenderer` ＋ `OfflineWorkerClient` ＋ `offline-render-worker.ts` ＋ `OfflineStreamPlayer` により **`Full` プロファイル** で逐次生成（初回 `1.5 s` ＋ `0.75 s` チャンク）・シーク・テンポ変更・WAV ダウンロードを提供する。
3. **ネイティブ境界 (`piano-engine`, `piano-cli`)**:
   - `PianoEngine` は `new_with_profile`、`set_perf_profile`、`set_perf_profile_runtime`、`begin_voicing_commit` / `step_voicing_commit` および `render_block` を提供する。
   - `piano-cli` は `bench`（64サンプル＠48kHz 実時間計測＋`bench_prep.rs` Half-pedal/Prep スイープ分離）、`scenarios`（3大課題＋統合ゲート）、`render-wav`、`render-demos`、`diag-radiator` / `diag-bridge` / `diag-hammer` / `diag-buzz`（ステージ別音響診断）、`check-lines`、`check-wasm-simd` の10サブコマンドと、`--perf-profile <full|balanced|realtime>` ＋ **19個の上書きフラグ（`CliProfileArgs`）** を備える。

---

## 7. 知財クリーンルーム（FTO）設計不変条件

| No. | 対象知財・リスク領域 | 禁止する設計・実装パターン | 本プロジェクトのクリーンルーム設計（Design-Around） |
|---|---|---|---|
| **1** | **OSSコピーレフト汚染** | `OpenPiano` 等の AGPL-3.0 コード参照・流用 | パーミッシブライセンスおよび純Rust自前実装のみ許可（`check-lines` で検査） |
| **2** | **Modartt基本特許** (`US7,915,515B2`) | RBF/SVM/NN/Taylor 補間によるプリシンセシス、IR事前畳み込み励振テーブル | 解析的分散式 $f_j = jF_0\sqrt{1+Bj^2}$ の直接評価と Stage 4 独立ランタイム畳み込み |
| **3** | **ヤマハ・ダンパー特許** (`US8,115,092B2` / `JP5614108B2`) | 押鍵・ペダルに応じたダンパー粘性係数テーブルの動的書換 | 幾何学的クリアランス境界入力による片側非線形ヒステリシス接触パッド（`DamperPadBank`：`damper_control_decim` 下でも毎サンプル `delta = max(\|y_s\| - y_c, 0)` を維持） |
| **4** | **ヤマハ・VRM共鳴特許** (`JP6176132B2`) | 主音源と並列に付加する「共鳴音生成エフェクト装置」構成 | 全230弦と開放弦を単一の縮約ブリッジ状態空間内で対等に双方向連成 |
| **5** | **カワイ・共鳴定位特許** (`JP6805060B2` / `US10,102,837B1`) | 鍵別共鳴を少数グループへ集約し後段の遅延回路で定位付与 | 弦の実ブリッジ接続座標に基づく8ゾーン空間重み射影（物理的空間位相差） |
| **6** | **ヤマハ・棚板衝突音特許** (`JP7306402B2` / `US11,961,499B2`) | 弦打撃音と棚板衝突音の2系統個別減衰速度制御 | 打鍵・離鍵衝撃力を第9ゾーン（Zone 8）外力インパルスとして放射器へ直接入力 |
| **7** | **カシオ・倍音ダンプ判定特許** (`JP7400925B2`) | 押鍵と「倍音関係にある他鍵がダンプ状態か」の `if-then` 分岐 | 倍音関係判定ロジックを一切持たず、全230弦へ共通ブリッジ力を一様作用（次数別地平線や受動上限 `passive_res_max_hz` も自弦内のみで完結） |
