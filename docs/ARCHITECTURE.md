# ピアノ物理モデリング音源 アーキテクチャ設計仕様 (`ARCHITECTURE.md`)

## 1. 設計基本方針と開発制約

本ドキュメントは、純Rustモノレポ構成のリアルタイム物理モデリング・コンサートグランドピアノ音源（`pianosim`）のアーキテクチャを定義する。

1. **完全クリーンルーム設計（FTO準拠）**
   - 商用知財の自由実施領域（FTO）を確保するため、AGPL-3.0ライセンス実装のコード参照・流用を禁止し、採用する外部依存はパーミッシブライセンス（`MIT` / `Apache-2.0` / `BSD` 等）および標準ライブラリのみに限定する（詳細は第7節）。
2. **純Rustモノレポ自己完結（オフライン物理生成含む）**
   - 10クレートの Cargo workspace（うち `piano-simd`, `piano-spec`, `piano-action`, `piano-strings`, `piano-bridge`, `piano-radiator` の6クレートは `#![cfg_attr(not(test), no_std)]` 対応）で構成する。
   - Python/JAX による実機録音からの逆問題同定や録音波形サンプルは一切使用せず、全88鍵・230本の弦／ハンマー物理定数および3種のボディ構造音響データ（`.pbdy`）は **Rust製オフライン物理ソルバ（`crates/piano-body-gen`）** から手続き生成する。
3. **Web / ネイティブ共通のゼロアロケーション・リアルタイムコア**
   - 初期化（`PianoEngine::new` / `new_with_voicing`）以降のオーディオコールバック（`render_block` / `AudioWorkletProcessor::process`）内では、**ヒープ確保ゼロ・ロック取得ゼロ・システムコールゼロ・パニック禁止**（`panic = "abort"`）を厳守する。
   - サンプル正確（sample-accurate）なイベント分割と MIDI 2.0 高分解能（16ビットベロシティ `u16`・連続ペダル開度 `0..=65535`）を標準入力境界（`SampleEvent`）とし、Web アプリケーション（`apps/web`）および検証・ベンチマーク・オフライン書き出し用ネイティブ CLI（`piano-cli`）へ同一の物理コアを提供する。
4. **Web リアルタイム実行モデル（単一 `+simd128` バイナリ・Prep Worker 分離・48 kHz 固定）**
   - GitHub Pages 等の静的ホスティングでも `COOP/COEP` ヘッダ不要で動作する **単一スレッド `AudioWorkletGlobalScope` + `wasm32-unknown-unknown`（標準 `+simd128`、`+relaxed-simd` / `+extended-const` / `fmaf` 不使用・`0 imports`）** を採用する。
   - Web 版 `AudioContext` は **`48 kHz` 固定**（`128` サンプル量子）で生成し、`44.1 / 96 kHz` のマルチレート検証はネイティブ CLI（`piano-cli`）およびテストで担保する。
   - 全88鍵・230本・約2.6万モードの物理定義を維持したまま、① ビット完全一致のゼロスキップ、② 物理エネルギー地平線（`-115〜-120 dB`）による有効モード範囲縮退（FTO #4/#7準拠）、③ サステイン踏下中の減衰済み鍵の受動共鳴降格、④ 融合 SAV 更新＋`-55 dB` 平滑休眠により実時間予算内へ収める。
   - 空間（ボディ・視点・大屋根）の IR／NUPOLS 事前計算および調律・弦スケールテーブル再計算は **専用 Prep Web Worker**（`SpacePrepContext` / `VoicingPrepContext`）へオフロードし、`BlobStreamer` の共有ステージング直列化と `Transferable ArrayBuffer` 経由の 8 ms 二乗余弦クロスフェードで適用する（シーケンス番号 `appliedSpaceSeq` / `appliedVoicingSeq` で非同期適用を確定追跡）。一方、`hammer_hardness` 単体変更（Voicing Dirty Mask）および `HallFdn` リバーブ（`reverb_decay_s`, `reverb_mix`）は Worklet 内で即座に反映し、オーディオスレッド停止 `0 ms` を保証する。
5. **1ファイル500行上限ルール**
   - 全ソースコード（`.rs`, `.ts`, `.js`, `.css`）および設計文書（`.md`）は **1ファイル500行以下** を絶対規約とし、`piano-cli check-lines` によって CI 上で機械的に検査する。
6. **代替方針の記録**
   - 採用を見送った設計選択肢（D1〜D8）および拡張境界（`NoteVoicingOffset[88]`・`PIANOBOD` バイナリ・`PianoEngine::render_block` 等）は [DECISIONS_AND_FALLBACKS.md](DECISIONS_AND_FALLBACKS.md)、クレート構成とマイルストーン実績は [ROADMAP.md](ROADMAP.md) にまとめる。

---

## 2. 既存物理モデル音源の3大聴覚課題への構造的アプローチ

| 既存モデルの課題 | 物理的発生要因 | 本アーキテクチャでの構造的解決策 |
|---|---|---|
| **1. 強打アタック初期の高域チャープ遊離感** | 単純ウェーブガイドや独立励振モデルでは、打撃パルスと響板・ボディの立ち上がりが時間・位相的に分離して聞こえる | 弦のブリッジ傾斜角連成による**縦波プリカーサー**先行到達に加え、有限ハンマー幅空間窓 $\Phi_w(j)$ と打弦位置比 $\beta(k)$ を導入し、さらに Stage 4 の**ゼロ追加遅延 64 タップ SIMD FIR＋Staggered MIMO NUPOLS** でボディ初期反射と単一時間軸上に統合する |
| **2. 中高音域の高速連打時のプラスチック的均一感** | メモリレスな打弦モデルおよび全鍵均一の理想境界により、トリルや連打（ribattuto）の有機的揺らぎが失われる | 2時定数 **Stulov ヒステリシス緩和フィルタ**の状態を離鍵・再打弦間も保持し羊毛フェルトの圧縮記憶効果を再現。さらに `SplitMix64` による決定論的ユニゾン偏差と経年変化 `Condition`（`0.0..=1.0`）を付与する |
| **3. 長時間演奏時の線の細さ・胴鳴り（身体性）不足** | 2次元平板のみの近似では、木製リム・内部空気室・鋳鉄フレームの低中域質量感や棚板の打鍵・離鍵衝撃が欠落する | **直交異方性リブ付き響板＋木製リム＋3次元空気室＋鋳鉄フレーム＋高域拡散テール**の連成構造モードを3種の `.pbdy`（`ConcertGrand` / `UprightStudio` / `VintageSaloon`）としてオフライン生成し、Stage 4 に **3D 初期反射＋8×8 Hadamard `HallFdn`** を統合。打鍵・離鍵時の**棚板衝撃（keybed thump）**を第9ゾーンへ直接入力する |

---

## 3. 全体2層アーキテクチャ構成

```
[第1層A: ビルド時オフライン物理生成 (crates/piano-body-gen, crates/piano-spec)]
  88鍵弦スケーリング則 + 15×20 Ritz 直交異方性響板/リブ/リム/3D空気室/鉄骨固有値ソルバ
  ──▶ 3種 .pbdy バイナリ (concert_grand / upright_studio / vintage_saloon)
      各 10 縮約ブリッジ極 + 極除去済み放射構造モード群 (1,305〜1,372 モード, <= 500 KB)
                                │
[第1層B: 初期化・空間/調律変更時 (crates/piano-radiator::ir_synth, Prep Web Worker)]
  Web: 48 kHz 固定 (CLIは 44.1 / 48 / 96 kHz) + 視点/大屋根/ボディ + 調律/音律/Condition
  ──▶ 9ゾーン×L/R (18経路) 0.5秒 IR (3.8 kHz HPF 統計ベルベットテール付) + NUPOLS 事前FFT
  ──▶ SpaceBlob / VoicingBlob を Transferable ArrayBuffer でゼロコピー転送 → 8 ms クロスフェード
                                │
                                ▼
[第2層: リアルタイムDSPコア (Zero Heap Alloc / No Locks / AudioWorklet 48 kHz)]
  MIDI 1.0/2.0 イベント (SampleEvent: 16bit ベロシティ, 連続 CC64/66/67)
   ├──▶ Stage 1 (piano-action): 接触中のみ 8x (384 kHz) Timoshenko + Stulov + K-method/24点p-grid ADAA + 片側ダンパー + 棚板打鍵/離鍵衝撃
   ├──▶ Stage 2 (piano-strings): ~26,184 モード SoA (ZOH + Φ_w/β + 3機構 T60 + 6kHz WDF 前置 + 3x6 低音ファントム + 融合 SAV)
   ├──▶ Stage 3 (piano-bridge): 10モード縮約ブリッジ 8x8 WDF 双方向連成 + SAV Hertz 接触 + 64モード DuplexBank + 3ペダル創発
   └──▶ Stage 4 (piano-radiator): SI校正 9ゾーン MIMO (64-tap 0遅延 FIR + Staggered NUPOLS) + HallFdn + C^2 StereoLimiter ─▶ Stereo Out
```

---

## 4. 第1層：オフライン物理生成と非同期IR／ボイシング合成

### 4.1 弦スケーリング・調律・3種ボディ固有値解析（`piano-spec`, `piano-body-gen`）
1. **88鍵・230本弦スケーリングと調律拡張**:
   - 低音単弦8鍵＋複弦18鍵＋中高音トリコード62鍵＝計230本に対し、$L(k), \mu(k), T_0(k), ESK^2(k)$ とインハーモニシティ $B(k) \approx 0.00036 \to 0.045$ を定義する。
   - 部分音周波数は $f_j = jF_0\sqrt{1+Bj^2}$ を直接評価し、自動 Railsback ストレッチ（スケーリング係数可変）、`C5+` ユニゾンデチューンテーパ（C8 で隣弦うなり `< 0.9 Hz` に抑制）、コンサートピッチ `A4`（`392.0..466.2 Hz`、既定 `440.0 Hz`）、A4基準の古典4音律（`Equal`, `WerckmeisterIII`, `KirnbergerIII`, `Vallotti`）、および `SplitMix64` による決定論的経年変化 `Condition`（`0.0..=1.0`、`0.0` 時は `NoteVoicingOffset[88]` とビット完全一致のゼロ偏差）を提供する。
2. **3種 `.pbdy` ボディ生成（`piano-body-gen`）**:
   - `15×20` 基底の Rayleigh-Ritz 直交異方性響板ソルバ（sum-factorization ＋ Householder QL）により、リブ・駒不均質性・木製リム境界・3次元空気キャビティ・鋳鉄フレーム・大屋根連成を解き、~16.5 kHz までの統計的 Rayleigh モードで補完する。
   - 3種類の物理ボディ（`concert_grand_v1.pbdy`: `1,372` モード、`upright_studio_v1.pbdy`: `1,305` モード、`vintage_saloon_v1.pbdy`: `1,342` モード）を `PIANOBOD`（`BodyPreset`、IEEE CRC-32 付き、各 `< 500 KB`）として `piano-radiator/assets/` に同梱する。
3. **10縮約ブリッジ極と厳密極除去（Pole Cancellation）**:
   - 駆動点モビリティを支配する **10個の縮約ブリッジ極**（`PRESET_REDUCED_POLES = 10`）を Stage 3 用に抽出し、Stage 4 放射モード留数からは台形則離散逆フィルタによって当該10極の寄与を差し引く（`verify_pole_cancellation <= 1e-4 dB`、CPU 間の `libm` IFUNC 差分に対する符号不変パリティ検証済み）。

### 4.2 適応 IR 合成と Prep Web Worker 非同期転送（`piano-radiator`, `piano-strings`）
- `piano-radiator::ir_synth` はホスト実レート（Web 48 kHz 固定、CLI 44.1/48/96 kHz）、3ボディ、`4` 聴取視点（`Player`, `CloseStereo`, `ConcertHall`, `BinauralHrtf`：Woodworth ITD + Brown-Duda ILD + 耳介 cues）、`3` 大屋根状態（`Closed`, `Half`, `Open`）から **9ゾーン×L/R（18経路、0.5秒 IR）** を合成する。高域拡散密度を高めるため、`3.8 kHz` HPF 処理した確率的ベルベット拡散テールを合成 IR へ統合する。
- Web 実行時は Prep Web Worker が `SpacePrepContext`（18経路 IR＋NUPOLS 事前FFT `SpaceBlob`）および `VoicingPrepContext`（全230弦の再調律テーブル `VoicingBlob`）を生成し、`BlobStreamer` が `WasmContext::staging`（`8,192` float チャンク分割）へ流し込んで 8 ms クロスフェードまたはインプレース振幅整合付きスワップを実行する。

---

## 5. 第2層：4ステージ・リアルタイムDSPパイプライン

### 5.1 Stage 1：局所 8x アクション・共有 $p$-グリッド ADAA・片側ダンパー（`crates/piano-action`）
- **局所8倍マルチレート（384 kHz＠48 kHz）＋ 16タップ間引き FIR**: 接触中の鍵のみ `8x` で 1次 Timoshenko 柔軟シャンク・Stulov 2時定数羊毛フェルト（離鍵後も圧縮履歴を保持）・Borin K-method を解き、間引き FIR の入力履歴（`prev_sub_forces`）を保持して離脱時のクリックを防止する。
- **共有 24 点 $p$-グリッド Hermite `AdaaPowerLut`（`PGridLuts`, 384 KB）**: ハンマー硬度に応じた冪指数 $p \in [2.10, 3.50]$ を88鍵で共有テーブル化し、1次区分エルミート ADAA で折り返し歪みを抑制する。
- **片側ヒステリシス接触ダンパー＆棚板衝撃（Keybed）**: `0..70` 鍵×2偏波の `460` 独立スロットにおいて、鍵位置・連続ペダル（CC64/66）から算出される幾何学的クリアランス入力の片側 $C^2$ 接触（`smooth_felt_engage`）を解く。さらに打鍵底突き・離鍵戻りの双方から生じる構造体インパルスを `KeybedBank` として第9ゾーン（Zone 8）へ直接入力する。

### 5.2 Stage 2：全2.6万モード ZOH＋エネルギー縮退＋融合 SAV（`crates/piano-strings`）
- **64バイト整列 SoA ＆ 実時間計算量制御**: 230本×2偏波＋低〜中音域縦モードの計 **約26,184 モード**（4レーン SIMD 境界パディング済み）を保持し、ビット完全一致ゼロスキップ、物理エネルギー地平線縮退（`-115〜-120 dB`）、サステイン踏下中減衰済み鍵の受動共鳴降格、および `-55 dB` 平滑 SAV 休眠を適用する。
- **空間窓・物理減衰・高音 WDF 再接続**: 有限ハンマー幅空間窓 $\Phi_w(j)$ と打弦位置比 $\beta(k)$ により高次打撃倍音を物理的に整形し、`damping_law.rs`（空気粘性・熱弾性・低音巻線界面摩擦の3機構）で $T_{60}(k, j)$ を決定する。高音域（`6 kHz` 前置帯域）は WDF 境界と再接続して高音減衰の整合性を保つ。
- **ドリフト制御 SAV ＆ `3×6` 低音ファントム**: Risse, Hélie & Bilbao（DAFx 2025）に基づくパワーバランス保存型ドリフト制御 SAV により強打連打でもピッチドリフトを `0.5 cents` 未満に抑える。低音鍵 `k = 0..16` には専用 `3×6`（`12` 横×`24` 縦）Bank-Sujbert SIMD ファントム畳み込みカーネルを適用し、偶数次（$2f_m$）・奇数次（$f_n \pm f_m$）ファントム倍音および縦波プリカーサーを創発させる。

### 5.3 Stage 3：10モード縮約ブリッジ WDF 連成＆デュープレックス（`crates/piano-bridge`）
- 全230弦の終端を、Stage 2 のプレフィックス剥離（`MAX_PEEL_SLOTS = 24`）と連携しながら、**10モード縮約響板アドミタンス状態空間（`AdmittanceSsm`）＋ `8×8` WDF ジャンクション＋SAV 保守型 Hertz 接触剛性＋調律連動 64 モード `DuplexBank`（Zones 2..8、`< 1e-20` 非正規化数フラッシュ付）** へ分岐なしで双方向結合する。
- これにより二段階減衰、無音押鍵・サステイン共鳴、連続 Una Corda（CC67 横シフトによる非打撃弦の受動吸収）、Sostenuto（CC66 選択保持）が条件分岐なしで自然創発する。

### 5.4 Stage 4：9ゾーンゼロ遅延 NUPOLS・`HallFdn`・統一リミッタ（`crates/piano-radiator`）
- **SI校正 9ゾーン MIMO 放射**: 8ブリッジゾーン（弦＋デュープレックス速度）と Zone 8（棚板衝撃力）を、**64タップ直接 `F32x4` FIR（追加遅延 0 サンプル）＋ 4x 千鳥分割 NUPOLS FFT（ゼロ入力スロットスキップ付）** でステレオ2chへ放射する。
- **空間残響 `HallFdn` ＆ $C^2$ `StereoLimiter`**: 後段に 6 タップ 3D 初期反射＋`8×8` 直交 Hadamard 帰還遅延網 `HallFdn`（`< 400 Hz` バイノーラル IACC シェルビング・拡散フィールド EQ 付、可変 `decay: 0..3.2 s`, `mix: 0..1.0`、既定 `1.6 s` / `0.18`）と、0遅延・ステレオ連動の $C^2$ ソフトシーリング安全リミッタ `StereoLimiter`（`LINEAR_THRESHOLD = 0.91`、`|\text{out}| \le 1.0`）を通す（Web/CLI 共通、任意ゲイン乗算なし）。

---

## 6. SIMD 抽象化・Web 実行境界・ネイティブ CLI

1. **ポータブル SIMD (`piano-simd`)**: `F32x4` が `wasm32`（標準 `+simd128`、非FMA命令対で `fmaf` ソフトウェアリブコールを回避）・`x86_64`・`aarch64`・スカラを統一し、`AlignedBuf`（64B整列）、`Radix2Fft`、`RealFft`、`SplitComplexBuf`、2次補間・エルミート `AdaaPowerLut` を提供する。
2. **Web Wasm + Worker + Studio UI (`piano-wasm`, `apps/web`)**:
   - `vite.config.ts` の `?worklet` / `?blobworker` プラグインが `esbuild` で自己完結 Blob スクリプトを生成し、単一 `.wasm`（`0 imports`, `+simd128`）を `AudioWorklet` と Prep Web Worker で共有する。
   - 入力系は `WebMidiManager`（14/16bit ベロシティ・連続 CC64/66/67）、`MidiPlayer`（60 ms 先読み＋`EventQueue` キャリーオーバ）、`PointerGlissando`（マルチタッチ・Y座標ベロシティスケーリング）、`TouchCurve`（`Light`/`Natural`/`Heavy`）を統合する。
   - 録音系は `WorkletRecorderPool`（事前確保済み 16 スロットの `Transferable` ピンポン転送によるオーディオ線内ゼロアロケーション収録）＋ `wav-recorder.ts`（24-bit TPDF ディザ PCM / 32-bit IEEE float WAV 書き出し）を提供する。
   - テレメトリは全ブロック走査の `maxCallbackUs`・p50・p99・D1 超過回数・Peak Hold / RMS（`dBFS` / 推定 `dB SPL`）・`appliedSpaceSeq` / `appliedVoicingSeq` を `window.__PIANO_DIAG__` と HUD へ配信する。
3. **ネイティブ境界 (`piano-engine`, `piano-cli`)**:
   - `PianoEngine::render_block(&mut self, events: &[SampleEvent], out_l: &mut [f32], out_r: &mut [f32])` を単一リアルタイム API とし、`piano-cli` から `bench`（64サンプル＠48kHz 実時間ベンチ）、`scenarios`（3大聴覚課題＋Wave 4 統合ゲート検証）、`render-wav`（`--a4`, `--format pcm16|pcm24|f32` オフライン書き出し）、`check-lines`（500行上限・クリーンルーム・E2E `waitForTimeout` 禁止検査）、`check-wasm-simd`（`0 imports` / `0xfd` / `relaxed-simd` 検査）を提供する。

---

## 7. 知財クリーンルーム（FTO）設計不変条件

| No. | 対象知財・リスク領域 | 禁止する設計・実装パターン | 本プロジェクトのクリーンルーム設計（Design-Around） |
|---|---|---|---|
| **1** | **OSSコピーレフト汚染** | `OpenPiano` 等の AGPL-3.0 コード参照・流用 | パーミッシブライセンスおよび純Rust自前実装のみ許可（`check-lines` で検査） |
| **2** | **Modartt基本特許** (`US7,915,515B2`) | RBF/SVM/NN/Taylor 補間によるプリシンセシス、IR事前畳み込み励振テーブル | 解析的分散式 $f_j = jF_0\sqrt{1+Bj^2}$ の直接評価と Stage 4 独立ランタイム畳み込み |
| **3** | **ヤマハ・ダンパー特許** (`US8,115,092B2` / `JP5614108B2`) | 押鍵・ペダルに応じたダンパー粘性係数テーブルの動的書換 | 幾何学的クリアランス境界入力による片側非線形ヒステリシス接触パッド（`DamperPadBank`） |
| **4** | **ヤマハ・VRM共鳴特許** (`JP6176132B2`) | 主音源と並列に付加する「共鳴音生成エフェクト装置」構成 | 全230弦と開放弦を単一の10極縮約ブリッジ状態空間内で対等に双方向連成 |
| **5** | **カワイ・共鳴定位特許** (`JP6805060B2` / `US10,102,837B1`) | 鍵別共鳴を少数グループへ集約し後段の遅延回路で定位付与 | 弦の実ブリッジ接続座標に基づく8ゾーン空間重み射影（物理的空間位相差） |
| **6** | **ヤマハ・棚板衝突音特許** (`JP7306402B2` / `US11,961,499B2`) | 弦打撃音と棚板衝突音の2系統個別減衰速度制御 | 打鍵・離鍵衝撃力を第9ゾーン（Zone 8）外力インパルスとして放射器へ直接入力 |
| **7** | **カシオ・倍音ダンプ判定特許** (`JP7400925B2`) | 押鍵と「倍音関係にある他鍵がダンプ状態か」の `if-then` 分岐 | 倍音関係判定ロジックを一切持たず、全230弦へ共通ブリッジ力を一様作用 |
