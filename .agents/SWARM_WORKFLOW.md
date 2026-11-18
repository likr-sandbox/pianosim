# Swarm マルチエージェント開発・Worktree・Git運用規約 (`SWARM_WORKFLOW.md`)

本ドキュメントは、本プロジェクトを `swarm`／エージェントチームで実装する際の役割分担、`workspace:"share"` worktree 運用、Tier 別検証ゲート、および `.agents/AGENTS.md` 継続学習プロトコルを定める。

関連仕様: [ARCHITECTURE.md](../docs/ARCHITECTURE.md) / [ROADMAP.md](../docs/ROADMAP.md) / [DECISIONS_AND_FALLBACKS.md](../docs/DECISIONS_AND_FALLBACKS.md) / [AGENTS.md](AGENTS.md)

---

## 1. エージェント編成と `workspace` モード分担

責務干渉および単一 `target/`・CPU 競合を防ぐため、役割を以下に分離する（**旧5名同時フル検証は廃止**）。

| 役割名（例） | `workspace` モード | 担当責務 | 禁止事項 |
|---|---|---|---|
| **`coordinator`** | （調整役） | タスク分割、`swarm{action:"add"}` 配置、ボード指示、マイルストーン完了判定、イベント駆動報告 | 自らのコード編集・長時間のビルド/テスト実行・Git操作 |
| **`author` 群**<br>（`simd`, `spec`, `strings`, `action`, `bridge`, `body-gen`, `radiator`, `web-ui` 等） | **`share`**<br>（独立 Git worktree） | 割り当てクレート／ファイルの設計・実装・対象テスト・自己ビルド、論理単位コミット、`Learnings:` 報告 | 他 `author` 所有ファイルの編集、500行超ファイル、`.agents/AGENTS.md` 直接編集、`git push` |
| **`reviewer`** | **`inherit`**<br>（Step A 検証主体） | 該当 Tier の静的検査（`fmt`/`clippy`/`check-lines`）・Rust テスト・ビルド・Playwright E2E 実行、設計/アーキレビュー、アーキ変更時の `docs/` 更新整合確認 | ソース直接編集、無関係 Tier の過剰実行 |
| **`auditor`** | **`inherit`**<br>（Step A 並行監査） | 差分・FTO 7原則・クリーンルーム・定数整合性・ワークアラウンド排除・`docs/` 同期および `.agents/` 隔離の独立監査（**Cargo/E2E を重複実行しない**） | ソース直接編集、`reviewer` と同一テストの重複実行 |
| **`challenger`** | **`inherit`**<br>（Step B 単独計測） | Step A 合格後の**無競合アイドル CPU** 上で `scenarios` → `bench` → `bench:web` を単独実行（Tier 2/3 のみ） | Step A 中や他 Rust ビルドとの同時実行 |
| **`integrator`** | **`inherit`**<br>（`main` 管理主体） | 検証済みブランチの `main` 統合（リベース/FF）、アーキ変更時の `docs/` 最終同期確認、`.agents/AGENTS.md` 更新、タグ＋`git push`、**`CI` & Pages 両確認** | 検証承認前の強行統合、マイルストーン途中 `git push` |
| **`monitor`** | `inherit` | `swarm_monitor` スキルに基づく停滞検知とデッドロック解除通知 | 停滞時以外の不要な発言 |

### 1.1 クォータ枯渇・モデル切替時のルール
- `[quota]` 警告や `RESOURCE_EXHAUSTED` 発生時、`coordinator` は `kill` ＋再 `add` を行わず、`swarm{action:"model", name, model}` で履歴と作業中 worktree を維持したまま代替モデルへ切り替える。

---

## 2. 着手前同期・`share` Worktree・数値校正・タイムアウト規約

### 2.0 開始前リモート同期ゲート（必須）
- 初期調査（Survey）および各 Wave / マイルストーン開始時の**最初のコマンド**として以下を必須実行し、`origin/main` の先行コミットを取り込んだ上で作業を開始する。
  ```bash
  git fetch origin && git rev-list --left-right --count HEAD...origin/main
  ```

### 2.1 1ファイル1オーナー原則（ウェーブ分割設計）
1. 同一ウェーブ内で並列稼働する複数の `author` に、同じクレートや同じファイルを絶対に共有させない。
2. 共通境界（`piano-spec` 型・`piano-wasm` C-ABI 等）の変更は先行ウェーブで `main` 統合してから後続並列ウェーブを分岐させる。
3. 複数施策で `.pbdy` 再ベイクが必要になる場合は、中間施策ごとの再ベイクを避け Wave 末尾のベイク工程へ1回に集約する。

### 2.2 `share` Worktree 上の作業・一括スイープ・プローブ制御
1. **作業ブランチ作成**: `author` は作業開始前にルート側最新 `main` から専用ブランチを切る（待機中の再割り当て時も最新 `main` から切り直す）。
   ```bash
   git checkout -B m<N>/<worker>-<topic> main
   ```
2. **一括パラメータスイープ規約（定数山登り禁止）**: 3回以上の定数・ゲイン・音響極値調整が見込まれるタスクでは、定数を1箇所書き換えては `cargo test` を回す手動ループを禁止する。1回のプログラム実行で候補レンジを掃引して目標誤差表を出力するスイープテスト／CLI ハーネスで定数を特定し、1〜2回で反映する。
3. **使い捨てプローブ／ブラウザ検証の `timeout` 強制**: ワンオフの Node/Playwright プローブを実行する際は必ず **`timeout -k 5s 180s ...`** でラップし、スクリプト末尾の `try / finally` で `browser.close()` と `server.close()` を保証する（子サーバー残置ハング防止）。
4. **完了前セルフチェック**: 対象クレートの `cargo test -p <crate>` / `cargo clippy`（Web側は `npm run typecheck` / `npm run build`）、`<= 500` 行、`git status --porcelain` が空であることを確認し、`Learnings:` 付きで報告する。

### 2.3 Worktree 破棄（`kill`）前の絶対安全条件
- `swarm{action:"kill"}` 実行前に ① 未コミット変更なし、② 全コミットの `main` 統合済み、③ 統合後 `main` 検証済み の3条件をすべて満たすこと。休止目的の `kill` は禁止。

---

## 3. コミット粒度・Tier 別排他検証・マイルストーン単位 `push`

### 3.1 コミット粒度と履歴保全統合
- Conventional Commits（`feat(<scope>): ...`, `fix(...)`, `test(...)`, `perf(...)`, `refactor(...)`, `docs(agents): ...`, `ci:`）で1論理責務ごとにコミットする。
- `integrator` は `git rebase` ＋ `--ff-only`（または論理順 `cherry-pick`）で履歴粒度を保って統合する（`--squash` 禁止）。

### 3.2 変更差分別ゲート（Tier 0〜3）＆ Step A / Step B 排他検証

変更パスと修正範囲に応じて検証コストを段階化し、同一リソースの重複実行を禁止する。

| Tier | 該当する変更範囲 | 必須実行ゲート（最小十分集合） | 省略する処理 |
|---|---|---|---|
| **Tier 0** | `docs/**`, `.agents/**`, `*.md` のみ | `piano-cli check-lines`, `git status --porcelain` | ビルド、全テスト、E2E、ベンチ |
| **Tier 1** | `apps/web/e2e/**` のみ | `check-lines`, `npm run typecheck`, 変更 spec `--repeat-each=5`, `npm run test:e2e` | Rust テスト、`scenarios`、ネイティブ `bench`、`/tmp` プローブ |
| **Tier 2** | `apps/web/src/**`, `apps/web/playwright.config.ts`, `package.json` のみ | `check-lines`, `npm run build`, `check-wasm-simd`, `cargo test -p piano-wasm`, `npm run test:e2e`, `bench:web` | 他9クレート Rust テスト、`scenarios`、ネイティブ `bench` |
| **Tier 3** | `crates/**`, `Cargo.*`, `.github/workflows/**` 変更あり | `fmt --check`, `check-lines`, `clippy` (workspace + `wasm32`), `cargo test --workspace`, `build`, `check-wasm-simd`, `test:e2e`, `scenarios`, `bench`, `bench:web` | — |

- **中間サブウェーブ統合時**: `crates/**` 変更時でも変更クレート＋直接下流（`cargo test -p <crate>` 等）に留め、ワークスペース全体テスト・E2E・ベンチは Wave / マイルストーン末尾に集約する。
- **排他2ステップ実行**:
  - **Step A（並列・非競合）**: `Reviewer` 1名（該当 Tier のビルド・テスト・E2E）＋ `Auditor` 1名（コード/FTO/クリーンルーム/変更範囲監査、Cargo/E2E 再実行なし）。
  - **Step B（Step A 合格後、CPU 競合ゼロで単独実行・Tier 2/3）**: `Challenger` 1名が `scenarios` → `sleep 3 && bench` → `sleep 3 && bench:web` を直列実行する。
- **同一ソースツリーの再検証省略**: `Final Integrator` または最終監査で直前の検証済み SHA との差分が `crates/ apps/ Cargo.toml Cargo.lock package.json` に無い場合（`.agents/AGENTS.md` 更新等）、`check-lines` ＋ `git diff --stat <verified>..HEAD` のみで通過としフルテストを再実行しない。

### 3.3 マイルストーン境界のみでの `git push` と CI ＋ Pages 両確認
1. マイルストーン途中の `git push` は禁止。全検収通過後に `integrator` が `.agents/AGENTS.md` を更新コミットし、タグ打刻と `git push origin main --tags` を1回実行する（例：`git tag -a m<N> -m "Milestone <N> verified"`）。
2. `git push` 後は [deploy.yml](../.github/workflows/deploy.yml) が [ci.yml](../.github/workflows/ci.yml) 成功を待って公開するため、**GitHub Actions `CI` の `success`** と **`https://likr-sandbox.github.io/pianosim/` の HTTP 200（および実ブラウザ無エラー）** の両方を完了条件とする。

---

## 4. `.agents/AGENTS.md` 継続的学習・`docs/` 同期プロトコル

1. **Single-Writer 原則**: `author` は `.agents/AGENTS.md` を直接編集せず `Learnings:` 欄で報告し、`integrator` のみが `main` 上で統合・剪定する。
2. **上限厳守と剪定**: 常に **500行以下かつ 24,000バイト未満** を維持し、古い記述は書き換えて重複を削除する。
3. **自動継承**: `.agents/rules/project-kb.md` により全エージェント・worktree へ自動注入される。

### 4.1 `docs/` 同期義務とエージェント情報の隔離原則
1. **アーキテクチャ変更時の `docs/` 同時更新義務**: クレート・モジュール構成、4ステージ物理モデル、Wasm/Worker/IPC 実行境界、公開 CLI/Web 機能、または D1〜D8 設計判断に関わるアーキテクチャレベルの更新を行う際は、同一変更／マイルストーン内で `docs/ARCHITECTURE.md`・`docs/ROADMAP.md`・`docs/DECISIONS_AND_FALLBACKS.md`（および必要に応じ `README.md`）を最新実装に合わせて必ず更新する（`reviewer` / `auditor` は実装と `docs/` の整合を必須確認する）。
2. **エージェント用情報の `.agents/` 限定原則**: Swarm・マルチエージェント編成、worktree 運用、Tier 別検証手順、エージェント向け規約・蓄積知見などのエージェント用情報は `docs/` や `README.md` には一切含めず、必ず `.agents/` 配下（`AGENTS.md`, `SWARM_WORKFLOW.md`, `rules/`, `skills/`）のみに配置する。

---

## 5. ボード通信・コンテキスト効率化ルール

- **`@mention` は起動ベル**: 不要な CC `@mention` や `@all` を避け、長文はボード投稿番号で参照する。
- **イベント駆動報告・出力サイズ規律**: 親会話への進捗報告は **Wave 完了・ブロッカー発生・ユーザー判断が必要な場合のみ** 送信し（高頻度定時通知による親コンテキスト消費とキャッシュ期限切れ再課金を防止）、設計・ハンドオフ本文は `<= 20 KiB`（詳細はファイルパス参照）に収める。
