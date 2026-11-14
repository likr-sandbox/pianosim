# Swarm マルチエージェント開発・Worktree・Git運用規約 (`SWARM_WORKFLOW.md`)

本ドキュメントは、本プロジェクトを `swarm` ベースのマルチエージェント体制で実装する際の役割分担、`workspace:"share"` worktree運用、コミット・プッシュ粒度、および `.agents/AGENTS.md` 継続学習プロトコルを定める。

関連仕様: [ARCHITECTURE.md](file:///home/likr/pianosim/docs/ARCHITECTURE.md) / [ROADMAP.md](file:///home/likr/pianosim/docs/ROADMAP.md) / [DECISIONS_AND_FALLBACKS.md](file:///home/likr/pianosim/docs/DECISIONS_AND_FALLBACKS.md) / [AGENTS.md](file:///home/likr/pianosim/.agents/AGENTS.md)

---

## 1. エージェント編成と `workspace` モード分担

Swarm起動時は、責務とGitワークスペースの干渉を防ぐため、必ず以下の4種別（＋停滞監視）に役割を分離する。

| 役割名（例） | `workspace` モード | 担当責務 | 禁止事項 |
|---|---|---|---|
| **`coordinator`** | （調整役） | タスク分割、`swarm{action:"add"}` によるワーカー配置、ボード上の次工程指示、マイルストーン完了判定、ユーザー報告 | 自らのコード編集・長時間のビルド/テスト実行・Git操作 |
| **`author` 群**<br>（`simd`, `spec`, `strings`, `action`, `bridge`, `body-gen`, `radiator`, `web-ui` 等） | **`share`**<br>（各ワーカー専用の独立Git worktree） | 割り当てられたクレート／ファイルの設計・実装・単体テスト・自己ビルド検証、自身のworktree内での論理単位コミット、発見事項の `Learnings:` 報告 | 他 `author` が所有するファイルの編集、500行超ファイル作成、`.agents/AGENTS.md` の直接編集、`git push` |
| **`verifier`** | **`inherit`**<br>（読み取り専用検証） | 差分レビュー、`cargo test --workspace`、`cargo clippy --workspace -- -D warnings`、`piano-cli check-lines`、音響／性能ゲート計測、Playwright E2E（実Wasmのみ・モック禁止）実行 | ソースコード・設定ファイルの直接編集（不具合は該当 `@<author>` へ差し戻す） |
| **`integrator`** | **`inherit`**<br>（ルート `main` 管理の唯一主体） | 検証済み `author` ブランチの `main` へのリベース＆統合、`.agents/AGENTS.md` への知見統合・剪定、マイルストーン完了時のタグ打刻・`git push`・GitHub Pages実稼働確認 | `verifier` 承認前の強行統合、マイルストーン途中の `git push` |
| **`monitor`** | `inherit` | `swarm_monitor` スキルに基づく停滞検知とデッドロック解除通知 | 停滞時以外の不要な発言 |

### 1.1 クォータ枯渇・モデル切替時のルール
- ワーカーまたは自身に `[quota]` 警告や `RESOURCE_EXHAUSTED` が発生した場合、`coordinator` は絶対に `kill` ＋再 `add` を行わず、`swarm{action:"model", name, model}` で会話履歴と作業中worktreeを維持したまま代替モデルへ切り替える。

---

## 2. `workspace:"share"` Worktree ライフサイクル規約

### 2.1 1ファイル1オーナー原則（ウェーブ分割設計）
1. 同一ウェーブ内で並列稼働する複数の `author` に、**同じクレートや同じファイルを絶対に共有させない**。
2. 共通の型・境界（例：`piano-spec` のトレイト・イベント構造体や `piano-wasm` のIPC契約）を変更する必要がある場合は、先行ウェーブでインターフェースを確定・`main` 統合してから、後続の並列実装ウェーブ（例：`piano-strings` と `piano-action` の同時開発）を分岐させる。

### 2.2 `share` Worktree 上の作業手順
1. **作業ブランチ作成**: `share` worktreeで起動した `author` は、作業開始前に必ず専用作業ブランチを作成する（`main` はルートworktreeがチェックアウトしているため直接操作しない）。
   ```bash
   git checkout -B m<N>/<worker>-<topic> main
   ```
   例：`m2/strings-zoh`、`m3/action-stulov`、`m6/radiator-nupols`
2. **ウェーブ間の同期**: 待機中の同一 `author` に次ウェーブの作業を割り当てる場合、`author` は必ずルート側の最新 `main` から新規ブランチを切り直してから作業する（古い親コミットの上に積み続けない）。
3. **完了前セルフチェック**: `author` は報告前に自身のworktree内で以下を確認する。
   - 対象クレートの `cargo test` / `cargo clippy -- -D warnings`（Web側は `npm run build`）が通過すること。
   - 変更・新規作成した全ファイルが **500行以下** であること。
   - 未追跡・未コミット変更が残っていないこと（`git status --porcelain` が空）。
4. **ボード完了報告フォーマット**:
   - 次のアクション担当（通常は `@verifier`、単体統合確認のみなら `@integrator`）**のみ**を `@mention` し、ブランチ名、コミットSHA（複数なら範囲）、変更ファイルと各行数、テスト結果、および **`Learnings:`**（環境罠・数値知見等があれば）を簡潔に投稿する。

### 2.3 Worktree 破棄（`kill`）前の絶対安全条件
- `swarm{action:"kill"}` は対象ワーカーの `share` worktree ディレクトリを**永久削除**する。
- **以下の3条件をすべて満たすまで、`author` を絶対に `kill` してはならない**:
  1. `author` のworktreeに未コミット変更が存在しないこと。
  2. 該当ブランチの全コミットが `integrator` によりルート `main` へ統合済みであること。
  3. 統合後の `main` 上で全テストがパスしていること。
- 待機中（`waiting`）ワーカーはトークンを消費しないため、一時休止目的で `kill` を使用しない。

---

## 3. コミット粒度とマイルストーン単位 `push` 規約

### 3.1 適切なコミット粒度（Conventional Commits）
1. **1コミット＝1論理責務**: 「1サブモジュール実装＋単体テスト」「1つのCIワークフロー」「1つのバグ修正」を最小単位とし、無関係な複数機能の抱き合わせコミットや、ビルドが通らないWIPコミットを禁止する。
2. **プレフィックス規約**:
   - `feat(<scope>): ...`（例：`feat(piano-strings): implement ZOH modal SoA SIMD kernel`）
   - `fix(<scope>): ...`
   - `test(<scope>): ...`
   - `perf(<scope>): ...`
   - `refactor(<scope>): ...`
   - `docs(<scope>): ...`（`.agents/AGENTS.md` 更新は `docs(agents): ...`）
   - `ci: ...` / `build: ...`
3. **履歴保全型の統合（`integrator` の責務）**:
   - `integrator` は `author` のブランチを統合する際、`git merge --squash` で粒度を潰さず、`author` ブランチを最新 `main` へリベースした上で Fast-Forward マージ（または論理順 `cherry-pick`）を行い、意味のあるコミット粒度を `main` 履歴に残す。
   - 例：
     ```bash
     # ルートワークtree (/home/likr/pianosim) 上の integrator 操作例
     git cherry-pick <author-commit-1> <author-commit-2>
     # または author に git rebase main を指示した後に git merge --ff-only m<N>/<branch>
     ```

### 3.2 マイルストーン境界のみでの `git push`
1. **マイルストーン途中の `git push` 禁止**: `main` へのプッシュのたびに `.github/workflows/deploy.yml` が起動し GitHub Pages が更新されるため、**マイルストーン未完了状態での `git push` を禁止**する。
2. **マイルストーン完了ゲートと `push` 手順**:
   1. `@verifier` が当該マイルストーン（`M<N>`）のすべての検収基準（ビルド・全テスト・500行上限・FTO検査・音響/性能指標・E2E）の通過を確認し、ボードへ `MILESTONE M<N> VERIFIED` を投稿して `@integrator` を呼ぶ。
   2. `@integrator` はルート `main` 上で全ウェーブのコミットが揃っていることを確認し、当該マイルストーンの学びを `.agents/AGENTS.md` に統合・剪定して `docs(agents): update AGENTS.md for M<N>` をコミットする。
   3. `@integrator` は最終ゲート（`cargo test --workspace`、`piano-cli check-lines`）を `main` で再確認した上で、マイルストーンタグを打刻して**1回だけプッシュ**する。
      - M1で旧TSコードを削除する直前に限り、先頭 `c75cf57` へ `v0.1.0-ts-dwg` タグを打刻する。
      - 各マイルストーン完了時：`git tag -a m<N> -m "Milestone <N> verified"` ＋ `git push origin main --tags`
   4. `integrator`（続いて必要に応じて `verifier`）は、`https://likr-sandbox.github.io/pianosim/` のHTTP 200応答および実ブラウザでの動作（コンソールエラー0件）を確認し、`@coordinator` へ完了報告する。

---

## 4. `.agents/AGENTS.md` 継続的学習・最新化プロトコル

プロジェクト遂行中に得た知見（数値安定の条件、SIMD/Wasm最適化の勘所、環境固有制約、テスト上の落とし穴）を確実に蓄積しつつ、マージ競合と肥大化を防ぐため以下を徹底する。

1. **Single-Writer原則（競合ゼロ）**:
   - 並列稼働する `author` ワーカーは自身の `share` worktree で `.agents/AGENTS.md` を編集しない。代わりに、ボード報告末尾に `Learnings:` ブロックを設けて知見を報告する。
   - `integrator` のみが `main` 上で `.agents/AGENTS.md` を更新・コミットする。
2. **陳腐化情報の剪定・最新化（Append-Only禁止）**:
   - `integrator` は新知見を追記する際、解決済みの古い注意点・実装変更で無効になった仕様を必ず書き換えまたは削除し、重複を統合する。
   - `.agents/AGENTS.md` は常に **500行以下かつ 24,000バイト未満**（Antigravityルールの単一ファイル上限）を維持する。
3. **自動継承**: `.agents/rules/project-kb.md` により `.agents/AGENTS.md` は全エージェント・全worktreeのコンテキストへ自動注入される。

---

## 5. ボード通信とコスト最適化ルール

- **`@mention` は次工程の起動ベル**: 応答や作業が不要な相手へのCC的 `@mention` や不要な `@all` は使わない（1メンションごとに1ターン分のトークンが消費される）。
- **シーケンス番号参照**: 先行ワーカーの長い出力を別ワーカーへの指示にコピペせず、ボードの投稿番号（例：`@verifier #12 と #15 の統合結果を検証してください`）で指示する。
- **ユーザーへの報告**: `@coordinator` は各マイルストーン完了の `push` と公開確認が終わった段階で、成果・検証結果・学んだ知見（`AGENTS.md` 更新要旨）を簡潔にユーザーへ報告する。
