//! Static checks for 500-line file limits, AGENTS.md byte budget, and clean-room rules.

use std::fs;
use std::path::{Path, PathBuf};

/// Maximum allowed lines per checked source or documentation file.
pub const MAX_LINES_PER_FILE: usize = 500;
/// Maximum allowed size in bytes for `.agents/AGENTS.md` (strictly `< 24_000`).
pub const MAX_AGENTS_BYTES: u64 = 24_000;

const CHECKED_EXTENSIONS: &[&str] = &["rs", "ts", "js", "css", "md"];
const SKIP_DIR_NAMES: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "dist",
    "test-results",
    "playwright-report",
    "blob-report",
];
const FORBIDDEN_PATTERNS: &[&str] = &["agpl", "openpiano"];

/// Summary metrics reported by `check_repo`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckSummary {
    pub checked_files: usize,
    pub max_lines: usize,
    pub max_lines_file: String,
    pub agents_bytes: u64,
    pub clean_room_files: usize,
}

/// Runs all static repository checks rooted at `root`.
pub fn check_repo(root: &Path) -> Result<CheckSummary, String> {
    let mut violations = Vec::new();
    let mut summary = CheckSummary::default();

    let agents_path = root.join(".agents/AGENTS.md");
    if agents_path.is_file() {
        let meta = fs::metadata(&agents_path)
            .map_err(|e| format!("failed to stat {}: {e}", agents_path.display()))?;
        summary.agents_bytes = meta.len();
        if meta.len() >= MAX_AGENTS_BYTES {
            violations.push(format!(
                ".agents/AGENTS.md is {} bytes (must be < {MAX_AGENTS_BYTES})",
                meta.len()
            ));
        }
    }

    let mut files = Vec::new();
    collect_files(root, root, &mut files)?;
    files.sort();

    for abs_path in files {
        let rel = rel_unix_path(root, &abs_path);
        let is_line_target = has_extension(&abs_path, CHECKED_EXTENSIONS);
        let is_clean_target = is_clean_room_target(&rel, &abs_path);

        if !is_line_target && !is_clean_target {
            continue;
        }

        let content = fs::read_to_string(&abs_path)
            .map_err(|e| format!("failed to read {rel} as UTF-8: {e}"))?;

        if is_line_target {
            let line_count = content.lines().count();
            summary.checked_files += 1;
            if line_count >= summary.max_lines {
                summary.max_lines = line_count;
                summary.max_lines_file.clone_from(&rel);
            }
            if line_count > MAX_LINES_PER_FILE {
                violations.push(format!(
                    "{rel}: {line_count} lines exceeds limit {MAX_LINES_PER_FILE}"
                ));
            }
        }

        if is_clean_target {
            summary.clean_room_files += 1;
            let lower = content.to_ascii_lowercase();
            for &pat in FORBIDDEN_PATTERNS {
                if lower.contains(pat) {
                    violations.push(format!("{rel}: contains forbidden pattern '{pat}'"));
                }
            }
        }
    }

    if violations.is_empty() {
        Ok(summary)
    } else {
        Err(violations.join("\n"))
    }
}

/// Locates the workspace root starting from `current_dir`.
#[must_use]
pub fn find_repo_root(start: &Path) -> PathBuf {
    for ancestor in start.ancestors() {
        if ancestor.join("Cargo.toml").is_file() && ancestor.join("crates").is_dir() {
            return ancestor.to_path_buf();
        }
    }
    start.to_path_buf()
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries =
        fs::read_dir(dir).map_err(|e| format!("failed to read dir {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("failed reading dir entry: {e}"))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|e| format!("failed to stat {}: {e}", path.display()))?;
        if file_type.is_dir() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            let rel = rel_unix_path(root, &path);
            if SKIP_DIR_NAMES.contains(&name_str.as_ref()) || rel == "apps/web/dist" {
                continue;
            }
            collect_files(root, &path, out)?;
        } else if file_type.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn rel_unix_path(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    rel.to_string_lossy().replace('\\', "/")
}

fn has_extension(path: &Path, exts: &[&str]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| exts.contains(&ext))
}

fn is_clean_room_target(rel: &str, path: &Path) -> bool {
    if rel.starts_with("docs/") || rel.starts_with(".agents/") {
        return false;
    }
    if rel == "crates/piano-cli/src/check_lines.rs" {
        return false;
    }
    if path.file_name().and_then(|n| n.to_str()) == Some("Cargo.toml") {
        return true;
    }
    (rel.starts_with("crates/") && has_extension(path, &["rs"]))
        || (rel.starts_with("apps/") && has_extension(path, &["ts"]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "pianosim_check_lines_{label}_{}_{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn check_repo_enforces_line_limits_and_clean_room_scan() {
        let dir = make_temp_dir("limits");
        fs::create_dir_all(dir.join(".agents")).unwrap();
        fs::create_dir_all(dir.join("crates/demo/src")).unwrap();
        fs::create_dir_all(dir.join("target")).unwrap();
        fs::write(dir.join(".agents/AGENTS.md"), "# ok\n").unwrap();
        fs::write(dir.join("Cargo.toml"), "[workspace]\n").unwrap();
        fs::write(dir.join("target/ignored.rs"), "line\n".repeat(600)).unwrap();
        fs::write(dir.join("crates/demo/src/lib.rs"), "pub fn ok() {}\n").unwrap();

        let ok = check_repo(&dir).expect("clean repo passes");
        assert!(ok.checked_files >= 2);
        assert!(ok.max_lines <= MAX_LINES_PER_FILE);

        fs::write(dir.join("crates/demo/src/too_long.rs"), "x\n".repeat(501)).unwrap();
        let err = check_repo(&dir).expect_err("501 lines must fail");
        assert!(err.contains("501 lines"));

        fs::remove_file(dir.join("crates/demo/src/too_long.rs")).unwrap();
        fs::write(
            dir.join("crates/demo/src/bad.rs"),
            "// reference from OpenPiano\n",
        )
        .unwrap();
        let err = check_repo(&dir).expect_err("forbidden mention must fail");
        assert!(err.contains("forbidden pattern"));

        let _ = fs::remove_dir_all(&dir);
    }
}
