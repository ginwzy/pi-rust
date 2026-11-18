//! Per-call display evidence, independent of prompt-scoped undo checkpoints.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::edit_diff::generate_diff_string;

pub const INPUT_LIMIT: usize = 128 * 1024;
const DIFF_LIMIT: usize = 32 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Operation {
    Create,
    Update,
    Write,
    Delete,
    Move,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", tag = "kind", content = "text")]
pub enum Preview<S = String> {
    Diff(S),
    ExistingEditDiff,
    Unavailable,
    Unchanged,
    PathOnly,
}

impl Preview {
    pub fn old_limit(new_len: usize) -> Option<usize> {
        INPUT_LIMIT.checked_sub(new_len)
    }

    pub fn between(old: &str, new: &str) -> Self {
        if Self::old_limit(new.len()).is_none_or(|limit| old.len() > limit) || old.contains('\0') || new.contains('\0')
        {
            return Self::Unavailable;
        }
        if old == new {
            return Self::Unchanged;
        }
        // Limit diff work even for small inputs containing thousands of tiny lines.
        if old.lines().chain(new.lines()).take(4097).count() > 4096 {
            return Self::Unavailable;
        }
        let (diff, _) = generate_diff_string(old, new);
        if diff.len() > DIFF_LIMIT { Self::Unavailable } else { Self::Diff(diff) }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FileChange<S = String> {
    pub operation: Operation,
    pub path: S,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub to: Option<S>,
    pub preview: Preview<S>,
}

/// Validated, allocation-free projection of persisted details, including referenced edit text.
pub struct View<'a> {
    entries: &'a [Value],
    edit: Option<&'a str>,
    pub omitted: usize,
    pub error: Option<&'a str>,
}

impl<'a> View<'a> {
    pub fn read(details: &'a Value) -> Option<Self> {
        let stored = FileResult::stored(details)?;
        let view = Self {
            entries: stored.get("entries")?.as_array()?,
            edit: details.get("diff").and_then(Value::as_str),
            omitted: stored.get("omitted")?.as_u64()?.try_into().ok()?,
            error: optional_text(stored.get("error"))?,
        };
        view.entries.iter().all(|entry| view.entry(entry).is_some()).then_some(view)
    }

    pub fn entries(&self) -> impl ExactSizeIterator<Item = FileChange<&'a str>> + '_ {
        self.entries.iter().map(|entry| self.entry(entry).expect("validated immutable file result"))
    }

    pub fn diff(&self, preview: &Preview<&'a str>) -> Option<&'a str> {
        match preview {
            Preview::Diff(text) => Some(*text),
            Preview::ExistingEditDiff => self.edit,
            _ => None,
        }
    }

    fn entry(&self, entry: &'a Value) -> Option<FileChange<&'a str>> {
        let operation = match entry.get("operation")?.as_str()? {
            "create" => Operation::Create,
            "update" => Operation::Update,
            "delete" => Operation::Delete,
            "move" => Operation::Move,
            "write" => Operation::Write,
            _ => return None,
        };
        let preview = entry.get("preview")?;
        let preview = match preview.get("kind")?.as_str()? {
            "diff" => Preview::Diff(preview.get("text")?.as_str()?),
            "existingEditDiff" if self.edit.is_some() => Preview::ExistingEditDiff,
            "unavailable" => Preview::Unavailable,
            "pathOnly" => Preview::PathOnly,
            "unchanged" => Preview::Unchanged,
            _ => return None,
        };
        let to = optional_text(entry.get("to"))?;
        if (operation == Operation::Move) != to.is_some() {
            return None;
        }
        Some(FileChange { operation, path: entry.get("path")?.as_str()?, to, preview })
    }
}

fn optional_text(value: Option<&Value>) -> Option<Option<&str>> {
    match value {
        None | Some(Value::Null) => Some(None),
        Some(v) => v.as_str().map(Some),
    }
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct FileResult {
    pub entries: Vec<FileChange>,
    pub omitted: usize,
    pub error: Option<String>,
    #[serde(skip)]
    bytes: usize,
}

#[derive(Clone, Copy)]
pub struct Budget(Option<usize>);

impl Budget {
    pub fn remaining(self, path: &str, to: Option<&str>) -> Option<usize> {
        self.0?.checked_sub(path.len())?.checked_sub(to.map_or(0, str::len))
    }
    pub fn preview(self, path: &str, to: Option<&str>, build: impl FnOnce() -> Preview) -> Preview {
        if self.remaining(path, to).unwrap_or(0) == 0 {
            return Preview::Unavailable;
        }
        build()
    }
}

impl FileResult {
    pub fn budget(&self) -> Budget {
        Budget((self.omitted == 0 && self.entries.len() < 128).then(|| (64 * 1024usize).saturating_sub(self.bytes)))
    }

    pub fn push(&mut self, mut entry: FileChange) {
        let names = entry.path.len() + entry.to.as_ref().map_or(0, String::len);
        let Some(remaining) = self.budget().remaining(&entry.path, entry.to.as_deref()) else {
            self.omitted += 1;
            return;
        };
        self.bytes += names;
        if let Preview::Diff(text) = &entry.preview {
            if text.len() > remaining {
                entry.preview = Preview::Unavailable;
            } else {
                self.bytes += text.len();
            }
        }
        self.entries.push(entry);
    }

    pub fn attach(self, details: &mut Value) {
        details["fileChanges"] = json!({"version": 1, "result": self});
    }

    pub fn stored(details: &Value) -> Option<&Value> {
        let report = details.get("fileChanges")?;
        (report.get("version")?.as_u64()? == 1).then(|| report.get("result"))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhausted_budget_never_acquires_preview_input() {
        for budget in [Budget(None), Budget(Some(4))] {
            assert!(matches!(budget.preview("file", None, || panic!("unexpected preview read")), Preview::Unavailable));
        }
    }
}
