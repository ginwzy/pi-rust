//! Appending entries. Each entry becomes a child of the leaf and the new leaf. Field order
//! follows the upstream object literals so files match ones the TypeScript pi writes.

use pi_ai::{ContentBlock, Message, SystemMessage, ThinkingLevel, Usage, UserContent};
use serde_json::{Value, json};

use super::SessionManager;
use super::entry::{Entry, EntryData};
use super::ids::{iso_timestamp, parse_iso};

/// Fields of a compaction entry. `first_kept_entry_id: None` keeps nothing before the entry.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NewCompaction {
    pub summary: String,
    pub first_kept_entry_id: Option<String>,
    pub tokens_before: u64,
    pub details: Option<Value>,
    pub from_hook: Option<bool>,
    pub usage: Option<Usage>,
}

fn to_json<T: serde::Serialize>(value: &T) -> Option<Value> {
    serde_json::to_value(value).ok()
}

pub(crate) struct PendingEntry<'a> {
    pub id: String,
    pub parent: Option<&'a str>,
}

impl PendingEntry<'_> {
    fn link(self) -> [(&'static str, Option<Value>); 3] {
        [
            ("id", Some(json!(self.id))),
            ("parentId", Some(json!(self.parent))),
            ("timestamp", Some(json!(iso_timestamp()))),
        ]
    }

    fn fields(self, kind: &str) -> Vec<(&'static str, Option<Value>)> {
        let mut fields = vec![("type", Some(json!(kind)))];
        fields.extend(self.link());
        fields
    }

    pub fn custom(self, custom_type: &str, data: Option<Value>) -> Entry {
        let mut fields =
            vec![("type", Some(json!("custom"))), ("customType", Some(json!(custom_type))), ("data", data)];
        fields.extend(self.link());
        Entry::build(fields)
    }

    pub fn custom_message(
        self,
        custom_type: &str,
        content: UserContent,
        display: bool,
        details: Option<Value>,
    ) -> Entry {
        let mut fields = vec![
            ("type", Some(json!("custom_message"))),
            ("customType", Some(json!(custom_type))),
            ("content", to_json(&content)),
            ("display", Some(json!(display))),
            ("details", details),
        ];
        fields.extend(self.link());
        Entry::build(fields)
    }

    pub fn compaction(self, mut system: Option<SystemMessage>, compaction: NewCompaction) -> Entry {
        let first_kept = compaction.first_kept_entry_id.as_deref().unwrap_or(&self.id).to_owned();
        let timestamp = iso_timestamp();
        if let Some(system) = &mut system {
            system.timestamp = parse_iso(&timestamp).unwrap_or_default();
        }
        Entry::build(vec![
            ("type", Some(json!("compaction"))),
            ("id", Some(json!(self.id))),
            ("parentId", Some(json!(self.parent))),
            ("timestamp", Some(json!(timestamp))),
            ("summary", Some(json!(compaction.summary))),
            ("firstKeptEntryId", Some(json!(first_kept))),
            ("tokensBefore", Some(json!(compaction.tokens_before))),
            ("details", compaction.details),
            ("usage", compaction.usage.as_ref().and_then(to_json)),
            ("fromHook", compaction.from_hook.map(Value::Bool)),
            ("systemMessage", system.as_ref().and_then(to_json)),
        ])
    }

    pub fn context_edit(self, target: &Entry, replacement: Option<Value>) -> anyhow::Result<Entry> {
        if replacement.as_ref().is_some_and(|content| !content.is_string() && !content.is_array()) {
            anyhow::bail!("Context edit replacement must be null or contain string/array content");
        }
        let target_id = target.id();
        let role = match target.data() {
            EntryData::CustomMessage(_) => "custom",
            EntryData::Message(message) if matches!(message.role(), "user" | "assistant" | "toolResult") => {
                message.role()
            }
            _ => anyhow::bail!("Entry {target_id} does not contribute editable model content"),
        };
        let replacement = match replacement {
            Some(Value::String(text)) if matches!(role, "assistant" | "toolResult") => {
                json!({ "content": [ContentBlock::text(text)] })
            }
            Some(content) => json!({ "content": content }),
            None => Value::Null,
        };
        let mut fields = self.fields("context_edit");
        fields.extend([("targetId", Some(json!(target_id))), ("replacement", Some(replacement))]);
        Ok(Entry::build(fields))
    }
}

impl SessionManager {
    fn append_typed(&mut self, kind: &str, fields: Vec<(&str, Option<Value>)>) -> anyhow::Result<String> {
        let mut all = self.entry_fields(self.new_entry_id(), kind);
        all.extend(fields);
        self.append_entry(Entry::build(all))
    }

    fn entry_fields(&self, id: String, kind: &str) -> Vec<(&'static str, Option<Value>)> {
        PendingEntry { id, parent: self.leaf_id() }.fields(kind)
    }

    fn pending_entry(&self) -> PendingEntry<'_> {
        PendingEntry { id: self.new_entry_id(), parent: self.leaf_id() }
    }

    /// Append a message. Summary roles belong in compaction and branch summary entries instead.
    pub fn append_message(&mut self, message: &Message) -> anyhow::Result<String> {
        let fields = self.entry_fields(self.new_entry_id(), "message");
        self.append_entry(Entry::build_message(fields, message)?)
    }

    /// Append a message under an id from [`reserve_entry_id`](Self::reserve_entry_id).
    pub fn append_message_as(&mut self, id: &str, message: &Message) -> anyhow::Result<String> {
        anyhow::ensure!(self.reserved.remove(id), "entry id {id} is not reserved");
        let fields = self.entry_fields(id.to_string(), "message");
        self.append_entry(Entry::build_message(fields, message)?)
    }

    pub fn append_thinking_level_change(&mut self, level: ThinkingLevel) -> anyhow::Result<String> {
        self.append_typed("thinking_level_change", vec![("thinkingLevel", Some(json!(level.as_str())))])
    }

    pub fn append_model_change(&mut self, provider: &str, model_id: &str) -> anyhow::Result<String> {
        self.append_typed("model_change", vec![("provider", Some(json!(provider))), ("modelId", Some(json!(model_id)))])
    }

    /// Model-attributed usage outside the LLM context, such as `cache_warm`.
    pub fn append_usage(
        &mut self,
        kind: &str,
        provider: &str,
        model: &str,
        usage: &Usage,
        note: Option<&str>,
    ) -> anyhow::Result<String> {
        self.append_typed(
            "usage",
            vec![
                ("kind", Some(json!(kind))),
                ("provider", Some(json!(provider))),
                ("model", Some(json!(model))),
                ("usage", to_json(usage)),
                ("note", note.filter(|note| !note.is_empty()).map(|note| json!(note))),
            ],
        )
    }

    /// A compaction with only the required fields.
    pub fn append_compaction(
        &mut self,
        summary: &str,
        first_kept_entry_id: &str,
        tokens_before: u64,
    ) -> anyhow::Result<String> {
        self.append_compaction_with(NewCompaction {
            summary: summary.to_string(),
            first_kept_entry_id: Some(first_kept_entry_id.to_string()),
            tokens_before,
            ..NewCompaction::default()
        })
    }

    /// A compaction entry with its system checkpoint: the replayed prompt and tools at this point.
    pub fn append_compaction_with(&mut self, compaction: NewCompaction) -> anyhow::Result<String> {
        let entry = self.pending_entry().compaction(self.current_system_message(), compaction);
        self.append_entry(entry)
    }

    /// Extension state that never enters the LLM context.
    pub fn append_custom_entry(&mut self, custom_type: &str, data: Option<Value>) -> anyhow::Result<String> {
        let entry = self.pending_entry().custom(custom_type, data);
        self.append_entry(entry)
    }

    /// Set the display name. Line breaks become spaces; an empty name clears it.
    pub fn append_session_info(&mut self, name: &str) -> anyhow::Result<String> {
        let mut sanitized = String::new();
        let mut in_break = false;
        for ch in name.chars() {
            if ch == '\r' || ch == '\n' {
                if !in_break {
                    sanitized.push(' ');
                }
                in_break = true;
            } else {
                sanitized.push(ch);
                in_break = false;
            }
        }
        self.append_typed("session_info", vec![("name", Some(json!(sanitized.trim())))])
    }

    /// An extension message that enters the LLM context as a `custom` role message.
    pub fn append_custom_message_entry(
        &mut self,
        custom_type: &str,
        content: UserContent,
        display: bool,
        details: Option<Value>,
    ) -> anyhow::Result<String> {
        let entry = self.pending_entry().custom_message(custom_type, content, display, details);
        self.append_entry(entry)
    }

    /// Edit an earlier entry's contribution to future model context on this branch. `None`
    /// omits the target; `Some(content)` (string or block array) replaces only its content.
    pub fn append_context_edit(&mut self, target_id: &str, replacement: Option<Value>) -> anyhow::Result<String> {
        if replacement.as_ref().is_some_and(|content| !content.is_string() && !content.is_array()) {
            anyhow::bail!("Context edit replacement must be null or contain string/array content");
        }
        let Some(target) = self.get_entry(target_id) else { anyhow::bail!("Entry {target_id} not found") };
        if !self.get_branch(None).iter().any(|entry| entry.id() == target_id) {
            anyhow::bail!("Entry {target_id} is not on the active branch");
        }
        let entry = self.pending_entry().context_edit(target, replacement)?;
        self.append_entry(entry)
    }

    /// Set (or clear with `None` or an empty label) a bookmark on an entry.
    pub fn append_label_change(&mut self, target_id: &str, label: Option<&str>) -> anyhow::Result<String> {
        if self.get_entry(target_id).is_none() {
            anyhow::bail!("Entry {target_id} not found");
        }
        let label_value = label.map(|label| json!(label));
        let id = self.append_typed("label", vec![("targetId", Some(json!(target_id))), ("label", label_value)])?;
        let timestamp = self.get_entry(&id).map(|entry| entry.timestamp().to_string()).unwrap_or_default();
        match label.filter(|label| !label.is_empty()) {
            Some(label) => {
                self.labels.insert(target_id.to_string(), (label.to_string(), timestamp));
            }
            None => {
                self.labels.remove(target_id);
            }
        }
        Ok(id)
    }

    /// Move the leaf to `branch_from` (`None` for a new root) and record a summary of the
    /// abandoned path there. `fromId` is the leaf before the move, or `root`.
    pub fn branch_with_summary(
        &mut self,
        branch_from: Option<&str>,
        summary: &str,
        details: Option<Value>,
        from_hook: Option<bool>,
        usage: Option<Usage>,
    ) -> anyhow::Result<String> {
        if let Some(id) = branch_from
            && self.get_entry(id).is_none()
        {
            anyhow::bail!("Entry {id} not found");
        }
        let from_id = self.leaf.clone().unwrap_or_else(|| "root".into());
        self.leaf = branch_from.map(String::from);
        self.bump_revision();
        self.append_typed(
            "branch_summary",
            vec![
                ("fromId", Some(json!(from_id))),
                ("summary", Some(json!(summary))),
                ("details", details),
                ("usage", usage.as_ref().and_then(to_json)),
                ("fromHook", from_hook.map(Value::Bool)),
            ],
        )
    }
}
