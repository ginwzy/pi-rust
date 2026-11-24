//! Boundary entries from extensions (upstream `_applyBoundaryDrafts`, `_buildBoundaryContext`, and
//! `_commitBoundaryDrafts`): handlers of `turn_end` and `agent_before_settle` see the context their
//! drafts would produce, and the drafts are appended once all handlers ran.

use pi_ai::Message;
use pi_js::HostValue;
use serde::Serialize;
use serde_json::Value;

use super::{AgentSession, lock};
use crate::compaction::estimate_projected_context_tokens;
use crate::session::{Entry, NewCompaction, PendingEntry, SessionManager, context};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Preview<'a> {
    context_entries: Vec<PreviewEntry<'a>>,
    context_messages: &'a [Message],
    llm_messages: &'a [Message],
    pending_messages: &'a [Message],
    can_continue: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PreviewEntry<'a> {
    source_entry: &'a Entry,
    messages: &'a [Message],
}

fn text(draft: &Value, key: &str) -> String {
    draft[key].as_str().unwrap_or_default().to_string()
}

fn optional(draft: &Value, key: &str) -> Option<Value> {
    draft.get(key).filter(|value| !value.is_null()).cloned()
}

/// Append `drafts` to `manager`; returns the new entry ids.
fn apply_drafts(manager: &mut SessionManager, drafts: &[Value]) -> anyhow::Result<Vec<String>> {
    let mut appended = Vec::new();
    for draft in drafts {
        let entry = build_draft(manager, &[], draft)?;
        appended.push(manager.append_entry(entry)?);
    }
    Ok(appended)
}

impl AgentSession {
    /// `BoundaryContextPreview` for `drafts` (the `boundary.context` operation).
    pub(crate) fn boundary_preview(&self, kind: &str, drafts: &Value) -> Result<HostValue, String> {
        let drafts = drafts.as_array().map_or(&[][..], Vec::as_slice);
        let mut pending = self.agent.peek_queued_messages();
        let queued = !pending.is_empty();
        let has_pending_custom = {
            let state = lock(&self.state);
            pending.extend(state.pending_custom.iter().cloned());
            !state.pending_custom.is_empty()
        };
        let manager = self.session_manager();
        let mut overlay = Vec::with_capacity(drafts.len());
        for draft in drafts {
            let entry = build_draft(&manager, &overlay, draft).map_err(|error| format!("{error:#}"))?;
            overlay.push(entry);
        }
        let projection = manager.project_overlay(&overlay);
        let llm = crate::messages::convert_to_llm(&projection.messages);
        let preview = Preview {
            context_entries: projection
                .entries
                .iter()
                .map(|entry| PreviewEntry { source_entry: entry.source, messages: &entry.messages })
                .collect(),
            context_messages: &projection.messages,
            llm_messages: &llm,
            pending_messages: &pending,
            can_continue: can_continue_with(&llm, kind, has_pending_custom, queued),
        };
        HostValue::serialized(&preview).map_err(|error| error.to_string())
    }

    /// Whether the current context can run another request.
    pub(crate) fn boundary_can_continue(&self, kind: &str) -> bool {
        let messages = self.session_manager().build_session_context().messages;
        let has_pending_custom = !lock(&self.state).pending_custom.is_empty();
        can_continue_with(
            &crate::messages::convert_to_llm(&messages),
            kind,
            has_pending_custom,
            self.agent.has_queued_messages(),
        )
    }

    /// Append the drafts the boundary handlers settled on.
    pub(crate) fn commit_boundary_drafts(&self, drafts: &[Value]) {
        if drafts.is_empty() {
            return;
        }
        let appended = apply_drafts(&mut self.session_manager(), drafts);
        self.refresh_context();
        match appended {
            Ok(ids) => ids.iter().for_each(|id| self.emit_entry(id)),
            Err(error) => eprintln!("Warning: could not append boundary entries: {error:#}"),
        }
    }
}

fn can_continue_with(llm: &[Message], kind: &str, pending_custom: bool, queued: bool) -> bool {
    let final_role = llm.last().map(Message::role);
    let has_context = llm.iter().any(|message| message.role() != "system");
    (has_context && final_role != Some("assistant"))
        || pending_custom
        || if kind == "turn_end" { queued } else { final_role == Some("assistant") && queued }
}

fn build_draft(manager: &SessionManager, overlay: &[Entry], draft: &Value) -> anyhow::Result<Entry> {
    let id = loop {
        let id = manager.new_entry_id();
        if !overlay.iter().any(|entry| entry.id() == id) {
            break id;
        }
    };
    let parent = overlay.last().map(Entry::id).or_else(|| manager.leaf_id());
    let pending = PendingEntry { id, parent };
    Ok(match draft["type"].as_str() {
        Some("custom") => pending.custom(&text(draft, "customType"), optional(draft, "data")),
        Some("custom_message") => pending.custom_message(
            &text(draft, "customType"),
            serde_json::from_value(draft["content"].clone())?,
            draft["display"].as_bool().unwrap_or(false),
            optional(draft, "details"),
        ),
        Some("context_edit") => {
            let branch = manager.get_branch(None);
            let id = text(draft, "targetId");
            let target = branch.into_iter().chain(overlay).find(|entry| entry.id() == id);
            let Some(target) = target else {
                if manager.get_entry(&id).is_some() {
                    anyhow::bail!("Entry {id} is not on the active branch");
                }
                anyhow::bail!("Entry {id} not found");
            };
            let replacement = draft["replacement"].get("content").filter(|value| !value.is_null()).cloned();
            pending.context_edit(target, replacement)?
        }
        Some("compaction") => {
            let mut path = manager.get_branch(None);
            path.extend(overlay);
            let projection = manager.project_overlay(overlay);
            let tokens_before = estimate_projected_context_tokens(&projection, &path).tokens;
            pending.compaction(
                context::current_system_message(path),
                NewCompaction {
                    summary: text(draft, "summary"),
                    first_kept_entry_id: draft["firstKeptEntryId"].as_str().map(str::to_string),
                    tokens_before,
                    details: optional(draft, "details"),
                    from_hook: Some(true),
                    usage: optional(draft, "usage").and_then(|usage| serde_json::from_value(usage).ok()),
                },
            )
        }
        other => anyhow::bail!("Unknown boundary entry type: {}", other.unwrap_or("<missing>")),
    })
}
