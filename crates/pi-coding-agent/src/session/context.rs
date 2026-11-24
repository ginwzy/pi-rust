//! Rebuild model context from the entry tree: follow the active branch, apply the latest
//! compaction, then apply `context_edit` entries. Port of upstream `buildSessionProjection`.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::{Arc, Mutex, Weak};

use pi_ai::transcript::replay_system_messages;
use pi_ai::{ContentBlock, Message, SystemMessage, ThinkingLevel, UserContent};
use serde::Deserialize;
use serde_json::Value;

use super::entry::{ContextReplacement, Entry, EntryData};
use crate::messages::{create_branch_summary_message, create_compaction_summary_message, create_custom_message};

/// Which entry a path ends at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leaf<'a> {
    /// The last entry in file order.
    Last,
    /// Before the first entry: an empty path.
    Root,
    /// This entry, or the last entry when the id is unknown.
    Entry(&'a str),
}

impl<'a> From<Option<&'a str>> for Leaf<'a> {
    fn from(leaf: Option<&'a str>) -> Self {
        leaf.map_or(Leaf::Root, Leaf::Entry)
    }
}

/// Messages, model, and thinking level rebuilt from the current branch.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionContext {
    pub messages: Vec<Message>,
    /// `(provider, model id)` from the latest model change or assistant message.
    pub model: Option<(String, String)>,
    /// The latest thinking level change, if any (upstream reports `off` when absent).
    pub thinking_level: Option<ThinkingLevel>,
}

/// One selected entry and the messages it contributes after context edits.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedEntry<'a> {
    pub source: &'a Entry,
    /// Empty for state-only entries and omitted targets.
    pub messages: Vec<Message>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionProjection<'a> {
    pub entries: Vec<ProjectedEntry<'a>>,
    pub messages: Vec<Message>,
    pub model: Option<(String, String)>,
    pub thinking_level: Option<ThinkingLevel>,
}

impl SessionProjection<'_> {
    pub fn into_context(self) -> SessionContext {
        SessionContext { messages: self.messages, model: self.model, thinking_level: self.thinking_level }
    }
}

/// Entries from the root to `leaf`, following `parentId` links.
pub fn build_session_path<'a>(entries: &'a [Entry], leaf: Leaf<'_>) -> Vec<&'a Entry> {
    let index: HashMap<&str, &Entry> = entries.iter().map(|entry| (entry.id(), entry)).collect();
    let start = match leaf {
        Leaf::Root => return Vec::new(),
        Leaf::Last => entries.last(),
        Leaf::Entry(id) => index.get(id).copied().or(entries.last()),
    };
    let mut path = Vec::new();
    let mut current = start;
    while let Some(entry) = current {
        path.push(entry);
        if path.len() > entries.len() {
            break;
        }
        current = entry.parent_id().and_then(|parent| index.get(parent).copied());
    }
    path.reverse();
    path
}

/// The path from the root to the last of `entries`, following `parentId` within them.
pub fn path_to_last<'a>(entries: &[&'a Entry]) -> Vec<&'a Entry> {
    let index: HashMap<&str, &Entry> = entries.iter().map(|entry| (entry.id(), *entry)).collect();
    let mut path = Vec::new();
    let mut current = entries.last().copied();
    while let Some(entry) = current {
        path.push(entry);
        if path.len() > entries.len() {
            break;
        }
        current = entry.parent_id().and_then(|parent| index.get(parent).copied());
    }
    path.reverse();
    path
}

/// The latest compaction entry in `entries`.
pub fn latest_compaction<'a>(entries: &[&'a Entry]) -> Option<&'a Entry> {
    entries.iter().rev().find(|entry| entry.compaction().is_some()).copied()
}

/// The active entries: the latest compaction first, then its kept non-system entries, then later entries.
pub fn build_context_entries<'a>(entries: &'a [Entry], leaf: Leaf<'_>) -> Vec<&'a Entry> {
    context_entries_of_path(build_session_path(entries, leaf))
}

pub(crate) fn context_entries_of_path(path: Vec<&Entry>) -> Vec<&Entry> {
    let Some(compaction_index) = path.iter().rposition(|entry| entry.compaction().is_some()) else { return path };
    let compaction = path[compaction_index];
    let first_kept = compaction.compaction().map(|data| data.first_kept_entry_id.as_str()).unwrap_or_default();
    let mut selected = vec![compaction];
    let mut found = false;
    for entry in &path[..compaction_index] {
        found |= entry.id() == first_kept;
        if found && !entry.is_message_role("system") {
            selected.push(entry);
        }
    }
    selected.extend(&path[compaction_index + 1..]);
    selected
}

/// Messages one entry contributes to context, before context edits.
pub fn session_entry_to_context_messages(entry: &Entry) -> Vec<Message> {
    let mut messages = Vec::new();
    visit_messages(entry, |message| messages.push(message));
    messages
}

fn visit_messages(entry: &Entry, mut emit: impl FnMut(Message)) {
    match entry.data() {
        EntryData::Message(message) => emit(message.clone()),
        EntryData::CustomMessage(custom) => emit(create_custom_message(
            &custom.custom_type,
            custom.content.clone(),
            custom.display,
            custom.details.clone(),
            entry.timestamp(),
        )),
        EntryData::BranchSummary(summary) if !summary.summary.is_empty() => {
            emit(create_branch_summary_message(&summary.summary, &summary.from_id, entry.timestamp()));
        }
        EntryData::Compaction(compaction) => {
            if let Some(system) = &compaction.system_message {
                emit(Message::from(system.clone()));
            }
            emit(create_compaction_summary_message(&compaction.summary, compaction.tokens_before, entry.timestamp()));
        }
        _ => {}
    }
}

fn block_content(content: &Value) -> Vec<ContentBlock> {
    match content {
        Value::String(text) => vec![ContentBlock::text(text.clone())],
        other => Vec::deserialize(other).unwrap_or_default(),
    }
}

fn replace_content(message: Message, replacement: &ContextReplacement) -> Message {
    match message {
        Message::User(mut user) => {
            Arc::make_mut(&mut user).content =
                UserContent::deserialize(&replacement.content).unwrap_or(UserContent::Blocks(Vec::new()));
            Message::User(user)
        }
        Message::Assistant(mut assistant) => {
            Arc::make_mut(&mut assistant).content = block_content(&replacement.content);
            Message::Assistant(assistant)
        }
        Message::ToolResult(mut result) => {
            Arc::make_mut(&mut result).content = block_content(&replacement.content);
            Message::ToolResult(result)
        }
        Message::Custom(mut custom) if custom.role == "custom" => {
            Arc::make_mut(&mut custom).fields.insert("content".into(), replacement.content.clone());
            Message::Custom(custom)
        }
        other => other,
    }
}

struct Source<'a> {
    entry: &'a Entry,
    edit: Option<&'a Entry>,
    omitted: bool,
    replacement: Option<&'a ContextReplacement>,
}

impl Source<'_> {
    fn append_messages(&self, messages: &mut Vec<Message>) {
        if self.omitted {
            return;
        }
        visit_messages(self.entry, |message| {
            messages.push(match self.replacement {
                Some(replacement) => replace_content(message, replacement),
                None => message,
            });
        });
    }
}

fn context_sources(path: Vec<&Entry>) -> impl Iterator<Item = Source<'_>> {
    let selected = context_entries_of_path(path);
    let edits: HashMap<_, _> = selected
        .iter()
        .filter_map(|entry| match entry.data() {
            EntryData::ContextEdit { target_id, replacement } => {
                Some((target_id.as_str(), (*entry, replacement.as_ref())))
            }
            _ => None,
        })
        .collect();
    selected.into_iter().enumerate().map(move |(index, entry)| {
        let edit = edits.get(entry.id()).copied();
        Source {
            entry,
            edit: edit.map(|(edit, _)| edit),
            omitted: matches!(edit, Some((_, None))) || (index > 0 && entry.compaction().is_some()),
            replacement: edit.and_then(|(_, replacement)| replacement),
        }
    })
}

pub(crate) fn has_context_messages(path: Vec<&Entry>) -> bool {
    context_sources(path).any(|source| {
        !source.omitted
            && match source.entry.data() {
                EntryData::Message(_) | EntryData::CustomMessage(_) | EntryData::Compaction(_) => true,
                EntryData::BranchSummary(summary) => !summary.summary.is_empty(),
                _ => false,
            }
    })
}

pub(crate) fn current_system_message(path: Vec<&Entry>) -> Option<SystemMessage> {
    replay_system_messages(context_sources(path).filter(|source| !source.omitted).filter_map(|source| {
        match source.entry.data() {
            EntryData::Message(Message::System(system)) => Some(system.as_ref()),
            EntryData::Compaction(compaction) => compaction.system_message.as_ref(),
            _ => None,
        }
    }))
}

/// Model and thinking level from the whole path, as upstream `getSessionContextSettings`.
pub fn context_settings(path: &[&Entry]) -> (Option<(String, String)>, Option<ThinkingLevel>) {
    let (mut model, mut thinking_level) = (None, None);
    for entry in path {
        update_settings(entry, &mut model, &mut thinking_level);
    }
    (model.map(|(provider, model)| (provider.clone(), model.clone())), thinking_level)
}

fn update_settings<'a>(
    entry: &'a Entry,
    model: &mut Option<(&'a String, &'a String)>,
    thinking_level: &mut Option<ThinkingLevel>,
) {
    match entry.data() {
        EntryData::ThinkingLevelChange { thinking_level: level } => {
            *thinking_level = level.parse::<ThinkingLevel>().ok().or(*thinking_level);
        }
        EntryData::ModelChange { provider, model_id } => *model = Some((provider, model_id)),
        EntryData::Message(Message::Assistant(assistant)) => *model = Some((&assistant.provider, &assistant.model)),
        _ => {}
    }
}

#[derive(Debug, Default)]
pub(super) struct ProjectionCache(pub Mutex<Option<CachedProjection>>);

impl Clone for ProjectionCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[derive(Debug)]
struct ProjectedSpan {
    index: usize,
    source: Weak<Value>,
    edit: Option<Weak<Value>>,
    omitted: bool,
    range: Range<usize>,
}

#[derive(Debug, Default)]
pub(super) struct CachedProjection {
    pub revision: u64,
    pub leaf: Option<String>,
    pub context: SessionContext,
    spans: Vec<ProjectedSpan>,
    edit_targets: HashSet<String>,
}

impl CachedProjection {
    pub fn rebuild(
        path: Vec<&Entry>,
        revision: u64,
        leaf: Option<String>,
        index: &HashMap<String, usize>,
        previous: Option<&Self>,
    ) -> Self {
        let (model, thinking_level) = context_settings(&path);
        let mut cached = Self {
            revision,
            leaf,
            context: SessionContext { model, thinking_level, ..SessionContext::default() },
            ..Self::default()
        };
        let reuse = previous.map(Self::by_source).unwrap_or_default();
        for source in context_sources(path) {
            if let EntryData::ContextEdit { target_id, .. } = source.entry.data() {
                cached.edit_targets.insert(target_id.clone());
            }
            let start = cached.context.messages.len();
            if let Some(messages) = previous.and_then(|previous| previous.reusable(&reuse, &source)) {
                cached.context.messages.extend_from_slice(messages);
            } else {
                source.append_messages(&mut cached.context.messages);
            }
            cached.spans.push(ProjectedSpan {
                index: index[source.entry.id()],
                source: Arc::downgrade(&source.entry.shared_raw()),
                edit: source.edit.map(|edit| Arc::downgrade(&edit.shared_raw())),
                omitted: source.omitted,
                range: start..cached.context.messages.len(),
            });
        }
        cached
    }

    pub fn append(&mut self, revision: u64, entry: &Entry, index: usize) {
        if self.revision != revision - 1
            || self.leaf.as_deref() != entry.parent_id()
            || matches!(entry.kind(), "compaction" | "context_edit")
            || self.edit_targets.contains(entry.id())
        {
            return;
        }
        let start = self.context.messages.len();
        visit_messages(entry, |message| self.context.messages.push(message));
        self.spans.push(ProjectedSpan {
            index,
            source: Arc::downgrade(&entry.shared_raw()),
            edit: None,
            omitted: false,
            range: start..self.context.messages.len(),
        });
        let mut model = None;
        update_settings(entry, &mut model, &mut self.context.thinking_level);
        if let Some((provider, model)) = model {
            self.context.model = Some((provider.clone(), model.clone()));
        }
        self.leaf = Some(entry.id().into());
        self.revision = revision;
    }

    pub fn project<'a>(&self, entries: &'a [Entry]) -> SessionProjection<'a> {
        let SessionContext { messages, model, thinking_level } = self.context.clone();
        let entries = self
            .spans
            .iter()
            .map(|span| ProjectedEntry {
                source: &entries[span.index],
                messages: messages[span.range.clone()].to_vec(),
            })
            .collect();
        SessionProjection { entries, messages, model, thinking_level }
    }

    pub fn overlay<'a>(&self, path: Vec<&'a Entry>) -> SessionProjection<'a> {
        let reuse = self.by_source();
        project_sources(path, |source, messages| {
            if let Some(reused) = self.reusable(&reuse, source) {
                messages.extend_from_slice(reused);
            } else {
                source.append_messages(messages);
            }
        })
    }

    fn by_source(&self) -> HashMap<*const Value, &ProjectedSpan> {
        self.spans.iter().map(|span| (span.source.as_ptr(), span)).collect()
    }

    fn reusable(&self, reuse: &HashMap<*const Value, &ProjectedSpan>, source: &Source<'_>) -> Option<&[Message]> {
        let span = reuse.get(&Arc::as_ptr(&source.entry.shared_raw()))?;
        let edit = source.edit.map(|edit| Arc::as_ptr(&edit.shared_raw()));
        (span.omitted == source.omitted && span.edit.as_ref().map(Weak::as_ptr) == edit)
            .then(|| &self.context.messages[span.range.clone()])
    }
}

/// Provenance-preserving, compaction-aware model context for `leaf`.
pub fn build_session_projection<'a>(entries: &'a [Entry], leaf: Leaf<'_>) -> SessionProjection<'a> {
    project_path(build_session_path(entries, leaf))
}

pub(crate) fn project_path(path: Vec<&Entry>) -> SessionProjection<'_> {
    project_sources(path, Source::append_messages)
}

fn project_sources<'a>(
    path: Vec<&'a Entry>,
    mut append_messages: impl FnMut(&Source<'a>, &mut Vec<Message>),
) -> SessionProjection<'a> {
    let (model, thinking_level) = context_settings(&path);
    let entries: Vec<ProjectedEntry> = context_sources(path)
        .map(|source| {
            let mut messages = Vec::new();
            append_messages(&source, &mut messages);
            ProjectedEntry { source: source.entry, messages }
        })
        .collect();
    let messages = entries.iter().flat_map(|entry| entry.messages.iter().cloned()).collect();
    SessionProjection { entries, messages, model, thinking_level }
}

/// The finalized model context for `leaf`.
pub fn build_session_context(entries: &[Entry], leaf: Leaf<'_>) -> SessionContext {
    build_context_of_path(build_session_path(entries, leaf))
}

pub(crate) fn build_context_of_path(path: Vec<&Entry>) -> SessionContext {
    let (model, thinking_level) = context_settings(&path);
    let mut messages = Vec::new();
    for source in context_sources(path) {
        source.append_messages(&mut messages);
    }
    SessionContext { messages, model, thinking_level }
}

#[cfg(test)]
mod tests;
