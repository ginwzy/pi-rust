//! Immutable session lines with one authoritative body and ordered JSON serialization.

use std::borrow::Cow;
use std::io::{self, Write};
use std::sync::{Arc, OnceLock, Weak};

use pi_ai::{Message, SystemMessage, Usage, UserContent};
use serde::{Deserialize, Deserializer, Serialize, Serializer, ser::SerializeMap};
use serde_json::value::{RawValue, to_raw_value};
use serde_json::{Map, Value};

/// One raw JSON line of a session file (header or entry), in stored key order.
pub type FileEntry = Map<String, Value>;

pub const CURRENT_SESSION_VERSION: u32 = 3;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionHeader {
    #[serde(rename = "type")]
    pub kind: String,
    /// Absent in version 1 files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
    pub id: String,
    #[serde(default)]
    pub timestamp: String,
    #[serde(default)]
    pub cwd: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session: Option<String>,
}

pub(crate) fn lenient_u64<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value
        .and_then(|value| value.as_u64().or_else(|| value.as_f64().map(|float| float.max(0.0) as u64)))
        .unwrap_or(0))
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageEntry {
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub model: String,
    #[serde(default, deserialize_with = "lenient_usage")]
    pub usage: Usage,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionEntry {
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub first_kept_entry_id: String,
    #[serde(default, deserialize_with = "lenient_u64")]
    pub tokens_before: u64,
    #[serde(default)]
    pub details: Option<Value>,
    #[serde(default, deserialize_with = "lenient_optional_usage")]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub from_hook: Option<bool>,
    #[serde(default, deserialize_with = "lenient_system_message")]
    pub system_message: Option<SystemMessage>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryEntry {
    #[serde(default)]
    pub from_id: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub details: Option<Value>,
    #[serde(default, deserialize_with = "lenient_optional_usage")]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub from_hook: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomMessageEntry {
    #[serde(default)]
    pub custom_type: String,
    #[serde(default = "empty_user_content", deserialize_with = "lenient_content")]
    pub content: UserContent,
    #[serde(default)]
    pub details: Option<Value>,
    #[serde(default)]
    pub display: bool,
}

/// Content of a context edit replacement: a string or an array of content blocks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextReplacement {
    #[serde(default)]
    pub content: Value,
}

/// Typed fields of an entry. `Other` covers unknown types, which are kept and ignored.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum EntryData {
    Message(Message),
    ThinkingLevelChange { thinking_level: String },
    ModelChange { provider: String, model_id: String },
    Usage(UsageEntry),
    Compaction(CompactionEntry),
    BranchSummary(BranchSummaryEntry),
    Custom { custom_type: String, data: Option<Value> },
    CustomMessage(CustomMessageEntry),
    ContextEdit { target_id: String, replacement: Option<ContextReplacement> },
    Label { target_id: String, label: Option<String> },
    SessionInfo { name: Option<String> },
    Other,
}

fn lenient_usage<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Usage, D::Error> {
    Ok(lenient_optional_usage(deserializer)?.unwrap_or_default())
}

fn lenient_optional_usage<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Usage>, D::Error> {
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value.filter(Value::is_object).map(|mut value| {
        normalize_usage(&mut value);
        serde_json::from_value(value).unwrap_or_default()
    }))
}

fn lenient_system_message<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<SystemMessage>, D::Error> {
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value.and_then(|mut value| {
        if let Some(object) = value.as_object_mut() {
            object.entry("content").or_insert(Value::String(String::new()));
            object.entry("timestamp").or_insert(Value::from(0));
        }
        serde_json::from_value(value).ok()
    }))
}

fn empty_user_content() -> UserContent {
    UserContent::Blocks(Vec::new())
}

fn lenient_content<'de, D: Deserializer<'de>>(deserializer: D) -> Result<UserContent, D::Error> {
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value.and_then(|value| serde_json::from_value(value).ok()).unwrap_or(UserContent::Blocks(Vec::new())))
}

fn normalize_usage(value: &mut Value) {
    let Some(usage) = value.as_object_mut() else { return };
    for key in ["input", "output", "cacheRead", "cacheWrite", "totalTokens"] {
        let number = usage.get(key).and_then(|value| value.as_f64()).unwrap_or(0.0).max(0.0) as u64;
        usage.insert(key.into(), Value::from(number));
    }
    let cost = usage.entry("cost").or_insert_with(|| Value::Object(Map::new()));
    if !cost.is_object() {
        *cost = Value::Object(Map::new());
    }
    if let Some(cost) = cost.as_object_mut() {
        for key in ["input", "output", "cacheRead", "cacheWrite", "total"] {
            let number = cost.get(key).and_then(Value::as_f64).unwrap_or(0.0);
            cost.insert(key.into(), Value::from(number));
        }
    }
}

/// Parse a stored message, filling fields that older or hand-edited files may lack,
/// as upstream `sessionEntryToContextMessages` tolerates missing content.
pub fn parse_message_lenient(value: &Value) -> Option<Message> {
    if let Ok(message) = Message::from_value(value) {
        return Some(message);
    }
    let mut object = value.as_object()?.clone();
    let role = object.get("role")?.as_str()?.to_string();
    let empty_content = if role == "system" { Value::String(String::new()) } else { Value::Array(Vec::new()) };
    if object.get("content").is_none_or(Value::is_null) {
        object.insert("content".into(), empty_content);
    }
    if !object.get("timestamp").is_some_and(Value::is_number) {
        object.insert("timestamp".into(), Value::from(0));
    }
    match role.as_str() {
        "assistant" => {
            for key in ["api", "provider", "model"] {
                object.entry(key).or_insert(Value::String(String::new()));
            }
            object.entry("stopReason").or_insert(Value::String("stop".into()));
            let usage = object.entry("usage").or_insert_with(|| Value::Object(Map::new()));
            normalize_usage(usage);
        }
        "toolResult" => {
            for key in ["toolCallId", "toolName"] {
                object.entry(key).or_insert(Value::String(String::new()));
            }
            object.entry("isError").or_insert(Value::Bool(false));
            if let Some(usage) = object.get_mut("usage") {
                normalize_usage(usage);
            }
        }
        _ => {}
    }
    serde_json::from_value(Value::Object(object)).ok()
}

#[derive(Clone, Debug)]
pub struct Entry(pub(super) Arc<StoredEntry>);

#[derive(Debug)]
pub struct WeakEntry(Weak<StoredEntry>);

impl WeakEntry {
    pub fn matches(&self, entry: &Entry) -> bool {
        std::ptr::eq(self.0.as_ptr(), Arc::as_ptr(&entry.0))
    }

    pub fn is_alive(&self) -> bool {
        self.0.strong_count() != 0
    }
}

#[derive(Debug)]
pub(super) struct StoredEntry {
    kind: String,
    id: String,
    parent_id: Option<String>,
    timestamp: String,
    body: Body,
    encoded_bytes: OnceLock<usize>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum Body {
    Native(EntryData),
    OrderedRaw(Box<RawValue>),
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Envelope<'a> {
    #[serde(rename = "type", borrow)]
    kind: Cow<'a, str>,
    #[serde(borrow)]
    id: Cow<'a, str>,
    #[serde(borrow)]
    parent_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    timestamp: Cow<'a, str>,
    #[serde(borrow)]
    message: Option<&'a RawValue>,
    #[serde(borrow)]
    data: Option<&'a RawValue>,
}

fn string_field(raw: &FileEntry, key: &str) -> Option<String> {
    raw.get(key).and_then(Value::as_str).map(String::from)
}

impl Entry {
    pub fn from_raw(raw: FileEntry) -> Self {
        let compact = to_raw_value(&raw).expect("serialize entry object");
        Self::candidate(compact.get()).unwrap_or_else(|| Self::ordered_raw(&raw, compact))
    }

    pub fn from_json_line(line: &str) -> serde_json::Result<Self> {
        if let Some(entry) = Self::candidate(line) {
            return Ok(entry);
        }
        let raw: FileEntry = serde_json::from_str(line)?;
        let compact = to_raw_value(&raw)?;
        Ok(Self::ordered_raw(&raw, compact))
    }

    fn ordered_raw(raw: &FileEntry, compact: Box<RawValue>) -> Self {
        Self(Arc::new(StoredEntry {
            id: string_field(raw, "id").unwrap_or_default(),
            parent_id: string_field(raw, "parentId"),
            timestamp: string_field(raw, "timestamp").unwrap_or_default(),
            kind: string_field(raw, "type").unwrap_or_default(),
            encoded_bytes: OnceLock::from(compact.get().len()),
            body: Body::OrderedRaw(compact),
        }))
    }

    fn candidate(line: &str) -> Option<Self> {
        let envelope: Envelope<'_> = serde_json::from_str(line).ok()?;
        let data = match envelope.kind.as_ref() {
            "message" => EntryData::Message(Message::from_json(envelope.message?.get()).ok()?),
            "custom" if envelope.data.is_some() => return None,
            kind => decode_data(kind, line)?,
        };
        if matches!(data, EntryData::Other) {
            return None;
        }
        let wire = Native {
            kind: &envelope.kind,
            id: &envelope.id,
            parent_id: envelope.parent_id.as_deref(),
            timestamp: &envelope.timestamp,
            data: &data,
        };
        if !exact_match(line.as_bytes(), &wire) {
            return None;
        }
        Some(Self(Arc::new(StoredEntry {
            kind: envelope.kind.into_owned(),
            id: envelope.id.into_owned(),
            parent_id: envelope.parent_id.map(Cow::into_owned),
            timestamp: envelope.timestamp.into_owned(),
            body: Body::Native(data),
            encoded_bytes: OnceLock::from(line.len()),
        })))
    }

    pub(super) fn native(kind: &str, id: String, parent_id: Option<&str>, timestamp: String, data: EntryData) -> Self {
        Self(Arc::new(StoredEntry {
            kind: kind.into(),
            id,
            parent_id: parent_id.map(str::to_owned),
            timestamp,
            body: Body::Native(data),
            encoded_bytes: OnceLock::new(),
        }))
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub fn downgrade(&self) -> WeakEntry {
        WeakEntry(Arc::downgrade(&self.0))
    }

    /// The entry `type`, such as `message` or `compaction`.
    pub fn kind(&self) -> &str {
        &self.0.kind
    }

    pub fn id(&self) -> &str {
        &self.0.id
    }

    pub fn parent_id(&self) -> Option<&str> {
        self.0.parent_id.as_deref()
    }

    /// ISO 8601 entry timestamp.
    pub fn timestamp(&self) -> &str {
        &self.0.timestamp
    }

    pub fn data(&self) -> Cow<'_, EntryData> {
        match &self.0.body {
            Body::Native(data) => Cow::Borrowed(data),
            Body::OrderedRaw(raw) => {
                let data = if self.kind() == "message" {
                    #[cfg(test)]
                    tests::note_decode(self.id());
                    self.raw_message()
                        .and_then(|message| {
                            Message::from_json(message.get())
                                .ok()
                                .or_else(|| parse_message_lenient(&serde_json::from_str(message.get()).ok()?))
                        })
                        .map_or(EntryData::Other, EntryData::Message)
                } else {
                    parse_data(self.kind(), &serde_json::from_str(raw.get()).expect("stored object"))
                };
                Cow::Owned(data)
            }
        }
    }

    pub fn ordered_json(&self) -> Option<&RawValue> {
        match &self.0.body {
            Body::OrderedRaw(raw) => Some(raw),
            Body::Native(_) => None,
        }
    }

    pub fn raw_message(&self) -> Option<&RawValue> {
        #[derive(Deserialize)]
        struct Fields<'a> {
            #[serde(borrow)]
            message: Option<&'a RawValue>,
        }
        serde_json::from_str::<Fields<'_>>(self.ordered_json()?.get()).ok()?.message
    }

    pub fn custom(&self) -> Option<(Cow<'_, str>, Option<&RawValue>)> {
        if self.kind() != "custom" {
            return None;
        }
        if let Body::Native(EntryData::Custom { custom_type, .. }) = &self.0.body {
            return Some((Cow::Borrowed(custom_type), None));
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Custom<'a> {
            #[serde(borrow)]
            custom_type: Cow<'a, str>,
            #[serde(borrow)]
            data: Option<&'a RawValue>,
        }
        let fields: Custom<'_> = serde_json::from_str(self.ordered_json()?.get()).ok()?;
        Some((fields.custom_type, fields.data))
    }

    pub fn to_raw_map(&self) -> FileEntry {
        let Value::Object(object) = self.to_value() else { unreachable!("stored object") };
        object
    }

    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("serialize stored object")
    }

    pub fn native_message(&self) -> Option<&Message> {
        match &self.0.body {
            Body::Native(EntryData::Message(message)) => Some(message),
            _ => None,
        }
    }

    pub fn matches_message(&self, message: &Message) -> bool {
        if !self.is_message_role(message.role()) {
            return false;
        }
        match (self.native_message(), message) {
            (Some(Message::User(saved)), Message::User(live)) => {
                saved.timestamp == live.timestamp && saved.content == live.content
            }
            (Some(saved), live) => saved == live,
            (None, live) => self.raw_message().is_some_and(|saved| {
                exact_match(saved.get().as_bytes(), live)
                    || serde_json::from_str::<Value>(saved.get())
                        .ok()
                        .zip(serde_json::to_value(live).ok())
                        .is_some_and(|(saved, live)| saved == live)
            }),
        }
    }

    pub fn message(&self) -> Option<Cow<'_, Message>> {
        if self.kind() != "message" {
            return None;
        }
        match self.data() {
            Cow::Borrowed(EntryData::Message(message)) => Some(Cow::Borrowed(message)),
            Cow::Owned(EntryData::Message(message)) => Some(Cow::Owned(message)),
            _ => None,
        }
    }

    pub fn compaction(&self) -> Option<Cow<'_, CompactionEntry>> {
        if self.kind() != "compaction" {
            return None;
        }
        match self.data() {
            Cow::Borrowed(EntryData::Compaction(data)) => Some(Cow::Borrowed(data)),
            Cow::Owned(EntryData::Compaction(data)) => Some(Cow::Owned(data)),
            _ => None,
        }
    }

    pub fn branch_summary(&self) -> Option<Cow<'_, BranchSummaryEntry>> {
        if self.kind() != "branch_summary" {
            return None;
        }
        match self.data() {
            Cow::Borrowed(EntryData::BranchSummary(data)) => Some(Cow::Borrowed(data)),
            Cow::Owned(EntryData::BranchSummary(data)) => Some(Cow::Owned(data)),
            _ => None,
        }
    }

    pub fn message_role(&self) -> Option<Cow<'_, str>> {
        if self.kind() != "message" {
            return None;
        }
        if let Some(message) = self.native_message() {
            return Some(Cow::Borrowed(message.role()));
        }
        #[derive(Deserialize)]
        struct Role<'a> {
            #[serde(borrow)]
            role: Cow<'a, str>,
        }
        Some(serde_json::from_str::<Role<'_>>(self.raw_message()?.get()).ok()?.role)
    }

    pub(super) fn assistant_model(&self) -> Option<(String, String)> {
        if self.kind() != "message" {
            return None;
        }
        if let Some(message) = self.native_message() {
            let Message::Assistant(assistant) = message else { return None };
            return Some((assistant.provider.clone(), assistant.model.clone()));
        }
        #[derive(Deserialize)]
        struct Model<'a> {
            #[serde(borrow)]
            role: Cow<'a, str>,
            #[serde(default)]
            provider: String,
            #[serde(default)]
            model: String,
        }
        let model: Model<'_> = serde_json::from_str(self.raw_message()?.get()).ok()?;
        (model.role == "assistant").then_some((model.provider, model.model))
    }

    pub fn is_message_role(&self, role: &str) -> bool {
        self.message_role().as_deref() == Some(role)
    }

    /// A copy with `key` set (or removed for `None`), keeping its position in the object.
    pub(crate) fn with_field(&self, key: &str, value: Option<Value>) -> Self {
        let mut raw = self.to_raw_map();
        if raw.get(key) == value.as_ref() {
            return self.clone();
        }
        match value {
            Some(value) => {
                raw.insert(key.to_string(), value);
            }
            None => {
                raw.shift_remove(key);
            }
        }
        let mut entry = Self::from_raw(raw);
        Arc::get_mut(&mut entry.0).expect("new entry").encoded_bytes.take();
        entry
    }

    pub(crate) fn with_parent(&self, parent_id: Option<&str>) -> Self {
        if let Body::Native(data) = &self.0.body {
            if self.parent_id() == parent_id {
                return self.clone();
            }
            return Self::native(self.kind(), self.id().into(), parent_id, self.timestamp().into(), data.clone());
        }
        self.with_field("parentId", Some(parent_id.map_or(Value::Null, |id| Value::String(id.into()))))
    }

    pub fn to_json_line(&self) -> String {
        let mut bytes = Vec::new();
        self.write_json_line(&mut bytes).expect("serialize history");
        String::from_utf8(bytes).expect("JSON is UTF-8")
    }

    pub fn encoded_bytes(&self) -> usize {
        *self.0.encoded_bytes.get_or_init(|| {
            let mut writer = Count { writer: io::sink(), bytes: 0 };
            self.write_body(&mut writer).expect("measure history");
            writer.bytes
        })
    }

    /// Serialize the stored object without a trailing line feed.
    pub fn write_json_line(&self, writer: &mut impl Write) -> io::Result<()> {
        if self.0.encoded_bytes.get().is_some() {
            return self.write_body(writer);
        }
        let mut writer = Count { writer, bytes: 0 };
        self.write_body(&mut writer)?;
        let _ = self.0.encoded_bytes.set(writer.bytes);
        Ok(())
    }

    fn write_body(&self, writer: &mut impl Write) -> io::Result<()> {
        match &self.0.body {
            Body::OrderedRaw(raw) => writer.write_all(raw.get().as_bytes()),
            Body::Native(_) => serde_json::to_writer(writer, self).map_err(io::Error::from),
        }
    }
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other)
            || (self.id() == other.id() && self.kind() == other.kind() && self.to_value() == other.to_value())
    }
}

impl Eq for Entry {}

impl Serialize for Entry {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match &self.0.body {
            Body::OrderedRaw(raw) => raw.serialize(serializer),
            Body::Native(data) => Native {
                kind: self.kind(),
                id: self.id(),
                parent_id: self.parent_id(),
                timestamp: self.timestamp(),
                data,
            }
            .serialize(serializer),
        }
    }
}

struct Native<'a> {
    kind: &'a str,
    id: &'a str,
    parent_id: Option<&'a str>,
    timestamp: &'a str,
    data: &'a EntryData,
}

impl Serialize for Native<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        let link = |map: &mut S::SerializeMap| {
            map.serialize_entry("id", self.id)?;
            map.serialize_entry("parentId", &self.parent_id)?;
            map.serialize_entry("timestamp", self.timestamp)
        };
        macro_rules! optional {
            ($key:literal, $value:expr) => {
                if let Some(value) = &$value {
                    map.serialize_entry($key, value)?;
                }
            };
        }
        map.serialize_entry("type", self.kind)?;
        match self.data {
            EntryData::Custom { custom_type, data } => {
                map.serialize_entry("customType", custom_type)?;
                optional!("data", data);
                link(&mut map)?;
                return map.end();
            }
            EntryData::CustomMessage(custom) => {
                map.serialize_entry("customType", &custom.custom_type)?;
                map.serialize_entry("content", &custom.content)?;
                map.serialize_entry("display", &custom.display)?;
                optional!("details", custom.details);
                link(&mut map)?;
                return map.end();
            }
            _ => {}
        }
        link(&mut map)?;
        match self.data {
            EntryData::Message(message) => map.serialize_entry("message", message)?,
            EntryData::ThinkingLevelChange { thinking_level } => {
                map.serialize_entry("thinkingLevel", thinking_level)?
            }
            EntryData::ModelChange { provider, model_id } => {
                map.serialize_entry("provider", provider)?;
                map.serialize_entry("modelId", model_id)?;
            }
            EntryData::Usage(data) => {
                map.serialize_entry("kind", &data.kind)?;
                map.serialize_entry("provider", &data.provider)?;
                map.serialize_entry("model", &data.model)?;
                map.serialize_entry("usage", &data.usage)?;
                optional!("note", data.note);
            }
            EntryData::Compaction(data) => {
                map.serialize_entry("summary", &data.summary)?;
                map.serialize_entry("firstKeptEntryId", &data.first_kept_entry_id)?;
                map.serialize_entry("tokensBefore", &data.tokens_before)?;
                optional!("details", data.details);
                optional!("usage", data.usage);
                optional!("fromHook", data.from_hook);
                if let Some(system) = &data.system_message {
                    map.serialize_entry("systemMessage", system)?;
                }
            }
            EntryData::BranchSummary(data) => {
                map.serialize_entry("fromId", &data.from_id)?;
                map.serialize_entry("summary", &data.summary)?;
                optional!("details", data.details);
                optional!("usage", data.usage);
                optional!("fromHook", data.from_hook);
            }
            EntryData::ContextEdit { target_id, replacement } => {
                map.serialize_entry("targetId", target_id)?;
                map.serialize_entry("replacement", replacement)?;
            }
            EntryData::Label { target_id, label } => {
                map.serialize_entry("targetId", target_id)?;
                optional!("label", label);
            }
            EntryData::SessionInfo { name } => optional!("name", name),
            _ => unreachable!("native entry kind"),
        }
        map.end()
    }
}

struct ExactMatchWriter<'a> {
    remaining: Option<&'a [u8]>,
}

impl Write for ExactMatchWriter<'_> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.remaining = self.remaining.and_then(|remaining| remaining.strip_prefix(data));
        self.remaining.map(|_| data.len()).ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) fn exact_match(bytes: &[u8], value: &impl Serialize) -> bool {
    let mut writer = ExactMatchWriter { remaining: Some(bytes) };
    serde_json::to_writer(&mut writer, value).is_ok() && writer.remaining == Some(b"")
}

struct Count<W> {
    writer: W,
    bytes: usize,
}

impl<W: Write> Write for Count<W> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let written = self.writer.write(data)?;
        self.bytes += written;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

fn parse_data(kind: &str, raw: &FileEntry) -> EntryData {
    let parse = || -> Option<EntryData> {
        Some(match kind {
            "message" => EntryData::Message(parse_message_lenient(raw.get("message")?)?),
            "thinking_level_change" => {
                EntryData::ThinkingLevelChange { thinking_level: string_field(raw, "thinkingLevel")? }
            }
            "model_change" => EntryData::ModelChange {
                provider: string_field(raw, "provider")?,
                model_id: string_field(raw, "modelId")?,
            },
            "usage" => EntryData::Usage(UsageEntry::deserialize(raw).ok()?),
            "compaction" => EntryData::Compaction(CompactionEntry::deserialize(raw).ok()?),
            "branch_summary" => EntryData::BranchSummary(BranchSummaryEntry::deserialize(raw).ok()?),
            "custom" => EntryData::Custom {
                custom_type: string_field(raw, "customType").unwrap_or_default(),
                data: raw.get("data").cloned(),
            },
            "custom_message" => EntryData::CustomMessage(CustomMessageEntry::deserialize(raw).ok()?),
            "context_edit" => EntryData::ContextEdit {
                target_id: string_field(raw, "targetId")?,
                replacement: raw.get("replacement").and_then(|value| ContextReplacement::deserialize(value).ok()),
            },
            "label" => {
                EntryData::Label { target_id: string_field(raw, "targetId")?, label: string_field(raw, "label") }
            }
            "session_info" => EntryData::SessionInfo { name: string_field(raw, "name") },
            _ => return None,
        })
    };
    parse().unwrap_or(EntryData::Other)
}

fn decode_data(kind: &str, line: &str) -> Option<EntryData> {
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Fields {
        thinking_level: Option<String>,
        provider: Option<String>,
        model_id: Option<String>,
        custom_type: Option<String>,
        target_id: Option<String>,
        replacement: Option<ContextReplacement>,
        label: Option<String>,
        name: Option<String>,
    }

    let fields = || serde_json::from_str::<Fields>(line).ok();
    Some(match kind {
        "thinking_level_change" => EntryData::ThinkingLevelChange { thinking_level: fields()?.thinking_level? },
        "model_change" => {
            let fields = fields()?;
            EntryData::ModelChange { provider: fields.provider?, model_id: fields.model_id? }
        }
        "usage" => EntryData::Usage(serde_json::from_str(line).ok()?),
        "compaction" => EntryData::Compaction(serde_json::from_str(line).ok()?),
        "branch_summary" => EntryData::BranchSummary(serde_json::from_str(line).ok()?),
        "custom" => EntryData::Custom { custom_type: fields()?.custom_type?, data: None },
        "custom_message" => EntryData::CustomMessage(serde_json::from_str(line).ok()?),
        "context_edit" => {
            let fields = fields()?;
            EntryData::ContextEdit { target_id: fields.target_id?, replacement: fields.replacement }
        }
        "label" => {
            let fields = fields()?;
            EntryData::Label { target_id: fields.target_id?, label: fields.label }
        }
        "session_info" => EntryData::SessionInfo { name: fields()?.name },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use std::cell::RefCell;
    use std::collections::HashMap;

    use super::*;

    thread_local! {
        static DECODES: RefCell<HashMap<String, usize>> = RefCell::default();
    }

    pub(super) fn note_decode(id: &str) {
        DECODES.with_borrow_mut(|counts| *counts.entry(id.into()).or_default() += 1);
    }

    #[test]
    fn keeps_raw_objects_and_parses_typed_views() {
        let line = r#"{"type":"custom","customType":"ext","data":{"n":1},"id":"a1","parentId":null,"timestamp":"t"}"#;
        let entry = Entry::from_raw(serde_json::from_str(line).unwrap());
        assert_eq!(entry.to_json_line(), line);
        assert_eq!((entry.kind(), entry.id(), entry.parent_id()), ("custom", "a1", None));
        assert!(matches!(entry.data().as_ref(), EntryData::Custom { custom_type, .. } if custom_type == "ext"));
        let moved = entry.with_parent(Some("p"));
        assert_eq!(moved.to_json_line(), line.replace("null", "\"p\""));
        assert!(entry.ordered_json().is_some());
        assert!(entry.ptr_eq(&entry.clone()));
        assert!(entry.ptr_eq(&entry.with_parent(None)));
        assert!(entry.ptr_eq(&entry.with_field("absent", None)));
        let explicit = entry.with_field("absent", Some(Value::Null));
        assert!(!entry.ptr_eq(&explicit));
        assert!(explicit.ptr_eq(&explicit.with_field("absent", Some(Value::Null))));
        assert!(!entry.ptr_eq(&moved));
        assert_eq!(entry.encoded_bytes(), line.len());
        assert_eq!(serde_json::to_string(&entry).unwrap(), line);
        assert_eq!(moved.to_raw_map()["parentId"], "p");
    }

    #[test]
    fn serialization_must_match_every_byte_including_exhaustion() {
        let canonical = r#"{"type":"message","id":"user","parentId":null,"timestamp":"t","message":{"role":"user","content":"hi","timestamp":1}}"#;
        let entry = Entry::from_json_line(canonical).unwrap();
        assert!(entry.native_message().is_some());
        assert!(entry.ptr_eq(&entry.with_parent(None)));
        for line in [
            format!("{canonical} "),
            canonical.replace("\"type\":\"message\",\"id\":\"user\"", "\"id\":\"user\",\"type\":\"message\""),
            canonical.replace("\"content\":", "\"extra\":true,\"content\":"),
            canonical.replace("\"hi\"", "null"),
        ] {
            let entry = Entry::from_json_line(&line).unwrap();
            assert!(entry.ordered_json().is_some());
            assert!(entry.message().is_some());
            assert_eq!(entry.to_json_line(), line.trim());
            assert_eq!(entry.encoded_bytes(), line.trim().len());
        }
        let compaction = Entry::from_json_line(r#"{"type":"compaction","id":"c","parentId":null,"timestamp":"t","summary":"summary","firstKeptEntryId":"a","tokensBefore":1}"#).unwrap();
        assert!(compaction.ptr_eq(&compaction.with_field("firstKeptEntryId", Some(json!("a")))));
        let changed = compaction.with_field("firstKeptEntryId", Some(json!("b")));
        assert!(!compaction.ptr_eq(&changed));
        assert_eq!(changed.compaction().unwrap().first_kept_entry_id, "b");
    }

    #[test]
    fn only_active_integer_cost_messages_decode_for_context_across_appends() {
        use crate::session::SessionManager;
        use pi_ai::{AssistantMessage, providers::faux::FauxProvider};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ts.jsonl");
        let mut manager = SessionManager::open(&path).unwrap();
        let assistant: Message = AssistantMessage::empty(&FauxProvider::model()).into();
        let inactive = manager.append_message(&assistant).unwrap();
        manager.reset_leaf();
        let cold = manager.append_message(&assistant).unwrap();
        let active = manager.append_message(&assistant).unwrap();
        manager.append_compaction("summary", &active, 1).unwrap();
        let floating = manager.to_jsonl();
        assert!(Entry::from_json_line(floating.lines().nth(1).unwrap()).unwrap().native_message().is_some());
        std::fs::write(&path, floating.replace(":0.0", ":0")).unwrap();
        let mut manager = SessionManager::open(&path).unwrap();
        for id in [&inactive, &cold, &active] {
            assert!(manager.get_entry(id).unwrap().ordered_json().is_some());
        }
        let first = manager.build_session_context();
        for round in 0..3 {
            manager.append_message(&Message::user(format!("round {round}"))).unwrap();
            let later = manager.build_session_context();
            let (Message::Assistant(before), Message::Assistant(after)) = (&first.messages[1], &later.messages[1])
            else {
                panic!("assistant")
            };
            assert!(Arc::ptr_eq(before, after));
        }
        DECODES.with_borrow(|counts| {
            assert_eq!(counts.get(&inactive), None);
            assert_eq!(counts.get(&cold), None);
            assert_eq!(counts.get(&active), Some(&1));
        });
    }

    #[test]
    fn tolerates_legacy_messages() {
        let value = json!({"role": "assistant", "content": [{"type": "text", "text": "hi"}], "api": "test",
            "provider": "test", "model": "test", "usage": {"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0},
            "stopReason": "stop", "timestamp": 2});
        let Some(Message::Assistant(message)) = parse_message_lenient(&value) else { panic!("assistant") };
        assert_eq!(message.usage.input, 1);
        let user = parse_message_lenient(&json!({"role": "user", "content": null})).unwrap();
        assert!(matches!(user, Message::User(user) if user.content.text().is_empty()));
    }
}
