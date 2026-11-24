//! Message, content, usage, tool, and model types. JSON field names match the
//! TypeScript pi packages, so session files stay readable by both implementations.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::de::Error as _;
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub type JsonObject = Map<String, Value>;

/// Milliseconds since the Unix epoch.
pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: JsonObject,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    /// OpenAI Responses namespace for namespaced or dynamically loaded tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// Set when the finished arguments text was not a JSON object; `arguments` is then empty. Holds the
    /// reason and the text the model sent, and tool argument validation fails with it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments_error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum ContentBlock {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text_signature: Option<String>,
    },
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thinking_signature: Option<String>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        redacted: bool,
    },
    Image {
        data: String,
        mime_type: String,
    },
    ToolCall(ToolCall),
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into(), text_signature: None }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text, .. } => Some(text),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Cost {
    /// Catalog-derived input cost.
    pub input: f64,
    /// Catalog-derived output cost.
    pub output: f64,
    /// Catalog-derived cache-read cost.
    pub cache_read: f64,
    /// Catalog-derived cache-write cost.
    pub cache_write: f64,
    /// Best available total: provider-reported billed cost when supported, otherwise the catalog estimate.
    pub total: f64,
    /// True when `total` is the billed cost reported by the provider. False when it is a catalog
    /// estimate; absent also means false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_from_provider: Option<bool>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    /// Part of `cache_write` written with 1h retention (Anthropic only).
    #[serde(default, rename = "cacheWrite1h", skip_serializing_if = "Option::is_none")]
    pub cache_write_1h: Option<u64>,
    /// Reasoning tokens, a subset of `output`, when the provider reports them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
    pub total_tokens: u64,
    pub cost: Cost,
}

impl Usage {
    /// Fill `cost` from per-million-token model prices. The highest tier whose threshold the
    /// total input exceeds prices the whole request; 1h cache writes cost twice the input rate.
    pub fn apply_cost(&mut self, cost: &ModelCost) {
        let input_tokens = self.input + self.cache_read + self.cache_write;
        let (mut rates, mut threshold) = ((cost.input, cost.output, cost.cache_read, cost.cache_write), None);
        for tier in &cost.tiers {
            if input_tokens > tier.input_tokens_above && threshold.is_none_or(|above| tier.input_tokens_above > above) {
                rates = (tier.input, tier.output, tier.cache_read, tier.cache_write);
                threshold = Some(tier.input_tokens_above);
            }
        }
        let (input, output, cache_read, cache_write) = rates;
        let long_write = self.cache_write_1h.unwrap_or_default().min(self.cache_write);
        let short_write = self.cache_write - long_write;
        let per_million = |tokens: u64, price: f64| tokens as f64 * price / 1_000_000.0;
        self.cost.input = per_million(self.input, input);
        self.cost.output = per_million(self.output, output);
        self.cost.cache_read = per_million(self.cache_read, cache_read);
        self.cost.cache_write = per_million(short_write, cache_write) + per_million(long_write, input * 2.0);
        self.cost.total = self.cost.input + self.cost.output + self.cost.cache_read + self.cost.cache_write;
        self.cost.total_from_provider = Some(false);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    Pending,
    #[default]
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
    Deferred,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl UserContent {
    /// Text parts joined with newlines; images are omitted.
    pub fn text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Blocks(blocks) => blocks.iter().filter_map(ContentBlock::as_text).collect::<Vec<_>>().join("\n"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessage {
    /// Host input identity, carried through queues and events but never sent to providers or saved in message content.
    #[serde(skip)]
    pub input_id: Option<String>,
    pub content: UserContent,
    pub timestamp: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    pub content: Vec<ContentBlock>,
    pub api: String,
    pub provider: String,
    pub model: String,
    /// Concrete model reported by the provider, when reported. May equal the requested `model`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Provider-native effort level used for this response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_thinking_level: Option<String>,
    /// Pi thinking level the agent loop requested for this response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<ThinkingLevel>,
    /// Redacted provider and runtime diagnostics for failures and recoveries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<Value>,
    pub usage: Usage,
    pub stop_reason: StopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred: Option<DeferredHandle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_stop_reason: Option<String>,
    /// Whether the provider reported that the model ended its turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_turn: Option<bool>,
    pub timestamp: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

/// Handle for a response that finishes later (batch or background APIs).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeferredHandle {
    pub provider: String,
    pub model_id: String,
    pub api: String,
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub poll_after_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl AssistantMessage {
    /// An empty pending response for `model`.
    pub fn empty(model: &Model) -> Self {
        Self {
            content: Vec::new(),
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model: None,
            response_id: None,
            provider_thinking_level: None,
            thinking_level: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::Pending,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: now_ms(),
            duration_ms: None,
        }
    }

    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.content.iter().filter_map(|block| match block {
            ContentBlock::ToolCall(call) => Some(call),
            _ => None,
        })
    }

    /// Text blocks joined without separators.
    pub fn text(&self) -> String {
        self.content.iter().filter_map(ContentBlock::as_text).collect()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<ContentBlock>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    /// Usage of the tool execution itself, outside the main context accounting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Calls this tool made to other tools. Kept in the session, not sent to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nested_calls: Option<NestedToolCalls>,
    pub is_error: bool,
    pub timestamp: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

/// A tool call another tool made while it ran, for example from a codemode script.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NestedToolCallRecord {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<JsonObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments_bytes: Option<u64>,
    /// `ok`, `error`, or `unfinished`.
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NestedToolCalls {
    pub calls: Vec<NestedToolCallRecord>,
    /// False when calls were dropped, arguments omitted, or calls had not finished.
    pub complete: bool,
}

/// System instructions and tool declarations at one point in the transcript. The leading
/// one is the system prompt; later ones add instructions, patch named `sections`
/// (`null` removes one), and change tools. See [`crate::transcript`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemMessage {
    pub content: UserContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sections: Option<JsonObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_added: Option<Vec<Tool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_removed: Option<Vec<ToolReference>>,
    pub timestamp: i64,
}

/// A message with an app-defined role, such as the coding agent's `bashExecution`,
/// `custom`, `branchSummary`, and `compactionSummary`. Providers never see these:
/// the agent loop converts them (`convert_to_llm`) and the registry drops leftovers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CustomMessage {
    pub role: String,
    /// Every field except `role`, including `timestamp`.
    pub fields: JsonObject,
}

impl CustomMessage {
    pub fn new(role: impl Into<String>, fields: JsonObject) -> Self {
        Self { role: role.into(), fields }
    }

    pub fn timestamp(&self) -> i64 {
        self.fields.get("timestamp").and_then(Value::as_i64).unwrap_or_default()
    }

    pub fn str_field(&self, key: &str) -> Option<&str> {
        self.fields.get(key).and_then(Value::as_str)
    }
}

/// An immutable message shared by the session, execution, and provider layers.
#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    System(Arc<SystemMessage>),
    User(Arc<UserMessage>),
    Assistant(Arc<AssistantMessage>),
    ToolResult(Arc<ToolResultMessage>),
    Custom(Arc<CustomMessage>),
}

macro_rules! message_from {
    ($($variant:ident($body:ty)),+ $(,)?) => {$(
        impl From<$body> for Message {
            fn from(message: $body) -> Self {
                Self::$variant(Arc::new(message))
            }
        }

        impl From<Arc<$body>> for Message {
            fn from(message: Arc<$body>) -> Self {
                Self::$variant(message)
            }
        }
    )+};
}

message_from!(
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
    Custom(CustomMessage)
);

#[cfg(test)]
#[test]
fn message_is_two_words() {
    assert_eq!(std::mem::size_of::<Message>(), 2 * std::mem::size_of::<usize>());
}

#[derive(Serialize)]
#[serde(tag = "role", rename_all = "camelCase")]
enum KnownMessageRef<'a> {
    System(&'a SystemMessage),
    User(&'a UserMessage),
    Assistant(&'a AssistantMessage),
    ToolResult(&'a ToolResultMessage),
}

enum KnownRole {
    System,
    User,
    Assistant,
    ToolResult,
}

impl KnownRole {
    fn of(role: &str) -> Option<Self> {
        Some(match role {
            "system" => Self::System,
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "toolResult" => Self::ToolResult,
            _ => return None,
        })
    }

    fn decode<'de, D: serde::Deserializer<'de>>(self, object: D) -> Result<Message, D::Error> {
        match self {
            Self::System => SystemMessage::deserialize(object).map(Message::from),
            Self::User => UserMessage::deserialize(object).map(Message::from),
            Self::Assistant => AssistantMessage::deserialize(object).map(Message::from),
            Self::ToolResult => ToolResultMessage::deserialize(object).map(Message::from),
        }
    }
}

fn message_role(object: &Map<String, Value>) -> Result<&str, serde_json::Error> {
    object.get("role").and_then(Value::as_str).ok_or_else(|| serde_json::Error::missing_field("role"))
}

impl Serialize for Message {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let known = match self {
            Self::System(message) => KnownMessageRef::System(message),
            Self::User(message) => KnownMessageRef::User(message),
            Self::Assistant(message) => KnownMessageRef::Assistant(message),
            Self::ToolResult(message) => KnownMessageRef::ToolResult(message),
            Self::Custom(custom) => {
                let len = custom.fields.len() + usize::from(!custom.fields.contains_key("role"));
                let mut object = serializer.serialize_map(Some(len))?;
                object.serialize_entry("role", &custom.role)?;
                for (key, value) in custom.fields.iter().filter(|(key, _)| *key != "role") {
                    object.serialize_entry(key, value)?;
                }
                return object.end();
            }
        };
        known.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Message {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut object = Map::deserialize(deserializer)?;
        if let Some(known) = KnownRole::of(message_role(&object).map_err(D::Error::custom)?) {
            return known.decode(object).map_err(D::Error::custom);
        }
        let Some(Value::String(role)) = object.shift_remove("role") else {
            return Err(D::Error::missing_field("role"));
        };
        Ok(CustomMessage { role, fields: object }.into())
    }
}

impl Message {
    pub fn from_json(text: &str) -> serde_json::Result<Self> {
        #[derive(Deserialize)]
        struct Envelope<'a> {
            #[serde(borrow)]
            role: std::borrow::Cow<'a, str>,
        }
        match KnownRole::of(&serde_json::from_str::<Envelope<'_>>(text)?.role) {
            Some(known) => {
                let mut deserializer = serde_json::Deserializer::from_str(text);
                known.decode(&mut deserializer)
            }
            None => serde_json::from_str(text),
        }
    }

    /// Decode a borrowed JSON object without copying a temporary value tree.
    pub fn from_value(value: &Value) -> Result<Self, serde_json::Error> {
        let object = value.as_object().ok_or_else(|| serde_json::Error::custom("expected a message object"))?;
        let role = message_role(object)?;
        match KnownRole::of(role) {
            Some(known) => known.decode(object),
            None => {
                let fields = object
                    .iter()
                    .filter(|(key, _)| *key != "role")
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                Ok(CustomMessage::new(role, fields).into())
            }
        }
    }

    pub fn user(text: impl Into<String>) -> Self {
        UserMessage { input_id: None, content: UserContent::Text(text.into()), timestamp: now_ms() }.into()
    }

    pub fn role(&self) -> &str {
        match self {
            Self::System(_) => "system",
            Self::User(_) => "user",
            Self::Assistant(_) => "assistant",
            Self::ToolResult(_) => "toolResult",
            Self::Custom(custom) => &custom.role,
        }
    }
}

/// A tool declaration sent to the model. `parameters` is a JSON Schema object.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    /// Grammar-constrained sampling config, or `false` to opt out.
    #[serde(default, rename = "constrainedSampling", skip_serializing_if = "Option::is_none")]
    pub constrained_sampling: Option<Value>,
}

impl Tool {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self { name: name.into(), description: description.into(), parameters, constrained_sampling: None }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolReference {
    pub name: String,
}

/// Request input for a provider.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    #[default]
    Off,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ThinkingLevel {
    pub const ALL: [ThinkingLevel; 7] =
        [Self::Off, Self::Minimal, Self::Low, Self::Medium, Self::High, Self::Xhigh, Self::Max];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

impl std::str::FromStr for ThinkingLevel {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|level| level.as_str() == value)
            .ok_or_else(|| format!("Unknown thinking level: {value}"))
    }
}

/// Prices in dollars per million tokens.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
    #[serde(default)]
    pub cache_read: f64,
    #[serde(default)]
    pub cache_write: f64,
    /// Request-wide tiers; the highest matching input threshold prices the whole request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiers: Vec<ModelCostTier>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostTier {
    pub input_tokens_above: u64,
    pub input: f64,
    pub output: f64,
    #[serde(default)]
    pub cache_read: f64,
    #[serde(default)]
    pub cache_write: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageResizeOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_height: Option<u32>,
    /// Maximum base64 payload size in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jpeg_quality: Option<u8>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelImageInputLimits {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resize: Option<ImageResizeOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_per_message: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_per_request: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInputLimits {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_request_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<ModelImageInputLimits>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    /// Wire protocol, for example `openai-completions` or `anthropic-messages`.
    pub api: String,
    pub provider: String,
    pub base_url: String,
    /// `chat` (the default when absent), `image`, or `classifier`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub model_type: Option<String>,
    #[serde(default)]
    pub reasoning: bool,
    /// Pi thinking level to provider value; `null` marks a level unsupported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<BTreeMap<String, Option<String>>>,
    /// Prompt cache lifetime in seconds per retention tier (`short`, `long`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache: Option<BTreeMap<String, f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_limits: Option<ModelInputLimits>,
    /// Output modalities of image models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Vec<String>>,
    /// Input modalities: `text`, `image`.
    #[serde(default = "default_input")]
    pub input: Vec<String>,
    #[serde(default)]
    pub cost: ModelCost,
    #[serde(default)]
    pub context_window: u64,
    #[serde(default)]
    pub max_tokens: u64,
    /// Default sampling parameters; request options override keys.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<JsonObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params_by_thinking_level: Option<BTreeMap<String, JsonObject>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    /// Provider-specific compatibility settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<Value>,
}

fn default_input() -> Vec<String> {
    vec!["text".into()]
}

impl Model {
    pub fn supports_images(&self) -> bool {
        self.input.iter().any(|input| input == "image")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_messages_keep_field_order_and_a_single_role() {
        let input = r#"{"command":"ls","output":"a","role":"bashExecution","exitCode":0,"timestamp":4}"#;
        let expected = r#"{"role":"bashExecution","command":"ls","output":"a","exitCode":0,"timestamp":4}"#;
        let value = serde_json::from_str(input).unwrap();
        let mut message: Message = serde_json::from_str(input).unwrap();
        assert_eq!(Message::from_value(&value).unwrap(), message);
        assert_eq!(serde_json::to_string(&message).unwrap(), expected);
        let Message::Custom(custom) = &mut message else { panic!("expected custom") };
        Arc::make_mut(custom).fields.insert("role".into(), Value::String("ignored".into()));
        assert_eq!(serde_json::to_string(&message).unwrap(), expected);
    }

    #[test]
    fn known_message_codecs_agree_and_preserve_json_shape() {
        for json in [
            r#"{"role":"system","content":"prompt","sections":{"one":"a","two":"b"},"timestamp":1}"#,
            r#"{"role":"user","content":"hello","timestamp":2}"#,
            r#"{"role":"assistant","content":[],"api":"test","provider":"test","model":"m","usage":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0.0,"output":0.0,"cacheRead":0.0,"cacheWrite":0.0,"total":0.0}},"stopReason":"stop","timestamp":3}"#,
            r#"{"role":"toolResult","toolCallId":"c1","toolName":"read","content":[],"isError":false,"timestamp":4}"#,
        ] {
            let value = serde_json::from_str(json).unwrap();
            let message: Message = serde_json::from_str(json).unwrap();
            assert_eq!(Message::from_value(&value).unwrap(), message);
            assert_eq!(serde_json::to_string(&message).unwrap(), json);
        }
    }

    #[test]
    fn messages_round_trip_with_upstream_field_names() {
        let json = r#"{"role":"assistant","content":[{"type":"text","text":"hi"},{"type":"thinking","thinking":"t","thinkingSignature":"s"},{"type":"toolCall","id":"c1","name":"read","arguments":{"path":"a"}}],"api":"anthropic-messages","provider":"anthropic","model":"m","usage":{"input":1,"output":2,"cacheRead":0,"cacheWrite":0,"totalTokens":3,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},"stopReason":"toolUse","timestamp":1}"#;
        let message: Message = serde_json::from_str(json).unwrap();
        let Message::Assistant(assistant) = &message else { panic!("expected assistant") };
        assert_eq!(assistant.stop_reason, StopReason::ToolUse);
        assert_eq!(assistant.tool_calls().next().unwrap().name, "read");
        let encoded = serde_json::to_value(&message).unwrap();
        assert_eq!(encoded["content"][2]["type"], "toolCall");
        assert_eq!(encoded["content"][1]["thinkingSignature"], "s");
        assert_eq!(encoded["stopReason"], "toolUse");
        let user: Message = serde_json::from_str(r#"{"role":"user","content":"hello","timestamp":2}"#).unwrap();
        assert_eq!(serde_json::to_value(&user).unwrap()["content"], "hello");
        let result: Message = serde_json::from_str(
            r#"{"role":"toolResult","toolCallId":"c1","toolName":"read","content":[{"type":"image","data":"AA==","mimeType":"image/png"}],"isError":false,"timestamp":3}"#,
        )
        .unwrap();
        assert!(matches!(result, Message::ToolResult(result) if !result.is_error));
        let custom: Message =
            serde_json::from_str(r#"{"role":"bashExecution","command":"ls","output":"a","exitCode":0,"timestamp":4}"#)
                .unwrap();
        let Message::Custom(bash) = &custom else { panic!("expected custom") };
        assert_eq!((bash.role.as_str(), bash.str_field("command"), bash.timestamp()), ("bashExecution", Some("ls"), 4));
        let encoded = serde_json::to_value(&custom).unwrap();
        assert_eq!((encoded["role"].as_str(), encoded["exitCode"].as_i64()), (Some("bashExecution"), Some(0)));
    }

    #[test]
    fn usage_cost_uses_per_million_prices() {
        let mut usage = Usage { input: 1_000_000, output: 500_000, ..Usage::default() };
        usage.apply_cost(&ModelCost {
            input: 3.0,
            output: 15.0,
            cache_read: 0.3,
            cache_write: 3.75,
            ..Default::default()
        });
        assert_eq!(usage.cost.input, 3.0);
        assert_eq!(usage.cost.output, 7.5);
        assert_eq!(usage.cost.total, 10.5);
    }

    #[test]
    fn usage_cost_applies_tiers_and_long_cache_writes() {
        let cost = ModelCost {
            input: 3.0,
            output: 15.0,
            cache_read: 0.3,
            cache_write: 3.75,
            tiers: vec![
                ModelCostTier {
                    input_tokens_above: 200_000,
                    input: 6.0,
                    output: 22.5,
                    cache_read: 0.6,
                    cache_write: 7.5,
                },
                ModelCostTier {
                    input_tokens_above: 100_000,
                    input: 4.0,
                    output: 20.0,
                    cache_read: 0.4,
                    cache_write: 5.0,
                },
            ],
        };
        let mut usage = Usage { input: 150_000, cache_read: 100_000, output: 1_000_000, ..Usage::default() };
        usage.apply_cost(&cost);
        assert_eq!((usage.cost.input, usage.cost.output), (0.9, 22.5));
        let mut usage = Usage { input: 200_000, output: 0, ..Usage::default() };
        usage.apply_cost(&cost);
        assert_eq!(usage.cost.input, 0.8);
        let mut usage = Usage { cache_write: 1_000_000, cache_write_1h: Some(400_000), ..Usage::default() };
        usage.apply_cost(&ModelCost { input: 3.0, cache_write: 3.75, ..Default::default() });
        assert!((usage.cost.cache_write - (0.6 * 3.75 + 0.4 * 6.0)).abs() < 1e-9);
    }

    #[test]
    fn catalog_derived_totals_are_not_provider_reported() {
        use serde_json::json;
        let mut usage = Usage { input: 1_000_000, total_tokens: 1_000_000, ..Usage::default() };
        usage.cost = Cost { total: 9.0, total_from_provider: Some(true), ..Cost::default() };
        usage.apply_cost(&ModelCost { input: 1.25, ..Default::default() });
        assert_eq!(usage.cost.total_from_provider, Some(false));
        assert_eq!(serde_json::to_value(usage.cost).unwrap()["totalFromProvider"], json!(false));
        let legacy: Cost = serde_json::from_value(json!({"input": 0, "output": 0, "cacheRead": 0,
            "cacheWrite": 0, "total": 0}))
        .unwrap();
        assert_eq!(legacy.total_from_provider, None);
        assert!(serde_json::to_value(legacy).unwrap().get("totalFromProvider").is_none());
    }
}
