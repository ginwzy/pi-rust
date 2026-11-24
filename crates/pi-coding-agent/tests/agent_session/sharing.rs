mod lifecycle;
mod streaming;

use std::sync::{Arc, Mutex};

use pi_agent::StreamFn;
use pi_ai::{
    AssistantMessage, ContentBlock, Context, Message, Model, StopReason, SystemMessage, ToolCall, ToolResultMessage,
    UserContent,
};
use pi_coding_agent::messages::convert_to_llm;
use pi_coding_agent::session::SessionManager;
use serde_json::{Value, json};

use crate::harness::{Options, event_order, harness, open_session_with, record};
use crate::http_mock::{Reply, Server};

pub(super) fn same_message(expected: &Message, actual: &Message) -> bool {
    match (expected, actual) {
        (Message::System(expected), Message::System(actual)) => Arc::ptr_eq(expected, actual),
        (Message::User(expected), Message::User(actual)) => Arc::ptr_eq(expected, actual),
        (Message::Assistant(expected), Message::Assistant(actual)) => Arc::ptr_eq(expected, actual),
        (Message::ToolResult(expected), Message::ToolResult(actual)) => Arc::ptr_eq(expected, actual),
        (Message::Custom(expected), Message::Custom(actual)) => Arc::ptr_eq(expected, actual),
        _ => false,
    }
}

fn assert_prefix(history: &[Message], actual: &[Message]) {
    assert!(actual.len() >= history.len());
    for (index, (expected, actual)) in history.iter().zip(actual).enumerate() {
        assert!(same_message(expected, actual), "copied {} at {index}", expected.role());
    }
}

fn seed(mut manager: SessionManager, model: &Model) -> SessionManager {
    let mut call = AssistantMessage::empty(model);
    call.stop_reason = StopReason::ToolUse;
    call.content = vec![ContentBlock::ToolCall(ToolCall {
        id: "saved-call".into(),
        name: "read".into(),
        arguments: json!({"path":"old.txt"}).as_object().unwrap().clone(),
        ..ToolCall::default()
    })];
    let mut answer = AssistantMessage::empty(model);
    answer.content = vec![ContentBlock::text("saved answer")];
    let mut ts_answer = AssistantMessage::empty(model);
    ts_answer.content = vec![ContentBlock::text("saved TS answer")];
    for message in [
        SystemMessage {
            content: UserContent::Text("saved instructions".into()),
            sections: None,
            tools_added: None,
            tools_removed: None,
            timestamp: 1,
        }
        .into(),
        Message::user("saved question"),
        call.into(),
        ToolResultMessage {
            tool_call_id: "saved-call".into(),
            tool_name: "read".into(),
            content: vec![ContentBlock::text("saved result")],
            details: Some(json!({"ordered": {"z":1,"a":2}})),
            usage: None,
            nested_calls: None,
            is_error: false,
            timestamp: 2,
            duration_ms: None,
        }
        .into(),
        answer.into(),
        Message::user("saved TS question"),
        ts_answer.into(),
    ] {
        manager.append_message(&message).unwrap();
    }
    let path = manager.session_file().unwrap().to_path_buf();
    let wire = manager.to_jsonl();
    let (prefix, last) = wire.trim_end_matches('\n').rsplit_once('\n').unwrap();
    assert!(last.contains(":0.0"));
    std::fs::write(&path, format!("{prefix}\n{}\n", last.replace(":0.0", ":0"))).unwrap();
    let manager = SessionManager::open(&path).unwrap();
    let (fallback, native) = manager.entries().split_last().unwrap();
    assert!(native.iter().all(|entry| entry.native_message().is_some()));
    assert!(fallback.ordered_json().is_some() && fallback.native_message().is_none());
    manager
}

fn sse(delta: Value, finish: &str) -> Reply {
    let chunk = |delta: Value, finish: Value| {
        json!({
            "id":"offline", "object":"chat.completion.chunk", "created":1, "model":"wire",
            "choices":[{"index":0,"delta":delta,"finish_reason":finish}]
        })
    };
    Reply {
        status: 200,
        content_type: "text/event-stream",
        body: format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            chunk(delta, Value::Null),
            chunk(json!({}), json!(finish))
        )
        .into_bytes(),
    }
}

#[tokio::test]
async fn unchanged_native_and_integer_cost_history_reaches_every_provider_request_by_identity() {
    let server = Server::start(|request| {
        let body = request.json();
        assert_eq!(request.path(), "/v1/chat/completions");
        let last = body["messages"].as_array().unwrap().last().unwrap();
        match last["role"].as_str().unwrap() {
            "tool" => sse(json!({"role":"assistant","content":"tool continuation"}), "stop"),
            "user" if last["content"].to_string().contains("read current") => sse(
                json!({
                    "role":"assistant", "tool_calls":[{"index":0,"id":"fresh-call","type":"function",
                    "function":{"name":"read","arguments":"{\"path\":\"current.txt\"}"}}]
                }),
                "tool_calls",
            ),
            "user" => sse(json!({"role":"assistant","content":"repeat answer"}), "stop"),
            role => panic!("unexpected last role {role}"),
        }
    })
    .await;
    let harness = harness(Options {
        settings: json!({"cacheWarming":"off","compaction":{"enabled":false}}),
        files: vec![("current.txt", "fresh result")],
        providers: json!({"wire": {"baseUrl":format!("{}/v1",server.url), "api":"openai-completions", "apiKey":"test", "models":[{"id":"wire"}]}}),
        ..Options::default()
    }).await;
    let model = harness.session.model_runtime().get_model("wire", "wire").unwrap();
    let manager = seed(SessionManager::create(&harness.cwd, Some(&harness.agent_dir.join("seed"))), &model);
    let history = manager.build_session_context().messages;
    let requests: Arc<Mutex<Vec<Context>>> = Arc::default();
    let captured = requests.clone();
    let stream: StreamFn = Arc::new(move |model, context, options| {
        captured.lock().unwrap().push(context.clone());
        pi_ai::stream(model, context, options)
    });
    let session = open_session_with(&harness.services, manager, |create| {
        create.model = Some(model.clone());
        create.stream_fn = Some(stream);
    })
    .await;
    session.agent().update_hooks(|hooks| {
        let convert = hooks.convert_to_llm.take().unwrap();
        let history = history.clone();
        hooks.convert_to_llm = Some(Arc::new(move |messages| {
            assert_prefix(&history, messages);
            let converted = convert(messages);
            assert_prefix(&history, &converted);
            converted
        }));
    });
    let events = record(&session);
    assert_prefix(&history, &session.agent().state().messages);
    for prompt in ["read current", "repeat"] {
        session.prompt(prompt, Vec::new()).await.unwrap();
        assert_prefix(&history, &session.session_manager().build_session_context().messages);
        assert_prefix(&history, &session.agent().state().messages);
    }
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert_prefix(&history, &convert_to_llm(&history));
    for request in requests.iter() {
        assert_prefix(&history, &request.messages);
    }
    let wire = server.requests();
    assert_eq!(wire.len(), 3);
    let first = wire[0].json();
    let messages = first["messages"].as_array().unwrap();
    let raw = String::from_utf8(wire[0].body.clone()).unwrap();
    for needle in [
        "saved instructions",
        "saved question",
        "saved-call",
        "old.txt",
        "saved result",
        "saved answer",
        "saved TS question",
        "saved TS answer",
    ] {
        assert!(raw.contains(needle), "wire history omitted {needle}");
    }
    for request in &wire[1..] {
        let body = request.json();
        assert_eq!(body["messages"].as_array().unwrap()[..messages.len()], messages[..]);
    }
    let order = event_order(&events.lock().unwrap());
    let lifecycle = order
        .iter()
        .map(String::as_str)
        .filter(|event| {
            matches!(
                *event,
                "agent_start"
                    | "turn_start"
                    | "turn_end"
                    | "agent_settled"
                    | "message_end:assistant"
                    | "tool_execution_start:read"
                    | "tool_execution_end:read"
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        lifecycle,
        [
            "agent_start",
            "turn_start",
            "message_end:assistant",
            "tool_execution_start:read",
            "tool_execution_end:read",
            "turn_end",
            "turn_start",
            "message_end:assistant",
            "turn_end",
            "agent_settled",
            "agent_start",
            "turn_start",
            "message_end:assistant",
            "turn_end",
            "agent_settled"
        ]
    );
}
