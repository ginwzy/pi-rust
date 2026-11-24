use std::sync::Arc;

use pi_agent::StreamFn;
use pi_ai::{ContentBlock, Message, MessageBuilder, StopReason, event_channel};
use serde_json::json;

use crate::harness::{Options, harness_with};

#[tokio::test]
async fn a_long_unread_stream_keeps_exact_deltas_and_final_tool_arguments() {
    const COUNT: usize = 96;
    let stream: StreamFn = Arc::new(|model, context, _| {
        let (sender, stream) = event_channel();
        let mut builder = MessageBuilder::new(model, sender);
        builder.start();
        if matches!(context.messages.last(), Some(Message::ToolResult(_))) {
            builder.text_delta("continued");
            builder.finish(StopReason::Stop);
        } else {
            for _ in 0..COUNT {
                builder.thinking_delta("ponder ");
            }
            for _ in 0..COUNT {
                builder.text_delta("text ");
            }
            let call = builder.tool_call_start("long-call", "read");
            for fragment in ["{\"path\":\"sample.txt\""].into_iter().chain(std::iter::repeat_n(" ", COUNT)).chain(["}"])
            {
                builder.tool_call_delta(call, fragment);
            }
            builder.finish(StopReason::ToolUse);
        }
        stream
    });
    let h = harness_with(Options { files: vec![("sample.txt", "read by tool")], ..Options::default() }, |create| {
        create.stream_fn = Some(stream);
    })
    .await;
    h.session.prompt("read", Vec::new()).await.unwrap();
    let updates = h.events_of("message_update");
    assert!(updates.len() <= 46, "{} unread updates escaped compression", updates.len());
    let deltas = |kind: &str| -> String {
        updates
            .iter()
            .map(|event| &event["assistantMessageEvent"])
            .filter(|event| event["type"] == kind)
            .map(|event| event["delta"].as_str().unwrap())
            .collect()
    };
    assert_eq!(deltas("thinking_delta"), "ponder ".repeat(COUNT));
    assert_eq!(deltas("text_delta"), format!("{}continued", "text ".repeat(COUNT)));
    assert_eq!(deltas("toolcall_delta"), format!("{{\"path\":\"sample.txt\"{}}}", " ".repeat(COUNT)));
    let messages = h.session.messages();
    let first = messages
        .iter()
        .find_map(|message| match message {
            Message::Assistant(assistant) => Some(assistant),
            _ => None,
        })
        .unwrap();
    assert_eq!(first.stop_reason, StopReason::ToolUse);
    assert_eq!(first.content.len(), 3);
    assert!(
        matches!(&first.content[0], ContentBlock::Thinking { thinking, .. } if thinking == &"ponder ".repeat(COUNT))
    );
    assert_eq!(first.content[1].as_text().unwrap(), "text ".repeat(COUNT));
    let call = first.tool_calls().next().unwrap();
    assert_eq!(call.arguments, *json!({"path":"sample.txt"}).as_object().unwrap());
    assert!(call.arguments_error.is_none());
    assert_eq!(h.session.last_assistant_text().as_deref(), Some("continued"));
    assert!(
        h.events_of("tool_execution_end")
            .iter()
            .any(|event| event["toolCallId"] == "long-call" && event["isError"] == false)
    );
}
