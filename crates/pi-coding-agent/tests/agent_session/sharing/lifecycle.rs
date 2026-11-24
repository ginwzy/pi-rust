use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use futures::FutureExt;
use pi_agent::{AgentEvent, StreamFn};
use pi_ai::providers::faux::FauxResponse;
use pi_ai::{ContentBlock, Message, MessageBuilder, StopReason, UserMessage, event_channel};
use pi_coding_agent::cache_warmer::CacheWarmingState;
use pi_coding_agent::settings::schema::CacheWarmingMode;
use serde_json::json;

use super::{assert_prefix, same_message};
use crate::harness::{Options, harness, harness_with, text};

#[tokio::test]
async fn tool_continuation_preserves_history_when_steering_and_the_loadout_change() {
    let h = harness(Options {
        responses: vec![
            FauxResponse::text("history"),
            FauxResponse::tool_call("fresh", "read", json!({"path":"file.txt"})),
            FauxResponse::text("before replacement"),
        ],
        files: vec![("file.txt", "file contents")],
        ..Options::default()
    })
    .await;
    h.session.prompt("remember", Vec::new()).await.unwrap();
    let history = h.session.messages();
    let superseded: Arc<Mutex<Option<Message>>> = Arc::default();
    h.session.agent().update_hooks(|hooks| {
        let prepare = hooks.prepare_next_turn.take().unwrap();
        let session = Arc::downgrade(&h.session);
        hooks.prepare_next_turn = Some(Arc::new(move |turn, cancel| {
            let session = session.upgrade().unwrap();
            let prepare = prepare.clone();
            async move {
                assert_eq!(turn.tool_results.len(), 1);
                session.set_active_tools_by_name(&["read".into()]);
                let update = prepare(turn, cancel).await.unwrap();
                assert!(update.messages.iter().any(|message| matches!(message, Message::System(system) if system.tools_removed.as_ref().is_some_and(|tools| !tools.is_empty()))));
                session.steer("between requests", Vec::new()).await.unwrap();
                Some(update)
            }.boxed()
        }));
        let on_event = hooks.on_event.take().unwrap();
        let captured = superseded.clone();
        hooks.on_event = Some(Arc::new(move |event, cancel| {
            let on_event = on_event.clone();
            let captured = captured.clone();
            async move {
                let replacement = on_event(event.clone(), cancel).await;
                if let AgentEvent::MessageEnd { message: Message::Assistant(assistant) } = event
                    && assistant.text() == "before replacement" {
                    assert!(replacement.is_none());
                    *captured.lock().unwrap() = Some(Message::Assistant(assistant.clone()));
                    let mut edited = assistant.as_ref().clone();
                    edited.content = vec![ContentBlock::text("after replacement")];
                    return Some(edited.into());
                }
                replacement
            }.boxed()
        }));
    });
    h.session.prompt("read the file", Vec::new()).await.unwrap();
    let requests = h.requests();
    assert_eq!(requests.len(), 3);
    for request in &requests[1..] {
        let mut request_history = request.messages.iter().filter(|message| !matches!(message, Message::System(_)));
        for expected in history.iter().filter(|message| !matches!(message, Message::System(_))) {
            let actual = request_history.next().unwrap();
            assert!(same_message(expected, actual), "history {} changed to {}", expected.role(), actual.role());
        }
    }
    assert_eq!(text(requests[2].messages.last().unwrap()), "between requests");
    let projected = h.session.session_manager().build_session_context().messages;
    assert_prefix(&history, &projected);
    assert!(
        projected[history.len()..]
            .iter()
            .any(|message| matches!(message, Message::System(system) if system.tools_removed.is_some()))
    );
    let superseded = superseded.lock().unwrap();
    assert_eq!(text(superseded.as_ref().unwrap()), "before replacement");
    assert_eq!(text(projected.last().unwrap()), "after replacement");
    assert!(!same_message(superseded.as_ref().unwrap(), projected.last().unwrap()));
    assert!(same_message(projected.last().unwrap(), h.session.messages().last().unwrap()));
}

#[tokio::test(start_paused = true)]
async fn warm_replay_shares_request_handles_and_drops_them_when_invalidated() {
    for (mode, invalidate) in
        [("off", "settle"), ("streaming", "settle"), ("idle", "stop"), ("idle", "model"), ("idle", "context")]
    {
        let retained: Arc<Mutex<Weak<UserMessage>>> = Arc::default();
        let marker = retained.clone();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let calls = requests.clone();
        let stream: StreamFn = Arc::new(move |model, context, options| {
            let Message::User(marker) = context.messages.last().unwrap() else { panic!("request-only marker") };
            calls.lock().unwrap().push(Arc::downgrade(marker));
            let (sender, stream) = event_channel();
            let mut builder = MessageBuilder::new(model, sender);
            builder.start();
            builder.text_delta("answer");
            if options.max_tokens == Some(1) {
                builder.message_mut().usage.cache_read = 20_000;
                builder.finish(StopReason::Length);
            } else {
                builder.message_mut().usage.cache_write = 20_000;
                builder.finish(StopReason::Stop);
            }
            stream
        });
        let h = harness_with(Options {
            settings: json!({"cacheWarming":mode}),
            tiny: json!({"promptCache":{"short":12},"cost":{"input":3,"output":15,"cacheRead":0.3,"cacheWrite":30}}),
            ..Options::default()
        }, |options| options.stream_fn = Some(stream)).await;
        h.session.agent().update_hooks(|hooks| {
            let original = hooks.transform_context.take().unwrap();
            hooks.transform_context = Some(Arc::new(move |messages, cancel| {
                let original = original.clone();
                let marker = marker.clone();
                async move {
                    let mut messages = original(messages, cancel).await;
                    let request_only = Message::user("request-only");
                    let Message::User(user) = &request_only else { unreachable!() };
                    *marker.lock().unwrap() = Arc::downgrade(user);
                    messages.push(request_only);
                    messages
                }
                .boxed()
            }));
        });
        h.session.prompt("prompt", Vec::new()).await.unwrap();
        if mode == "idle" {
            assert_eq!(h.session.cache_warming_status().state, CacheWarmingState::Scheduled);
            assert!(retained.lock().unwrap().upgrade().is_some());
            tokio::time::sleep(Duration::from_millis(2_100)).await;
            assert_eq!(requests.lock().unwrap().len(), 2);
            assert!(requests.lock().unwrap().iter().all(|request| Weak::ptr_eq(request, &retained.lock().unwrap())));
            match invalidate {
                "stop" => h.session.set_cache_warming_mode(CacheWarmingMode::Off),
                "model" => {
                    let other = h.session.model_runtime().get_model("local", "other").unwrap();
                    h.session.set_model(other, false).await.unwrap();
                }
                "context" => h.session.agent().update_state(|state| state.messages.clear()),
                _ => unreachable!(),
            }
            assert_eq!(h.session.cache_warming_status().state, CacheWarmingState::Inactive);
        } else {
            tokio::time::sleep(Duration::from_millis(2_100)).await;
            assert_eq!(requests.lock().unwrap().len(), 1);
        }
        assert!(retained.lock().unwrap().upgrade().is_none(), "{mode}/{invalidate} retained the old context");
    }
}
