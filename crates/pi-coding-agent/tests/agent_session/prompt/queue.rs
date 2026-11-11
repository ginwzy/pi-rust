use crate::harness::{Options, harness, wait_until};
use pi_agent::AgentEvent;
use pi_ai::{Message, UserContent, UserMessage, providers::faux::FauxResponse};
use pi_coding_agent::agent_session::{PromptDisposition, PromptOptions, StreamingBehavior};
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

#[tokio::test]
async fn prepared_queue_callback_precedes_delivery_and_ids_survive_identical_content() {
    let h = harness(Options {
        responses: vec![FauxResponse::text("x".repeat(400)), FauxResponse::text("one"), FauxResponse::text("two")],
        delay: Some(Duration::from_millis(2)),
        ..Options::default()
    })
    .await;
    let delivered = Arc::new(Mutex::new(Vec::new()));
    let seen = delivered.clone();
    h.session.agent().subscribe(move |event| {
        if let AgentEvent::MessageEnd { message: Message::User(user) } = event {
            seen.lock().unwrap().push(user.input_id.clone());
        }
    });
    let session = h.session.clone();
    let running = tokio::spawn(async move {
        session.prompt_with("start", PromptOptions { input_id: Some("start".into()), ..Default::default() }).await
    });
    wait_until(|| !h.events_of("message_update").is_empty()).await;
    for id in ["one", "two"] {
        let session = Arc::downgrade(&h.session);
        let result = h
            .session
            .prompt_with(
                "same text",
                PromptOptions {
                    input_id: Some(id.into()),
                    streaming_behavior: Some(StreamingBehavior::Steer),
                    queued_input: Some(Arc::new(move |user, behavior| {
                        assert_eq!(user.input_id.as_deref(), Some(id));
                        assert_eq!(behavior, StreamingBehavior::Steer);
                        assert!(session.upgrade().unwrap().pending_message_count() < 2);
                        assert!(serde_json::to_value(user).unwrap().get("input_id").is_none());
                        Ok(())
                    })),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(result, PromptDisposition::Queued);
    }
    assert_eq!(*delivered.lock().unwrap(), [Some("start".into())]);
    running.await.unwrap().unwrap();
    assert_eq!(*delivered.lock().unwrap(), [Some("start".into()), Some("one".into()), Some("two".into())]);
}

#[tokio::test]
async fn failed_queue_save_prevents_delivery_and_prepared_restore_does_not_start_a_run() {
    let h = harness(Options { responses: vec![FauxResponse::Hang], ..Default::default() }).await;
    let session = h.session.clone();
    let running = tokio::spawn(async move { session.prompt("start", vec![]).await });
    wait_until(|| !h.requests().is_empty()).await;
    let error = h
        .session
        .prompt_with(
            "do not enqueue",
            PromptOptions {
                streaming_behavior: Some(StreamingBehavior::FollowUp),
                queued_input: Some(Arc::new(|_, _| anyhow::bail!("fixture save failed"))),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "fixture save failed");
    assert_eq!(h.session.pending_message_count(), 0);
    h.session.abort().await;
    running.await.unwrap().unwrap();
    h.session.restore_queued_input(
        UserMessage {
            input_id: Some("restored".into()),
            content: UserContent::Text("already expanded".into()),
            timestamp: 1,
        },
        StreamingBehavior::FollowUp,
    );
    assert_eq!(h.session.follow_up_messages(), ["already expanded"]);
    assert!(!h.session.is_streaming());
    assert_eq!(h.requests().len(), 1);
}
