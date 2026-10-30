use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use pi_ai::providers::faux::FauxResponse;
use pi_coding_agent::agent_session::{PromptCancelled, PromptDisposition, PromptOptions, StreamingBehavior};
use pi_coding_agent::extensions::backend::ExtensionRuntime;
use tokio_util::sync::CancellationToken;

use crate::harness::{Setup, eventually, harness};

const SOURCE: &str = r#"export default function(pi) {
    const releases = new Map();
    async function wait(key) {
        pi.appendEntry("waiting", { key });
        await new Promise(resolve => { releases.set(key, resolve); });
    }
    pi.registerCommand("release", { handler: async (key) => { releases.get(key)(); } });
    pi.registerCommand("command", { handler: async () => wait("/command") });
    pi.on("input", async (event) => {
        if (event.text === "collected") {
            pi.appendEntry("waiting", {});
            await new Promise(() => {});
        }
        if (event.text === "throws") throw new Error("ordinary hook failure");
        if (!event.text.startsWith("input ")) return { action: "continue" };
        await wait(event.text);
        const action = event.text.slice(6);
        return action === "transform" ? { action, text: "transformed" } : { action };
    });
    pi.on("before_agent_start", async (event) => {
        if (event.prompt === "before") await wait("before");
    });
}"#;

fn setup() -> Setup {
    Setup {
        files: vec![("cancel.ts", SOURCE.into())],
        load: vec!["cancel.ts"],
        responses: vec![FauxResponse::text("next")],
        runtime: Some(ExtensionRuntime::QuickJs),
        ..Setup::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn cancelled_input_does_not_join_another_run_as_a_follow_up() {
    let h = harness(setup()).await;
    let session = h.session.clone();
    let running = tokio::spawn(async move { session.prompt("before", Vec::new()).await });
    eventually(|| h.entries().iter().any(|entry| entry["data"]["key"] == "before")).await;
    let cancel = CancellationToken::new();
    let options = PromptOptions {
        cancel: Some(cancel.clone()),
        streaming_behavior: Some(StreamingBehavior::FollowUp),
        ..Default::default()
    };
    let session = h.session.clone();
    let pending = tokio::spawn(async move { session.prompt_with("input transform", options).await });
    eventually(|| h.entries().iter().any(|entry| entry["data"]["key"] == "input transform")).await;
    cancel.cancel();
    h.session.prompt("/release input transform", Vec::new()).await.unwrap();
    assert!(pending.await.unwrap().unwrap_err().is::<PromptCancelled>());
    h.session.prompt("/release before", Vec::new()).await.unwrap();
    assert_eq!(running.await.unwrap().unwrap(), PromptDisposition::Started);
    assert_eq!(h.provider.requests().len(), 1);
    assert_eq!(h.last_prompt(), "before");
    assert_eq!(h.entries().iter().filter(|entry| entry["message"]["role"] == "user").count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn shutting_down_an_input_handler_does_not_release_the_original_text() {
    let h = harness(setup()).await;
    let session = h.session.clone();
    let task = tokio::spawn(async move { session.prompt("input continue", Vec::new()).await });
    eventually(|| h.entries().iter().any(|entry| entry["customType"] == "waiting")).await;
    h.runner.shutdown();
    let error = tokio::time::timeout(std::time::Duration::from_secs(5), task).await.unwrap().unwrap().unwrap_err();
    assert!(error.to_string().contains("input handler"));
    assert!(h.provider.requests().is_empty());
    assert!(!h.entries().iter().any(|entry| entry["message"]["role"] == "user"));
}

#[tokio::test(flavor = "multi_thread")]
async fn cancelled_preparation_never_reports_admission_and_does_not_poison_the_next_input() {
    for input in ["input continue", "input transform", "input handled", "/command", "before"] {
        let h = harness(setup()).await;
        let cancel = CancellationToken::new();
        let count = Arc::new(AtomicUsize::new(0));
        let seen = count.clone();
        let options = PromptOptions {
            cancel: Some(cancel.clone()),
            preflight_result: Some(Arc::new(move |_| {
                seen.fetch_add(1, Ordering::SeqCst);
            })),
            ..Default::default()
        };
        let session = h.session.clone();
        let task = tokio::spawn(async move { session.prompt_with(input, options).await });
        eventually(|| h.entries().iter().any(|entry| entry["customType"] == "waiting")).await;
        cancel.cancel();
        assert_eq!(
            h.session.prompt(&format!("/release {input}"), Vec::new()).await.unwrap(),
            PromptDisposition::Handled
        );
        let error = task.await.unwrap().unwrap_err();
        assert!(error.is::<PromptCancelled>(), "{input}: {error:#}");
        assert_eq!(count.load(Ordering::SeqCst), 0);
        assert!(h.provider.requests().is_empty());
        assert!(!h.entries().iter().any(|entry| entry["message"]["role"] == "user"));
        assert!(h.session.is_idle());
        assert_eq!(h.session.prompt("next", Vec::new()).await.unwrap(), PromptDisposition::Started);
        assert_eq!(h.last_prompt(), "next");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn already_cancelled_input_does_not_invoke_extensions() {
    let h = harness(setup()).await;
    let cancel = CancellationToken::new();
    cancel.cancel();
    let error = h
        .session
        .prompt_with("input continue", PromptOptions { cancel: Some(cancel), ..Default::default() })
        .await
        .unwrap_err();
    assert!(error.is::<PromptCancelled>());
    assert!(!h.entries().iter().any(|entry| entry["customType"] == "waiting"));
    assert!(h.provider.requests().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn cancellation_after_admission_keeps_the_user_message() {
    let h = harness(setup()).await;
    let cancel = CancellationToken::new();
    let signal = cancel.clone();
    let session = h.session.clone();
    let result = h
        .session
        .prompt_with(
            "accepted",
            PromptOptions {
                cancel: Some(cancel),
                preflight_result: Some(Arc::new(move |disposition| {
                    assert_eq!(disposition, PromptDisposition::Started);
                    signal.cancel();
                    session.request_abort();
                })),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(result, PromptDisposition::Started);
    assert_eq!(h.entries().iter().filter(|entry| entry["message"]["role"] == "user").count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn broken_input_calls_fail_but_ordinary_handler_errors_keep_their_semantics() {
    let h = harness(setup()).await;
    let error = tokio::time::timeout(std::time::Duration::from_secs(5), h.session.prompt("collected", Vec::new()))
        .await
        .unwrap()
        .unwrap_err();
    assert!(error.to_string().contains("input handler"), "{error:#}");
    assert!(h.provider.requests().is_empty());
    assert!(!h.entries().iter().any(|entry| entry["message"]["role"] == "user"));
    assert_eq!(h.session.prompt("throws", Vec::new()).await.unwrap(), PromptDisposition::Started);
    assert_eq!(h.errors.lock().unwrap().len(), 1);
    assert_eq!(h.last_prompt(), "throws");
}
