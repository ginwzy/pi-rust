//! Tree navigation, names, labels, stats, and export (upstream `agent-session-tree-navigation`
//! and `agent-session-stats` tests).

use pi_ai::providers::faux::FauxResponse;
use pi_ai::{Message, ThinkingLevel};
use pi_coding_agent::agent_session::NavigateTreeOptions;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::harness::{Harness, Options, harness, roles};

/// Session entry ids of user messages, in order.
fn user_entry_ids(h: &Harness) -> Vec<String> {
    h.entries()
        .iter()
        .filter(|entry| entry["type"] == "message" && entry["message"]["role"] == "user")
        .map(|entry| entry["id"].as_str().unwrap().to_string())
        .collect()
}

async fn two_turns(extra: Vec<FauxResponse>) -> Harness {
    let mut responses = vec![FauxResponse::text("first answer"), FauxResponse::text("second answer")];
    responses.extend(extra);
    let h = harness(Options { responses, ..Options::default() }).await;
    h.session.prompt("first question", Vec::new()).await.unwrap();
    h.session.prompt("second question", Vec::new()).await.unwrap();
    h
}

#[tokio::test]
async fn conditional_name_write_rejects_intervening_rename_and_clear() {
    let h = harness(Options::default()).await;
    assert!(h.session.set_session_name_if_unchanged("generated", None).unwrap());
    let prior = h.entries().last().unwrap()["id"].as_str().unwrap().to_owned();
    h.session.set_session_name("extension").unwrap();
    let count = h.entries().len();
    assert!(!h.session.set_session_name_if_unchanged("late", Some(&prior)).unwrap());
    assert_eq!(h.entries().len(), count);
    assert_eq!(h.session.session_manager().session_name().as_deref(), Some("extension"));
    h.session.set_session_name("").unwrap();
    assert!(!h.session.set_session_name_if_unchanged("late", None).unwrap());
    let current = h.entries().last().unwrap()["id"].as_str().unwrap().to_owned();
    assert!(h.session.set_session_name_if_unchanged("new", Some(&current)).unwrap());
    assert_eq!(h.session.session_manager().session_name().as_deref(), Some("new"));
}

#[tokio::test]
async fn navigates_to_a_user_message_and_returns_its_text() {
    let h = two_turns(Vec::new()).await;
    let second = user_entry_ids(&h)[1].clone();
    let result = h.session.navigate_tree(&second, NavigateTreeOptions::default()).await.unwrap();
    assert_eq!(result.editor_text.as_deref(), Some("second question"));
    assert!(!result.cancelled && result.summary_entry_id.is_none());
    assert_eq!(roles(&h.session.messages()), ["system", "user", "assistant"]);
    assert_eq!(h.requests().len(), 2, "no summary without summarize");

    let same = h.session.session_manager().leaf_id().unwrap().to_string();
    let noop = h.session.navigate_tree(&same, NavigateTreeOptions::default()).await.unwrap();
    assert_eq!(noop.editor_text, None);
    let missing = h.session.navigate_tree("nope", NavigateTreeOptions::default()).await.unwrap_err();
    assert_eq!(missing.to_string(), "Entry nope not found");
}

#[tokio::test]
async fn host_navigation_keeps_exact_user_targets_and_can_clear_to_root() {
    let h = two_turns(Vec::new()).await;
    let target = user_entry_ids(&h)[1].clone();
    let result =
        h.session.navigate_to(Some(&target), NavigateTreeOptions::default(), CancellationToken::new()).await.unwrap();
    assert!(result.editor_text.is_none());
    assert_eq!(h.session.session_manager().leaf_id(), Some(target.as_str()));
    assert_eq!(roles(&h.session.messages()), ["system", "user", "assistant", "user"]);
    h.session
        .navigate_to(None, NavigateTreeOptions { summarize: true, ..Default::default() }, CancellationToken::new())
        .await
        .unwrap();
    assert!(h.session.session_manager().leaf_id().is_none());
    assert!(h.session.messages().is_empty());
    assert_eq!(h.requests().len(), 2);
}

#[tokio::test]
async fn host_summary_attaches_to_the_exact_target_and_honors_pre_cancel() {
    let h = two_turns(vec![FauxResponse::text("branch summary")]).await;
    let target = user_entry_ids(&h)[0].clone();
    let old = h.session.session_manager().leaf_id().map(String::from);
    let cancel = CancellationToken::new();
    cancel.cancel();
    let result = h.session.navigate_to(Some(&target), NavigateTreeOptions::default(), cancel).await.unwrap();
    assert!(result.aborted && result.cancelled);
    assert_eq!(h.session.session_manager().leaf_id(), old.as_deref());
    let result = h
        .session
        .navigate_to(
            Some(&target),
            NavigateTreeOptions { summarize: true, ..Default::default() },
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let summary = result.summary_entry_id.unwrap();
    assert_eq!(h.session.session_manager().get_entry(&summary).unwrap().parent_id(), Some(target.as_str()));
    assert_eq!(h.requests().len(), 3);
}

#[tokio::test]
async fn summarizes_the_abandoned_branch_with_custom_instructions() {
    let h = two_turns(vec![FauxResponse::text("## Branch\nWe discussed the second question.")]).await;
    let second = user_entry_ids(&h)[1].clone();
    let options = NavigateTreeOptions {
        summarize: true,
        custom_instructions: Some("Mention the answers".into()),
        label: Some("checkpoint".into()),
        ..NavigateTreeOptions::default()
    };
    let result = h.session.navigate_tree(&second, options).await.unwrap();
    let summary_id = result.summary_entry_id.clone().unwrap();
    let request = h.requests().pop().unwrap();
    let prompt = crate::harness::text(request.messages.last().unwrap());
    assert!(prompt.contains("Mention the answers"), "{prompt}");

    let entries = h.entries();
    let summary = entries.iter().find(|entry| entry["id"] == summary_id.as_str()).unwrap();
    assert_eq!(summary["type"], "branch_summary");
    assert!(summary["summary"].as_str().unwrap().contains("second question"));
    let labels: Vec<&Value> = entries.iter().filter(|entry| entry["type"] == "label").collect();
    assert_eq!(labels.len(), 1);
    assert_eq!(labels[0]["targetId"], summary_id.as_str());
    let messages = h.session.messages();
    assert!(matches!(messages.last(), Some(Message::Custom(custom)) if custom.role == "branchSummary"));
}

#[tokio::test]
async fn names_labels_and_reports_stats() {
    let h = two_turns(Vec::new()).await;
    h.session.set_session_name("  Release prep ").unwrap();
    assert_eq!(h.session.session_name().as_deref(), Some("Release prep"));
    assert_eq!(h.events_of("session_info_changed"), [json!({"type": "session_info_changed", "name": "Release prep"})]);
    let first = user_entry_ids(&h)[0].clone();
    h.session.set_label(&first, Some("start")).unwrap();
    assert!(h.entries().iter().any(|entry| entry["type"] == "label" && entry["label"] == "start"));

    let report = h.session.session_stats().to_json();
    assert_eq!((report["userMessages"].clone(), report["assistantMessages"].clone()), (json!(2), json!(2)));
    assert!(report["tokens"]["input"].as_u64().unwrap() > 0);
    assert_eq!(report["sessionId"], h.session.session_id());
    let usage = h.session.context_usage().unwrap();
    assert_eq!(usage.context_window, 128_000);
    assert!(usage.tokens.unwrap() > 0);

    let path = h.session.export_to_html(Some(h.cwd.join("out.html").to_str().unwrap())).unwrap();
    assert_eq!(path, h.cwd.join("out.html"));
    assert!(std::fs::read_to_string(path).unwrap().contains("<html"));
    let forks = h.session.user_messages_for_forking();
    assert_eq!(forks.iter().map(|(_, text)| text.as_str()).collect::<Vec<_>>(), ["first question", "second question"]);
}

#[tokio::test]
async fn in_memory_sessions_cannot_export_html() {
    let h = harness(Options { persist: false, ..Options::default() }).await;
    let error = h.session.export_to_html(None).unwrap_err();
    assert_eq!(error.to_string(), "Cannot export in-memory session to HTML");
}

// Upstream issue 11451: a resume restored the level of a change left on the abandoned branch.
#[tokio::test]
async fn prompts_after_tree_navigation_record_the_live_thinking_level() {
    let responses = ["one", "two", "three"].map(FauxResponse::text).to_vec();
    let settings = json!({"defaultThinkingLevel": "high"});
    let h = harness(Options { responses, settings, ..Options::default() }).await;
    h.session.prompt("first question", Vec::new()).await.unwrap();
    let first_answer = h.session.session_manager().leaf_id().unwrap().to_string();
    h.session.set_thinking_level(ThinkingLevel::Low, false);
    h.session.prompt("second question", Vec::new()).await.unwrap();
    h.session.navigate_tree(&first_answer, NavigateTreeOptions::default()).await.unwrap();
    let level = || h.session.session_manager().build_session_context().thinking_level;
    assert_eq!(level(), Some(ThinkingLevel::High), "the branch still names the old level");
    h.session.prompt("third question", Vec::new()).await.unwrap();
    assert_eq!(h.session.thinking_level(), ThinkingLevel::Low);
    assert_eq!(level(), Some(ThinkingLevel::Low));
    let changes = |h: &Harness| h.entries().iter().filter(|entry| entry["type"] == "thinking_level_change").count();
    let before = changes(&h);
    h.session.prompt("fourth question", Vec::new()).await.unwrap();
    assert_eq!(changes(&h), before, "an unchanged level is not recorded again");
}
