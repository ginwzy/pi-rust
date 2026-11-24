use std::path::Path;
use std::sync::Arc;

use pi_agent::Agent;
use pi_ai::providers::{faux::FauxProvider, transform_messages};
use pi_ai::{AssistantMessage, ContentBlock, Message, ThinkingLevel, UserContent, UserMessage};
use serde_json::json;

use crate::messages::convert_to_llm;
use crate::session::SessionManager;

use super::project_path;

fn assert_equivalent(manager: &SessionManager) {
    let expected = project_path(manager.get_branch(None));
    assert_eq!(manager.build_session_projection(), expected);
    assert_eq!(manager.build_session_context(), expected.into_context());
}

fn assert_same(left: &Message, right: &Message) {
    let same = match (left, right) {
        (Message::System(left), Message::System(right)) => Arc::ptr_eq(left, right),
        (Message::User(left), Message::User(right)) => Arc::ptr_eq(left, right),
        (Message::Assistant(left), Message::Assistant(right)) => Arc::ptr_eq(left, right),
        (Message::Custom(left), Message::Custom(right)) => Arc::ptr_eq(left, right),
        _ => false,
    };
    assert!(same, "different allocations for {} and {}", left.role(), right.role());
}

#[test]
fn append_reuses_messages_after_edits_compactions_and_branch_switches() {
    let mut manager = SessionManager::in_memory(Path::new("."));
    let user = Message::user("keep");
    let first = manager.append_message(&user).unwrap();
    let custom = manager.append_custom_message_entry("test", UserContent::Text("custom".into()), false, None).unwrap();
    let omitted = manager.append_message(&Message::user("omit")).unwrap();
    manager.append_context_edit(&custom, Some(json!("edited custom"))).unwrap();
    manager.append_context_edit(&omitted, None).unwrap();
    let first_compaction = manager.append_compaction("first", &first, 20).unwrap();
    assert_equivalent(&manager);
    let original = manager.build_session_context().messages;
    assert_same(&original[1], &user);

    let previous_revision = manager.revision();
    let model = manager.append_model_change("test", "model").unwrap();
    manager.append_thinking_level_change(ThinkingLevel::High).unwrap();
    manager.append_message(&Message::user("append")).unwrap();
    assert!(manager.revision() > previous_revision);
    assert_equivalent(&manager);
    for (old, new) in original.iter().zip(manager.build_session_context().messages.iter()) {
        assert_same(old, new);
    }

    let second_compaction = manager.append_compaction("second", &first, 30).unwrap();
    assert_equivalent(&manager);
    let compacted = manager.build_session_context().messages;
    assert_same(&original[2], &compacted[2]);
    manager.branch(&model).unwrap();
    assert_equivalent(&manager);
    assert_same(&original[2], &manager.build_session_context().messages[2]);
    manager.branch(&second_compaction).unwrap();
    assert_equivalent(&manager);

    let before_replace = manager.build_session_context().messages;
    let changed = manager.get_entry(&second_compaction).unwrap().with_field("summary", Some(json!("replacement")));
    manager.replace_entry(changed).unwrap();
    assert_equivalent(&manager);
    let after_replace = manager.build_session_context().messages;
    assert_ne!(before_replace[0], after_replace[0]);
    assert_same(&before_replace[2], &after_replace[2]);
    assert!(manager.get_entry(&first_compaction).is_some());

    manager.reset_leaf();
    assert_equivalent(&manager);
    assert!(manager.build_session_context().messages.is_empty());
}

#[test]
fn saved_execution_and_llm_share_the_body_but_not_delivery_metadata() {
    let model = FauxProvider::model();
    let mut manager = SessionManager::in_memory(Path::new("."));
    let answer = AssistantMessage { content: vec![ContentBlock::text("original")], ..AssistantMessage::empty(&model) };
    let answer = Message::from(answer);
    let answer_id = manager.append_message(&answer).unwrap();
    let stored = manager.get_entry(&answer_id).unwrap().native_message().unwrap();
    assert_same(&answer, stored);
    let agent = Agent::new(model.clone());
    agent.update_state(|state| state.messages = manager.build_session_context().messages);
    let snapshot = agent.state();
    assert_same(stored, &snapshot.messages[0]);
    let llm = convert_to_llm(&snapshot.messages);
    assert_same(stored, &llm[0]);
    let transformed = transform_messages(&llm, &model, &|id, _| id.to_owned());
    assert_same(stored, &transformed[0]);

    let queued = Arc::new(UserMessage {
        input_id: Some("queued-id".into()),
        content: UserContent::Text("queued".into()),
        timestamp: 1,
    });
    let queued_id = manager.append_message(&Message::User(queued.clone())).unwrap();
    let saved = manager.get_entry(&queued_id).unwrap();
    assert!(saved.to_value()["message"].get("inputId").is_none());
    let Message::User(saved) = saved.native_message().unwrap() else { panic!("expected user") };
    assert!(!Arc::ptr_eq(&queued, saved));
    assert!(saved.input_id.is_none());
    assert_eq!(queued.input_id.as_deref(), Some("queued-id"));
    assert_same(&Message::User(saved.clone()), &manager.build_session_context().messages[1]);

    manager.append_context_edit(&answer_id, Some(json!("edited"))).unwrap();
    let projection = manager.build_session_context();
    let Message::Assistant(edited) = &projection.messages[0] else { panic!("expected assistant") };
    let Message::Assistant(original) = &answer else { panic!("expected assistant") };
    assert!(!Arc::ptr_eq(original, edited));
    assert_eq!(original.text(), "original");
    assert_eq!(edited.text(), "edited");
    assert_same(&answer, &agent.state().messages[0]);
}
