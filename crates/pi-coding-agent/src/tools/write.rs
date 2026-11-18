//! `write` tool: create or overwrite a file, creating parent directories.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use pi_agent::{AgentTool, AgentToolResult, ToolUpdateCallback};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use super::file_changes::{SharedFileChangeRecorder, write_recorded};
use super::mutation_queue::with_file_mutation_queue;
use super::operations::{WriteOperations, WriteToolOptions};
use super::path_utils::resolve_to_cwd;
use super::{check_aborted, parse_args};

pub struct WriteTool {
    cwd: PathBuf,
    operations: Arc<dyn WriteOperations>,
    recorder: Option<SharedFileChangeRecorder>,
}

#[derive(Deserialize)]
struct WriteArgs {
    path: String,
    content: String,
}

impl WriteTool {
    pub fn new(cwd: PathBuf) -> Self {
        Self::with_options(cwd, WriteToolOptions::default())
    }

    pub fn with_options(cwd: PathBuf, options: WriteToolOptions) -> Self {
        Self { cwd, operations: options.resolve(), recorder: options.recorder }
    }
}

#[async_trait]
impl AgentTool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to write (relative or absolute)" },
                "content": { "type": "string", "description": "Content to write to the file" }
            },
            "required": ["path", "content"]
        })
    }

    fn prompt_snippet(&self) -> Option<&str> {
        super::builtin_snippet("write")
    }

    fn prompt_guidelines(&self) -> Vec<String> {
        super::builtin_guidelines("write")
    }

    fn constrained_sampling(&self) -> Option<Value> {
        Some(super::prefer_json_schema())
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        args: Value,
        cancel: CancellationToken,
        _on_update: ToolUpdateCallback,
    ) -> anyhow::Result<AgentToolResult> {
        let args: WriteArgs = parse_args("write", args)?;
        let absolute = resolve_to_cwd(&args.path, &self.cwd);
        with_file_mutation_queue(&absolute, || async {
            check_aborted(&cancel)?;
            if let Some(dir) = absolute.parent() {
                self.operations
                    .mkdir(dir)
                    .await
                    .with_context(|| format!("Could not create directory {}", dir.display()))?;
            }
            check_aborted(&cancel)?;
            let write = || self.operations.write_file(&absolute, &args.content);
            use super::{
                file_changes::PriorContent,
                file_result::{FileChange, FileResult, Operation, Preview},
            };
            let before = self.operations.preview_before_write(&absolute, Preview::old_limit(args.content.len())).await;
            check_aborted(&cancel)?;
            let known = match &before {
                Some(PriorContent::Bytes(bytes)) => Some(bytes.as_slice()),
                _ => None,
            };
            write_recorded(self.recorder.as_ref(), &absolute, known, &args.content, write)
                .await
                .with_context(|| format!("Could not write file {}", absolute.display()))?;
            check_aborted(&cancel)?;
            let bytes = args.content.len();
            let result = AgentToolResult::text(format!("Successfully wrote {bytes} bytes to {}", args.path));
            let (operation, preview) = match before {
                Some(PriorContent::Missing) => (Operation::Create, Preview::between("", &args.content)),
                Some(PriorContent::Bytes(old)) => (
                    Operation::Update,
                    std::str::from_utf8(&old).map_or(Preview::Unavailable, |old| Preview::between(old, &args.content)),
                ),
                Some(PriorContent::NotRestorable(_)) => (Operation::Update, Preview::Unavailable),
                _ => (Operation::Write, Preview::Unavailable),
            };
            let mut report = FileResult::default();
            report.push(FileChange { operation, path: args.path, to: None, preview });
            let mut details = json!({});
            report.attach(&mut details);
            Ok(result.with_details(details))
        })
        .await
    }
}
