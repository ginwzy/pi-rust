//! Host capabilities an embedding can refuse (`ExtensionBindings::unsupported`). A refused call
//! throws the host's message in JavaScript instead of returning the default of an absent UI or
//! session runtime, so an extension can tell that nothing happened.

use pi_js::{JsError, UNSUPPORTED_CODE};

use super::ops::Args;
use super::runner::ExtensionRunner;

/// A `ctx` method an embedding may not provide.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Capability {
    /// `ctx.newSession()`.
    NewSession,
    /// `ctx.fork()`.
    Fork,
    /// `ctx.switchSession()`.
    SwitchSession,
    /// `ctx.navigateTree()`.
    NavigateTree,
    /// `ctx.reload()`.
    Reload,
    /// `ctx.ui.getEditorText()`.
    EditorText,
    /// `ctx.ui.setFooter()`.
    Footer,
    /// `ctx.ui.setHeader()`.
    Header,
    /// `ctx.ui.setWidget()` with a component factory; plain lines are not this capability.
    WidgetComponent,
    /// `ctx.ui.custom()`.
    Custom,
    /// `ctx.ui.setEditorComponent()`.
    EditorComponent,
    /// `ctx.ui.onTerminalInput()`.
    TerminalInput,
    /// `ctx.ui.writeTerminal()`.
    WriteTerminal,
    /// `ctx.ui.addAutocompleteProvider()`.
    Autocomplete,
    /// `ctx.ui.getToolsExpanded()` and `setToolsExpanded()`.
    ToolsExpanded,
    /// `ctx.ui.getAllThemes()` and `setTheme()`.
    Themes,
}

impl Capability {
    /// The capability a host operation needs, if a host can refuse it.
    fn of(op: &str) -> Option<Self> {
        Some(match op {
            "session.new" => Self::NewSession,
            "session.fork" => Self::Fork,
            "session.switch" => Self::SwitchSession,
            "session.navigateTree" => Self::NavigateTree,
            "session.reload" => Self::Reload,
            "ui.getEditorText" => Self::EditorText,
            "ui.setFooter" => Self::Footer,
            "ui.setHeader" => Self::Header,
            "ui.setWidgetComponent" => Self::WidgetComponent,
            "ui.custom" => Self::Custom,
            "ui.setEditorComponent" => Self::EditorComponent,
            "ui.terminalInput" => Self::TerminalInput,
            "ui.writeTerminal" => Self::WriteTerminal,
            "ui.addAutocompleteProvider" => Self::Autocomplete,
            "ui.getToolsExpanded" | "ui.setToolsExpanded" => Self::ToolsExpanded,
            "ui.getAllThemes" | "ui.setTheme" => Self::Themes,
            _ => return None,
        })
    }
}

/// The capabilities a host refuses, and the message every refused call throws.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Unsupported {
    pub capabilities: Vec<Capability>,
    pub message: String,
}

/// Refuse `op` when the host of the session it targets (its first argument) does not provide it.
pub(crate) fn check(runner: &ExtensionRunner, op: &str, args: &Args) -> Result<(), JsError> {
    let Some(capability) = Capability::of(op) else { return Ok(()) };
    match runner.target_runner(args.u64(0)).bindings().unsupported {
        Some(unsupported) if unsupported.capabilities.contains(&capability) => {
            Err(JsError::new(unsupported.message).with_code(UNSUPPORTED_CODE))
        }
        _ => Ok(()),
    }
}
