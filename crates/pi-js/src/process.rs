//! How a session starts child processes. Hosts supply a [`Launcher`] to place them, for example
//! in their own resource group; the default starts them directly.

use std::ffi::OsStr;
use std::process::{Command, ExitStatus};
use std::sync::Arc;

/// Builds the commands of one session's child processes.
pub trait Launcher: Send + Sync {
    /// A command that runs `program`. The caller adds the arguments, environment, directory,
    /// and stdio, so a launcher that wraps `program` must end its own arguments before it.
    fn launch(&self, program: &OsStr) -> Launch;
}

/// A command from [`Launcher::launch`] and what follows the process it starts.
pub struct Launch {
    pub command: Command,
    pub watch: Box<dyn Watch>,
}

impl Launch {
    /// `program` started directly, unwatched.
    pub fn direct(program: &OsStr) -> Self {
        Self { command: Command::new(program), watch: Box::new(Unwatched) }
    }
}

/// Follows one launched process. A launcher retains any group membership separately:
/// the command's exit does not imply its descendants have exited.
pub trait Watch: Send {
    /// The process started with `pid`.
    fn spawned(&mut self, _pid: u32) {}

    /// The process ended with `status`. Returns why it was stopped when the exit status alone
    /// does not say, such as a kill for exceeding a memory limit.
    fn ended(self: Box<Self>, _status: &ExitStatus) -> Option<Termination> {
        None
    }
}

/// Why a launched process was stopped from outside.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Termination {
    /// The processes of the group exceeded their memory limit of `limit` bytes.
    MemoryLimit { limit: u64 },
    /// A process the command started was killed for exceeding the memory limit of `limit`
    /// bytes, while the command itself exited with its own status.
    ProcessKilled { limit: u64 },
    /// The system stopped it to relieve memory pressure.
    MemoryPressure,
    /// Ended by SIGKILL, or with the status 137 a wrapping shell reports for it, without
    /// evidence that a memory limit or pressure caused it.
    Killed,
}

impl Termination {
    /// Whether the cause is known to have stopped the command. Otherwise its exit status stands.
    pub fn stopped(self) -> bool {
        matches!(self, Self::MemoryLimit { .. } | Self::MemoryPressure)
    }

    pub fn message(self) -> String {
        let gib = |limit: u64| limit as f64 / (1024.0 * 1024.0 * 1024.0);
        match self {
            Self::MemoryLimit { limit } => {
                format!("Command was killed: tool processes exceeded their memory limit ({:.1} GiB).", gib(limit))
            }
            Self::ProcessKilled { limit } => format!(
                "A process started by the command was killed: tool processes exceeded their memory limit ({:.1} GiB).",
                gib(limit)
            ),
            Self::MemoryPressure => "Command was killed to relieve system memory pressure.".into(),
            // A command can also exit with 137 itself, so this does not claim who stopped it.
            Self::Killed => {
                "Exit status 137 usually means the process was killed by SIGKILL; the tool memory limit did not cause it."
                    .into()
            }
        }
    }
}

/// Starts processes directly.
#[derive(Clone, Copy, Debug, Default)]
pub struct Direct;

impl Launcher for Direct {
    fn launch(&self, program: &OsStr) -> Launch {
        Launch::direct(program)
    }
}

struct Unwatched;

impl Watch for Unwatched {}

/// The default launcher.
pub fn direct() -> Arc<dyn Launcher> {
    Arc::new(Direct)
}

/// Keeps the Node engine's watch alive through startup, shutdown, and startup failures.
pub(crate) struct Child {
    pub process: std::process::Child,
    watch: Option<Box<dyn Watch>>,
}

impl Child {
    pub fn new(process: std::process::Child, mut watch: Box<dyn Watch>) -> Self {
        watch.spawned(process.id());
        Self { process, watch: Some(watch) }
    }

    pub fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
        let status = self.process.try_wait()?;
        if let (Some(status), Some(watch)) = (&status, self.watch.take_if(|_| status.is_some())) {
            watch.ended(status);
        }
        Ok(status)
    }

    pub fn kill(&mut self) -> std::io::Result<()> {
        self.process.kill()
    }

    pub fn wait(&mut self) -> std::io::Result<ExitStatus> {
        let status = self.process.wait()?;
        if let Some(watch) = self.watch.take() {
            watch.ended(&status);
        }
        Ok(status)
    }
}

impl Drop for Child {
    fn drop(&mut self) {
        if self.watch.is_some() {
            let _ = self.kill();
            let _ = self.wait();
        }
    }
}
