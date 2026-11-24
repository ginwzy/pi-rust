//! Requests [`JsHandle`] and [`JsRef`] send to the Node worker, and their results.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};

use super::client::{Waiter, shut_down};
use super::wire;
use crate::error::JsError;
use crate::host::HostValue;
use crate::runtime::{JsHandle, JsRef};

/// How often a waiting call checks that the `node` child is still running.
const LIVENESS_POLL: Duration = Duration::from_millis(250);

/// How the worker returns a result.
#[derive(Clone, Copy)]
pub(crate) enum Want {
    Value,
    Ref,
    OptRef,
}

impl Want {
    fn name(self) -> &'static str {
        match self {
            Self::Value => "value",
            Self::Ref => "ref",
            Self::OptRef => "optRef",
        }
    }
}

pub(crate) fn frame(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

pub(crate) fn import_frame(specifier: &str, default: bool) -> Map<String, Value> {
    frame(json!({ "op": "import", "specifier": specifier, "default": default, "want": "ref" }))
}

#[derive(serde::Serialize)]
pub(crate) struct CallFrame<'a> {
    op: &'static str,
    target: u64,
    args: Vec<wire::Argument<'a>>,
    want: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    method: Option<&'a str>,
}

pub(crate) fn call_frame<'a>(
    target: &JsRef,
    method: Option<&'a str>,
    args: &'a [HostValue],
    want: Want,
) -> Result<CallFrame<'a>, JsError> {
    let args = args.iter().map(|value| wire::argument(target.handle(), value)).collect::<Result<_, _>>()?;
    Ok(CallFrame { op: "call", target: target.id(), args, want: want.name(), method })
}

pub(crate) fn get_frame(target: &JsRef, key: &str, want: Want) -> Map<String, Value> {
    frame(json!({ "op": "get", "target": target.id(), "key": key, "want": want.name() }))
}

pub(crate) fn set_frame(target: &JsRef, key: &str, value: &HostValue) -> Result<Map<String, Value>, JsError> {
    let value = wire::encode(target.handle(), value)?;
    Ok(frame(json!({ "op": "set", "target": target.id(), "key": key, "value": value, "want": "unit" })))
}

pub(crate) fn json_frame(target: &JsRef) -> Map<String, Value> {
    frame(json!({ "op": "json", "target": target.id(), "want": "json" }))
}

pub(crate) async fn ask(
    handle: &JsHandle,
    frame: Result<impl serde::Serialize, JsError>,
) -> Result<HostValue, JsError> {
    let client = handle.node_client().ok_or_else(shut_down)?;
    let (sender, mut receiver) = tokio::sync::oneshot::channel();
    client.send_request(&handle.shared().alive, frame?, Waiter::Async(sender))?;
    loop {
        // The timer belongs to the runtime's own tokio, whatever executor polls this future.
        let poll = {
            let _entered = handle.shared().tokio.enter();
            tokio::time::timeout(LIVENESS_POLL, &mut receiver)
        };
        match poll.await {
            Ok(answer) => return answer.map_err(|_| client.exited())?,
            Err(_) => {
                if let Some(error) = client.fail_if_exited() {
                    return Err(error);
                }
            }
        }
    }
}

/// [`ask`] for synchronous callers; fails on the bridge thread, which must read the reply.
pub(crate) fn blocking_ask(
    handle: &JsHandle,
    frame: Result<impl serde::Serialize, JsError>,
    timeout: Option<Duration>,
) -> Result<HostValue, JsError> {
    handle.refuse_on_js_thread()?;
    let client = handle.node_client().ok_or_else(shut_down)?;
    let (sender, receiver) = mpsc::channel();
    client.send_request(&handle.shared().alive, frame?, Waiter::Blocking(sender))?;
    let deadline = timeout.map(|timeout| Instant::now() + timeout);
    loop {
        let wait = deadline
            .map_or(LIVENESS_POLL, |deadline| deadline.saturating_duration_since(Instant::now()).min(LIVENESS_POLL));
        match receiver.recv_timeout(wait) {
            Ok(result) => return result,
            Err(mpsc::RecvTimeoutError::Disconnected) => return Err(client.exited()),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Some(error) = client.fail_if_exited() {
                    return Err(error);
                }
                if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                    return Err(JsError::new("JavaScript call timed out"));
                }
            }
        }
    }
}

pub(crate) fn as_ref(value: HostValue) -> Result<JsRef, JsError> {
    match value {
        HostValue::Ref(reference) => Ok(reference),
        _ => Err(JsError::new("the Node worker returned a value where a reference was expected")),
    }
}

pub(crate) fn as_opt_ref(value: HostValue) -> Result<Option<JsRef>, JsError> {
    match value {
        HostValue::Undefined => Ok(None),
        other => as_ref(other).map(Some),
    }
}
