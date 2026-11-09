//! The Node backend's transport: raw output on the child's stdout, shutdown with grandchildren
//! holding stdout, a killed child, and the bootstrap directory.

mod common;
mod node_common;

use std::time::{Duration, Instant};

use common::*;
use node_common::*;
use pi_js::{HostFunction, JsRuntime};
use serde_json::json;

async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(15), future).await.expect("the call returned in time")
}

#[tokio::test]
async fn frames_preserve_partitioned_unicode_and_message_order() {
    let (_dir, root) = tempdir();
    let Some((runtime, _capture)) = node_runtime(&root) else { return };
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let reader = manifest.join("src/node_backend/js/frames.cjs");
    let fixture = manifest.join("tests/fixtures/node_backend/framing.cjs");
    let source = format!(
        "require('node:fs').writeFileSync('stage', 'framing'); return require({})(require({}));",
        json!(fixture),
        json!(reader)
    );
    let result = run_cjs_within(&runtime, &root, &source, 15).await;
    assert_eq!(result, json!({ "partitions": 6, "frames": 2, "characters": 2359296 }));
}

#[tokio::test]
async fn large_host_replies_preserve_sync_async_and_following_calls() {
    let (_dir, root) = tempdir();
    let Some((runtime, _capture)) = node_runtime(&root) else { return };
    let pattern = "历史🙂e\u{301}\\\"\n";
    let count = 262_144;
    let payload = json!({ "text": pattern.repeat(count), "values": [false, 0, null] });
    let sync = payload.clone();
    runtime.register_host_function("large", HostFunction::sync(move |_| Ok(sync.clone().into())));
    runtime.register_host_function(
        "largeAsync",
        HostFunction::asynchronous(move |_| {
            let value = payload.clone();
            async move { Ok(value.into()) }
        }),
    );
    let source = format!(
        r#"const fs = require('node:fs');
const host = (await import('pi:host')).default;
const expected = {}.repeat({count});
const results = [];
for (const name of ['large', 'largeAsync', 'large']) {{
  fs.writeFileSync('stage', name);
  const reply = await host.call(name);
  results.push([reply.text === expected, reply.values]);
}}
return results;"#,
        json!(pattern)
    );
    let results = run_cjs_within(&runtime, &root, &source, 15).await;
    assert_eq!(results, json!([[true, [false, 0, null]], [true, [false, 0, null]], [true, [false, 0, null]]]));
    assert_eq!(within(runtime.eval("1 + 1")).await.unwrap().into_json(), json!(2));
    drop(runtime);
}

#[tokio::test]
async fn raw_writes_to_stdout_do_not_break_calls() {
    let (_dir, root) = tempdir();
    let Some((runtime, capture)) = node_runtime(&root) else { return };
    let spawned = r#"process.getBuiltinModule("node:child_process")
  .spawnSync("sh", ["-c", "printf garbage; printf oops >&2"], { stdio: "inherit" }).status"#;
    assert_eq!(within(runtime.eval(spawned)).await.unwrap().into_json(), json!(0));
    let direct = r#"process.getBuiltinModule("node:fs").writeSync(1, "direct{\"t\":\"res\"}\n")"#;
    within(runtime.eval(direct)).await.unwrap();
    assert_eq!(within(runtime.eval("1 + 1")).await.unwrap().into_json(), json!(2));
    within(wait_for(|| capture.stdout().contains("garbage") && capture.stdout().contains("direct{"))).await;
    within(wait_for(|| capture.stderr().contains("oops"))).await;
}

#[tokio::test]
async fn drop_does_not_wait_for_a_grandchild_holding_stdout() {
    let (_dir, root) = tempdir();
    let Some((runtime, _capture)) = node_runtime(&root) else { return };
    let spawn = r#"const child = process.getBuiltinModule("node:child_process")
  .spawn("sleep", ["6"], { stdio: "inherit", detached: true });
child.unref();
child.pid"#;
    let pid = within(runtime.eval(spawn)).await.unwrap().into_json();
    let started = Instant::now();
    drop(runtime);
    let elapsed = started.elapsed();
    let _ = std::process::Command::new("kill").arg(pid.to_string()).status();
    assert!(elapsed < Duration::from_millis(900), "drop took {elapsed:?}");
}

fn kill_node(pid: &serde_json::Value) {
    let status = std::process::Command::new("kill").args(["-9", &pid.to_string()]).status().unwrap();
    assert!(status.success());
}

#[tokio::test]
async fn killing_node_fails_async_and_blocking_calls() {
    let (_dir, root) = tempdir();
    let Some((runtime, _capture)) = node_runtime(&root) else { return };
    let pid = within(runtime.eval("process.pid")).await.unwrap().into_json();
    let source = "export default { spin() { const end = Date.now() + 30000; while (Date.now() < end) {} } };";
    within(runtime.eval_module("busy.mjs", source)).await.unwrap();
    let busy = within(runtime.import_default("./busy.mjs")).await.unwrap();
    let pending = runtime.eval("new Promise(() => {})");
    let blocking = std::thread::spawn(move || busy.blocking_call_method("spin", vec![]));
    let killer = async {
        tokio::time::sleep(Duration::from_millis(200)).await;
        kill_node(&pid);
    };
    let (pending, ()) = within(async { tokio::join!(pending, killer) }).await;
    assert_eq!(pending.unwrap_err().code.as_deref(), Some("ERR_PI_NODE_EXITED"));
    let error = blocking.join().unwrap().unwrap_err();
    assert_eq!(error.code.as_deref(), Some("ERR_PI_NODE_EXITED"), "{error:?}");
    assert!(within(runtime.eval("1")).await.is_err());
}

#[cfg(unix)]
#[tokio::test]
async fn bootstrap_directory_is_private() {
    use std::os::unix::fs::PermissionsExt;
    let (_dir, root) = tempdir();
    let capture = std::sync::Arc::new(Capture::default());
    let mut options = options(&root, &capture);
    let cache = root.join("agent/cache/pi-js-node");
    options.cache_dir = Some(cache.clone());
    options.tokio = Some(tokio::runtime::Handle::current());
    let Some(runtime) = node_runtime_with(options.clone()) else { return };
    assert_eq!(within(runtime.eval("1")).await.unwrap().into_json(), json!(1));
    drop(runtime);
    let mode = |path: &std::path::Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&cache), 0o700);
    let scripts = std::fs::read_dir(&cache).unwrap().map(|entry| entry.unwrap().path());
    let scripts: Vec<_> =
        scripts.filter(|path| path.file_name().unwrap().to_string_lossy().starts_with("node-")).collect();
    assert_eq!(scripts.len(), 1);
    assert_eq!(mode(&scripts[0]), 0o700);

    std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o777)).unwrap();
    options.backend = pi_js::JsBackend::Node { executable: node().unwrap() };
    let error = JsRuntime::new(options).err().expect("a shared bootstrap directory is refused");
    assert_eq!(error.code.as_deref(), Some("ERR_PI_NODE_START"));
    assert!(error.message.contains("permissions"), "{error:?}");
}
