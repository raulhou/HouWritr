//! A language model that runs on this machine, for AI text cleanup and meeting
//! notes.
//!
//! OpenWritr drives Qualcomm's GenieX command-line tool (`geniex serve` is an
//! OpenAI-compatible server that runs AI Hub models on the Hexagon NPU) as a
//! private child process, and the normal OpenAI-compatible cleanup pipeline
//! talks to it over loopback. Nothing is bound in-process: GenieX is a
//! developer-preview SDK, so keeping it behind a process boundary means a
//! crash or an ABI change can never take the app down.
//!
//! What this module owns:
//! * finding (or, with the user's consent, installing) the official GenieX CLI;
//! * downloading / removing the model with `geniex pull` / `geniex remove`;
//! * starting `geniex serve` on a private port when a request needs it, and
//!   stopping it again when idle or when OpenWritr exits.
//!
//! The server is only ever started for the *managed* endpoint
//! ([`MANAGED_BASE_URL`]). Any other loopback address (Ollama, LM Studio, a
//! `geniex serve` the user runs themselves) is simply talked to.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

use crate::cleanup::EndpointScope;

/// Private port for the server OpenWritr starts. Deliberately not GenieX's
/// default (18181) so it never collides with a server the user runs.
pub const MANAGED_PORT: u16 = 18651;
pub const MANAGED_BASE_URL: &str = "http://127.0.0.1:18651/v1";
/// The built-in choice a fresh setup starts with: the faster one, which
/// peaked below 50% CPU on the test PC with other work in the foreground.
pub const DEFAULT_MODEL: &str = "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_0@hybrid";

/// Models from before the list was cut to the two llama.cpp choices:
/// uninstalling GenieX still removes them if they are on disk.
const LEGACY_MODELS: &[&str] = &[
    "ai-hub-models/Qwen3-4B-Instruct-2507",
    "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_0",
];

/// Where GenieX runs a built-in model: its llama.cpp runtime, on the NPU
/// through the Hexagon backend, or split between the NPU and the CPU. (The
/// precompiled Qualcomm bundle wrote about one token a second on the test PC
/// and is gone; the CPU alone is not offered.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Runtime {
    /// All layers on the NPU.
    NpuLlama,
    /// Split between the NPU and the CPU by llama.cpp, which GenieX calls
    /// the fast path on Snapdragon.
    Hybrid,
}

impl Runtime {
    /// For the transcript window and the log.
    pub fn place(self) -> &'static str {
        match self {
            Runtime::NpuLlama => "NPU, llama.cpp",
            Runtime::Hybrid => "NPU + CPU",
        }
    }

    /// GenieX's `compute` request field.
    fn compute(self) -> &'static str {
        match self {
            Runtime::NpuLlama => "npu",
            Runtime::Hybrid => "hybrid",
        }
    }
}

/// A model HouWritr can download and run through GenieX by itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuiltInModel {
    /// What the settings file stores; unique per choice.
    pub id: &'static str,
    /// The name GenieX downloads and serves it under (`name[:precision]`).
    /// Both choices share one file and differ only in `runtime`.
    pub name: &'static str,
    /// Shown in Settings.
    pub label: &'static str,
    /// One line for Settings: size and what to expect.
    pub detail: &'static str,
    pub runtime: Runtime,
    /// Download size in bytes: GenieX's own total can be far too small (it
    /// sizes its bar from the first file it sees).
    pub download_bytes: Option<u64>,
}

const QWEN3_4B_GGUF: &str = "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_0";
const QWEN3_4B_GGUF_BYTES: u64 = 2_375_773_280;

/// The models offered in Settings: one Qwen3 4B file from Hugging Face, run
/// two ways.
pub const BUILT_IN_MODELS: &[BuiltInModel] = &[
    BuiltInModel {
        id: DEFAULT_MODEL,
        name: QWEN3_4B_GGUF,
        label: "Qwen3 4B on the NPU and CPU",
        detail: "About 2.4 GB from Hugging Face. Split between the NPU and the processor by llama.cpp: the faster choice (about 18 tokens a second on a Snapdragon X Elite).",
        runtime: Runtime::Hybrid,
        download_bytes: Some(QWEN3_4B_GGUF_BYTES),
    },
    BuiltInModel {
        id: "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_0@npu",
        name: QWEN3_4B_GGUF,
        label: "Qwen3 4B on the NPU only (llama.cpp)",
        detail: "The same file on the NPU alone: about 13 tokens a second, the lightest on the CPU.",
        runtime: Runtime::NpuLlama,
        download_bytes: Some(QWEN3_4B_GGUF_BYTES),
    },
];

/// The built-in model the setting `model` names, if it is one.
pub fn built_in(model: &str) -> Option<&'static BuiltInModel> {
    let model = model.trim();
    BUILT_IN_MODELS
        .iter()
        .find(|known| known.id.eq_ignore_ascii_case(model))
}

/// The choice a request for setting `model` uses: the one it names, or the
/// first for a setting from before the list was cut down.
fn chosen(model: &str) -> &'static BuiltInModel {
    built_in(model).unwrap_or(&BUILT_IN_MODELS[0])
}

/// The name GenieX knows the model of setting `model` by.
pub fn geniex_name(model: &str) -> &'static str {
    chosen(model).name
}

/// Where the model of setting `model` runs.
pub fn runtime(model: &str) -> Runtime {
    chosen(model).runtime
}

/// Make a chat request for setting `model` one GenieX runs where it should:
/// the model's GenieX name and compute unit, all layers offloaded.
pub fn add_request_options(body: &mut serde_json::Value, model: &str) {
    let Some(body) = body.as_object_mut() else {
        return;
    };
    let choice = chosen(model);
    body.insert("model".into(), serde_json::json!(choice.name));
    body.insert(
        "compute".into(),
        serde_json::json!(choice.runtime.compute()),
    );
    body.insert("ngl".into(), serde_json::json!(-1));
    // GenieX asks models to think first unless told otherwise; this
    // instruct model does not, and must not start to.
    body.insert("enable_think".into(), serde_json::json!(false));
}

/// The arguments `geniex pull` needs for setting `model`, without prompts.
fn pull_arguments(model: &str) -> Vec<&str> {
    // A Hugging Face download asks whether the model reads images.
    vec!["pull", geniex_name(model), "--model-type", "llm"]
}

/// Pinned official installer (GenieX v0.8.0, Windows ARM64, Inno Setup).
const INSTALLER_URL: &str = "https://github.com/qualcomm/GenieX/releases/download/v0.8.0/geniex-cli-setup-windows-arm64-v0.8.0.exe";
const INSTALLER_SHA256: &str = "fcc4d932f371a9b3ad3d08a1b34912fe21a7fdf7c54d47c869250b50d81ec734";
const INSTALLER_BYTES: u64 = 65_196_910;

/// How long a freshly started server may take to answer.
const START_TIMEOUT: Duration = Duration::from_secs(90);
/// After a failed start, further requests fail fast for this long instead of
/// each waiting out [`START_TIMEOUT`].
const RETRY_AFTER_FAILURE: Duration = Duration::from_secs(60);
/// The server (and with it the NPU) is released after this much idleness.
/// GenieX drops an idle model after five minutes anyway, so keeping the
/// process longer only holds memory for a server that must reload first.
const IDLE_STOP: Duration = Duration::from_secs(5 * 60);

const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Whether `endpoint` is the server OpenWritr manages itself.
pub fn is_managed(endpoint: &EndpointScope) -> bool {
    endpoint.is_loopback()
        && endpoint.port() == Some(MANAGED_PORT)
        && endpoint.scheme() == "http"
        && endpoint.path() == "/v1"
}

/// [`is_managed`] for a raw settings string.
pub fn is_managed_url(raw: &str) -> bool {
    EndpointScope::parse(raw).is_ok_and(|scope| is_managed(&scope))
}

/// Why the managed local model cannot be used right now.
#[derive(Debug, thiserror::Error)]
pub enum LocalLlmError {
    #[error("{0}")]
    Unsupported(String),
    #[error("the GenieX command-line tool is not installed")]
    CliMissing,
    #[error("the local model server did not start: {0}")]
    StartFailed(String),
    #[error("the local model server failed to start recently; not retrying yet")]
    RecentlyFailed,
}

/// The managed local model needs Qualcomm's NPU runtime, which only exists for
/// Windows on ARM64.
pub fn platform_support() -> std::result::Result<(), String> {
    platform_support_for(
        cfg!(target_arch = "aarch64"),
        std::env::var("PROCESSOR_ARCHITEW6432").ok().as_deref(),
    )
}

fn platform_support_for(
    native_arm64: bool,
    emulated_host: Option<&str>,
) -> std::result::Result<(), String> {
    if native_arm64 || emulated_host.is_some_and(|host| host.eq_ignore_ascii_case("ARM64")) {
        Ok(())
    } else {
        Err(
            "The built-in local model runs on Qualcomm Snapdragon (Windows on ARM). \
             On this PC use \"Another local server\" instead."
                .into(),
        )
    }
}

// ---------------------------------------------------------------------------
// Finding the CLI
// ---------------------------------------------------------------------------

fn cli_candidates(
    local_app_data: Option<&Path>,
    program_files: Option<&Path>,
    path_var: Option<&std::ffi::OsStr>,
) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(dir) = local_app_data {
        out.push(dir.join("GenieX CLI").join("geniex.exe"));
    }
    if let Some(dir) = program_files {
        out.push(dir.join("GenieX CLI").join("geniex.exe"));
    }
    if let Some(path) = path_var {
        out.extend(std::env::split_paths(path).map(|dir| dir.join("geniex.exe")));
    }
    out
}

/// The installed GenieX CLI, if any.
pub fn locate_cli() -> Option<PathBuf> {
    let local = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    let program_files = std::env::var_os("ProgramFiles").map(PathBuf::from);
    let path = std::env::var_os("PATH");
    cli_candidates(local.as_deref(), program_files.as_deref(), path.as_deref())
        .into_iter()
        .find(|candidate| candidate.is_file())
}

fn work_dir() -> PathBuf {
    crate::paths::models_dir()
        .parent()
        .map(|base| base.join("local-llm"))
        .unwrap_or_else(|| PathBuf::from("local-llm"))
}

fn hidden(command: &mut Command) -> &mut Command {
    use std::os::windows::process::CommandExt;
    command.creation_flags(CREATE_NO_WINDOW)
}

// ---------------------------------------------------------------------------
// Installing the CLI (explicit user action)
// ---------------------------------------------------------------------------

/// Download the pinned official GenieX installer, verify it, and run it
/// silently for the current user. The installer adds `geniex` to the user's
/// PATH; nothing is installed machine-wide and no administrator rights are
/// needed. `progress` receives a 0..=1 fraction while downloading.
pub fn install_cli(progress: &mut dyn FnMut(f32), cancel: &AtomicBool) -> Result<PathBuf> {
    platform_support().map_err(|reason| anyhow!(reason))?;
    let dir = work_dir();
    std::fs::create_dir_all(&dir).context("create the local model folder")?;
    let part = dir.join("geniex-cli-setup.exe.part");
    let installer = dir.join("geniex-cli-setup.exe");

    download_verified(
        INSTALLER_URL,
        INSTALLER_SHA256,
        INSTALLER_BYTES,
        &part,
        progress,
        cancel,
    )?;
    std::fs::rename(&part, &installer).context("finish the installer download")?;

    info!("running the GenieX installer");
    let status = hidden(Command::new(&installer).args([
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/NOICONS",
        "/TASKS=",
    ]))
    .status()
    .context("start the GenieX installer")?;
    let _ = std::fs::remove_file(&installer);
    if !status.success() {
        bail!("the GenieX installer exited with {status}");
    }
    locate_cli().ok_or_else(|| anyhow!("GenieX was installed but geniex.exe was not found"))
}

fn download_verified(
    url: &str,
    sha256: &str,
    expected_bytes: u64,
    dest: &Path,
    progress: &mut dyn FnMut(f32),
    cancel: &AtomicBool,
) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .timeout(None)
        .build()?;
    let mut response = client.get(url).send()?.error_for_status()?;
    let total = response.content_length().unwrap_or(expected_bytes).max(1);
    let mut file = std::fs::File::create(dest).context("create the download file")?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 256 * 1024];
    let mut done = 0_u64;
    loop {
        if cancel.load(Ordering::Relaxed) {
            drop(file);
            let _ = std::fs::remove_file(dest);
            bail!("cancelled");
        }
        let read = response.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        done += read as u64;
        if done > expected_bytes {
            drop(file);
            let _ = std::fs::remove_file(dest);
            bail!("the download is larger than expected");
        }
        hasher.update(&buffer[..read]);
        file.write_all(&buffer[..read])?;
        progress((done as f32 / total as f32).min(1.0));
    }
    file.flush()?;
    drop(file);
    let digest = hex(&hasher.finalize());
    if !digest.eq_ignore_ascii_case(sha256) {
        let _ = std::fs::remove_file(dest);
        bail!("the download does not match the expected checksum");
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Model management
// ---------------------------------------------------------------------------

/// Whether `model` is already downloaded, according to `geniex list`.
pub fn model_installed(cli: &Path, model: &str) -> Result<bool> {
    let output = hidden(Command::new(cli).args(["list", "--format", "json"]))
        .output()
        .context("run geniex list")?;
    if !output.status.success() {
        bail!("geniex list failed ({})", output.status);
    }
    Ok(lists_model(&String::from_utf8_lossy(&output.stdout), model))
}

/// Whether `listing` (`geniex list --format json`) contains `model`.
///
/// GenieX stores `ai-hub-models/<name>` under its canonical `qualcomm/<name>`,
/// so names are compared by their last path segment, ignoring case and any
/// `:precision` suffix.
fn lists_model(listing: &str, model: &str) -> bool {
    fn repo(name: &str) -> String {
        let name = name.split(':').next().unwrap_or(name);
        name.rsplit('/')
            .next()
            .unwrap_or(name)
            .trim()
            .to_ascii_lowercase()
    }
    let wanted = repo(model);
    if wanted.is_empty() {
        return false;
    }
    match serde_json::from_str::<serde_json::Value>(listing) {
        Ok(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|item| item.get("name").and_then(|name| name.as_str()))
            .any(|name| repo(name) == wanted),
        // Unexpected output: fall back to a plain search for the bare name.
        _ => listing.to_ascii_lowercase().contains(&wanted),
    }
}

/// Download attempts before a download is reported as failed.
const PULL_ATTEMPTS: usize = 5;
/// Seconds to wait before attempt 2, 3, … (a dropped connection often needs
/// a moment).
const PULL_RETRY_WAITS: [u64; PULL_ATTEMPTS - 1] = [5, 15, 30, 60];

/// Download `model` with `geniex pull`. `progress` gets the fraction done
/// (when it is known) and one line saying how far the download is. A
/// download that stops is resumed by itself (GenieX keeps what it has) up to
/// [`PULL_ATTEMPTS`] times; cancelling keeps what was downloaded so far.
pub fn pull_model(
    cli: &Path,
    model: &str,
    progress: &mut dyn FnMut(Option<f32>, &str),
    cancel: &AtomicBool,
) -> Result<()> {
    let mut attempt = 1;
    loop {
        let error = match pull_once(cli, model, progress, cancel) {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        if attempt >= PULL_ATTEMPTS || !worth_retrying(&error) {
            return Err(error);
        }
        let wait = PULL_RETRY_WAITS[attempt - 1];
        attempt += 1;
        warn!(error = %format!("{error:#}"), attempt, "model download stopped; resuming");
        for left in (1..=wait).rev() {
            if cancel.load(Ordering::Relaxed) {
                bail!("cancelled");
            }
            progress(
                None,
                &format!(
                    "The download stopped; resuming in {left} s (attempt {attempt} of {PULL_ATTEMPTS})…"
                ),
            );
            std::thread::sleep(Duration::from_secs(1));
        }
    }
}

/// Whether trying a failed download again could help: not when it was
/// cancelled, or failed for a reason a retry does not change.
fn worth_retrying(error: &anyhow::Error) -> bool {
    let text = format!("{error:#}").to_ascii_lowercase();
    const FINAL: &[&str] = &[
        "cancelled",
        "start geniex pull",
        "no precision",
        "not found",
        "404",
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "gated",
        "no space",
        "not enough space",
        "disk full",
        "unknown model type",
    ];
    !FINAL.iter().any(|word| text.contains(word))
}

fn pull_once(
    cli: &Path,
    model: &str,
    progress: &mut dyn FnMut(Option<f32>, &str),
    cancel: &AtomicBool,
) -> Result<()> {
    let expected = built_in(model).and_then(|known| known.download_bytes);
    let mut child = hidden(
        Command::new(cli)
            .args(pull_arguments(model))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped()),
    )
    .spawn()
    .context("start geniex pull")?;

    let (tx, rx) = mpsc::channel::<String>();
    for stream in [
        child
            .stdout
            .take()
            .map(|s| Box::new(s) as Box<dyn Read + Send>),
        child
            .stderr
            .take()
            .map(|s| Box::new(s) as Box<dyn Read + Send>),
    ]
    .into_iter()
    .flatten()
    {
        let tx = tx.clone();
        std::thread::spawn(move || split_progress_lines(stream, &tx));
    }
    drop(tx);

    // The last few lines the tool printed: when it fails, they say why.
    let mut recent: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            bail!("cancelled");
        }
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(line) => {
                let clean = strip_ansi(&line);
                let clean = clean.trim();
                if !clean.is_empty() {
                    match parse_pull_progress(clean) {
                        Some(bar) => {
                            let (fraction, text) = describe_pull(&bar, expected);
                            progress(fraction, &text);
                        }
                        None => progress(None, clean),
                    }
                    if parse_pull_progress(clean).is_none() {
                        info!("geniex pull: {clean}");
                        recent.push_back(clean.to_string());
                        if recent.len() > 4 {
                            recent.pop_front();
                        }
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let status = child.wait()?;
    if !status.success() {
        let detail = recent.into_iter().collect::<Vec<_>>().join(" | ");
        if detail.is_empty() {
            bail!("geniex pull failed ({status})");
        }
        bail!("geniex pull failed ({status}): {detail}");
    }
    Ok(())
}

pub fn remove_model(cli: &Path, model: &str) -> Result<()> {
    let status = hidden(Command::new(cli).args(["remove", model, "--yes"]))
        .status()
        .context("run geniex remove")?;
    if !status.success() {
        bail!("geniex remove failed ({status})");
    }
    Ok(())
}

/// Uninstall GenieX: remove its model, run the uninstaller its installer
/// left next to `geniex.exe` (silently, for this user), and delete the files
/// HouWritr kept for it. A GenieX installed some other way has no such
/// uninstaller and is left to Windows Settings → Apps.
pub fn uninstall_cli(cli: &Path) -> Result<()> {
    shutdown();
    let mut names: Vec<&str> = BUILT_IN_MODELS.iter().map(|model| model.name).collect();
    names.extend(LEGACY_MODELS);
    names.sort_unstable();
    names.dedup();
    for name in names {
        if model_installed(cli, name).unwrap_or(true) {
            if let Err(error) = remove_model(cli, name) {
                warn!(error = %error, model = name, "could not remove a GenieX model before uninstalling");
            }
        }
    }
    let uninstaller = find_uninstaller(cli).ok_or_else(|| {
        anyhow!(
            "no GenieX uninstaller was found next to {}; remove GenieX in Windows Settings → Apps",
            cli.display()
        )
    })?;
    info!(uninstaller = %uninstaller.display(), "running the GenieX uninstaller");
    let status =
        hidden(Command::new(&uninstaller).args(["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]))
            .status()
            .context("start the GenieX uninstaller")?;
    if !status.success() {
        bail!("the GenieX uninstaller exited with {status}");
    }
    let _ = std::fs::remove_dir_all(work_dir());
    Ok(())
}

/// The Inno Setup uninstaller (`unins000.exe`, …) in the folder of
/// `geniex.exe` or the folder above it.
fn find_uninstaller(cli: &Path) -> Option<PathBuf> {
    cli.ancestors().skip(1).take(2).find_map(|folder| {
        let mut found: Vec<PathBuf> = std::fs::read_dir(folder)
            .ok()?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| is_uninstaller_name(path))
            .collect();
        found.sort();
        found.into_iter().next()
    })
}

fn is_uninstaller_name(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_ascii_lowercase())
        .is_some_and(|name| name.starts_with("unins") && name.ends_with(".exe"))
}

/// Emit text split on `\r` and `\n` (progress bars redraw with `\r`).
fn split_progress_lines(mut stream: Box<dyn Read + Send>, tx: &mpsc::Sender<String>) {
    let mut current = Vec::new();
    let mut byte = [0_u8; 1];
    while matches!(stream.read(&mut byte), Ok(1)) {
        if byte[0] == b'\r' || byte[0] == b'\n' {
            if !current.is_empty() {
                let _ = tx.send(String::from_utf8_lossy(&current).into_owned());
                current.clear();
            }
        } else {
            current.push(byte[0]);
        }
    }
    if !current.is_empty() {
        let _ = tx.send(String::from_utf8_lossy(&current).into_owned());
    }
}

fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for next in chars.by_ref() {
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// One redraw of GenieX's download bar, e.g.
/// `downloading  45% |████      | (1.1/2.4 GB, 25 MB/s) [40s:52s]`.
/// Only the bytes downloaded and the speed are trusted: GenieX sizes the bar
/// (its percentage, total and time left) from the first file it sees, so it
/// can read 100% early on.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PullProgress {
    downloaded: f64,
    /// Bytes a second.
    rate: Option<f64>,
    /// GenieX's total, which may be too small.
    total: Option<f64>,
}

fn parse_pull_progress(line: &str) -> Option<PullProgress> {
    let line = line.trim();
    if !line.starts_with("downloading") {
        return None;
    }
    let open = line.find('(')?;
    let close = open + line[open..].find(')')?;
    let mut parts = line[open + 1..close].split(", ");
    let amount = parts.next()?;
    let (current, total) = match amount.split_once('/') {
        Some((current, total)) => (current.trim(), Some(total.trim())),
        None => (amount.trim(), None),
    };
    let total_unit = total.and_then(|total| total.rsplit_once(' ').map(|(_, unit)| unit));
    let downloaded =
        parse_bytes(current).or_else(|| parse_bytes(&format!("{current} {}", total_unit?)))?;
    let rate = parts.find_map(|part| part.trim().strip_suffix("/s").and_then(parse_bytes));
    Some(PullProgress {
        downloaded,
        rate,
        total: total.and_then(parse_bytes),
    })
}

/// `2.4 GB`, `512 MB`, `1.5 GiB`, ` 7 B` as bytes.
fn parse_bytes(text: &str) -> Option<f64> {
    let (number, unit) = text.trim().rsplit_once(' ')?;
    let value: f64 = number.trim().parse().ok()?;
    let scale = match unit {
        "B" => 1.0,
        "kB" => 1e3,
        "MB" => 1e6,
        "GB" => 1e9,
        "TB" => 1e12,
        "KiB" => 1024.0,
        "MiB" => 1024.0 * 1024.0,
        "GiB" => 1024.0 * 1024.0 * 1024.0,
        "TiB" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    Some(value * scale)
}

/// The fraction done and one line for Settings, e.g.
/// `45% · 1.1 of 2.4 GB · 25.0 MB/s · about 1 min left`.
fn describe_pull(bar: &PullProgress, expected: Option<u64>) -> (Option<f32>, String) {
    let total = expected
        .map(|bytes| bytes as f64)
        .or(bar.total.filter(|total| *total > bar.downloaded));
    // Never "done" before GenieX says so: the size can be a little off.
    let fraction = total.map(|total| (bar.downloaded / total).clamp(0.0, 0.99) as f32);
    let mut parts = Vec::new();
    if let Some(fraction) = fraction {
        parts.push(format!("{:.0}%", fraction * 100.0));
    }
    parts.push(match total {
        Some(total) => format!("{} of {}", human_bytes(bar.downloaded), human_bytes(total)),
        None => format!("{} downloaded", human_bytes(bar.downloaded)),
    });
    if let Some(rate) = bar.rate.filter(|rate| *rate > 0.0) {
        parts.push(format!("{}/s", human_bytes(rate)));
        if let Some(total) = total {
            let seconds = ((total - bar.downloaded).max(0.0) / rate).round() as u64;
            parts.push(time_left(seconds));
        }
    }
    (fraction, parts.join(" · "))
}

fn human_bytes(bytes: f64) -> String {
    const UNITS: [&str; 4] = ["kB", "MB", "GB", "TB"];
    let mut value = bytes / 1e3;
    let mut unit = 0;
    while value >= 1000.0 && unit + 1 < UNITS.len() {
        value /= 1000.0;
        unit += 1;
    }
    if bytes < 1e3 {
        return format!("{bytes:.0} B");
    }
    format!("{value:.1} {}", UNITS[unit])
}

fn time_left(seconds: u64) -> String {
    match seconds {
        0..=59 => format!("about {seconds} s left"),
        60..=3599 => format!("about {} min left", (seconds + 30) / 60),
        _ => format!(
            "about {} h {} min left",
            seconds / 3600,
            seconds % 3600 / 60
        ),
    }
}

// ---------------------------------------------------------------------------
// The server process
// ---------------------------------------------------------------------------

struct Server {
    child: Option<Child>,
    last_used: Instant,
    failed_at: Option<Instant>,
    reaper_running: bool,
    /// The model has answered at least once since the server started.
    model_loaded: bool,
    /// Which model the server was last asked to load.
    loaded_model: Option<String>,
    /// The model has written at normal speed since the server started.
    was_fast: bool,
    /// Slow answers in a row from this server.
    slow_answers: u32,
}

static SERVER: Lazy<Mutex<Server>> = Lazy::new(|| {
    Mutex::new(Server {
        child: None,
        last_used: Instant::now(),
        failed_at: None,
        reaper_running: false,
        model_loaded: false,
        loaded_model: None,
        was_fast: false,
        slow_answers: 0,
    })
});

/// Make sure a request for `model` to `endpoint` can be answered (with no
/// model, only that the server runs). A no-op for every endpoint other than
/// the managed one.
pub fn ensure_running(
    endpoint: &EndpointScope,
    model: Option<&str>,
) -> std::result::Result<(), LocalLlmError> {
    if !is_managed(endpoint) {
        return Ok(());
    }
    let mut server = SERVER.lock();
    // Another built-in model is wanted: stop the server so the one it holds
    // does not stay in memory next to the new one.
    if let (Some(wanted), Some(loaded)) = (model, server.loaded_model.as_deref()) {
        if !wanted.trim().eq_ignore_ascii_case(loaded) && server.child.is_some() {
            info!(
                from = loaded,
                to = wanted.trim(),
                "switching the local model"
            );
            stop_locked(&mut server);
        }
    }
    let previous_use = server.last_used;
    server.last_used = Instant::now();

    let running = match server.child.as_mut().map(Child::try_wait) {
        Some(Ok(None)) => true,
        Some(Ok(Some(status))) => {
            warn!(%status, "local model server exited; restarting");
            server.child = None;
            server.model_loaded = false;
            server.loaded_model = None;
            server.was_fast = false;
            server.slow_answers = 0;
            false
        }
        Some(_) => {
            warn!("local model server exited; restarting");
            server.child = None;
            server.model_loaded = false;
            server.was_fast = false;
            server.slow_answers = 0;
            false
        }
        None => false,
    };
    if !running {
        // Something else (a server the user started) already answers there.
        if !server_answers(Duration::from_millis(500)) {
            if server
                .failed_at
                .is_some_and(|at| at.elapsed() < RETRY_AFTER_FAILURE)
            {
                return Err(LocalLlmError::RecentlyFailed);
            }
            match start_server() {
                Ok(child) => {
                    server.child = Some(child);
                    server.model_loaded = false;
                    server.was_fast = false;
                    server.slow_answers = 0;
                    if !server.reaper_running {
                        server.reaper_running = true;
                        std::thread::spawn(reap_when_idle);
                    }
                }
                Err(error) => {
                    server.failed_at = Some(Instant::now());
                    return Err(error);
                }
            }
        }
    }

    // The server answers before its model is on the NPU, and GenieX unloads
    // the model after five idle minutes. The first request after either has
    // to wait for the load, which takes far longer than a cleanup request may
    // wait, so load it here, once, before any real request is sent.
    let Some(model) = model.map(str::trim).filter(|model| !model.is_empty()) else {
        server.failed_at = None;
        return Ok(());
    };
    if !server.model_loaded || previous_use.elapsed() >= MODEL_RESIDENT {
        let place = runtime(model).place();
        info!(model, place, "loading the local model");
        let started = Instant::now();
        match load_model(model) {
            Ok(()) => {
                info!(took = ?started.elapsed(), "local model is loaded");
                server.model_loaded = true;
                server.loaded_model = Some(model.to_string());
                server.failed_at = None;
            }
            Err(error) => {
                warn!(error = %error, took = ?started.elapsed(), "loading the local model failed");
                server.model_loaded = false;
                server.failed_at = Some(Instant::now());
                return Err(LocalLlmError::StartFailed(format!(
                    "the model did not load: {error}"
                )));
            }
        }
    }
    server.failed_at = None;
    Ok(())
}

/// How long GenieX keeps an idle model loaded (its default is 300 s); a
/// little less, so a request just before the unload is not trusted.
const MODEL_RESIDENT: Duration = Duration::from_secs(240);
/// The longest a model load may take.
const LOAD_TIMEOUT: Duration = Duration::from_secs(600);

/// Make the server load `model` by asking it for one token.
fn load_model(model: &str) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(LOAD_TIMEOUT)
        .build()?;
    let mut body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
        "temperature": 0.0,
    });
    add_request_options(&mut body, model);
    let response = client
        .post(format!("{MANAGED_BASE_URL}/chat/completions"))
        .json(&body)
        .send()
        .context("no answer")?;
    if !response.status().is_success() {
        let status = response.status();
        let detail = response.text().unwrap_or_default();
        let detail: String = detail.chars().take(160).collect();
        bail!("the server answered with {status} {detail}");
    }
    Ok(())
}

fn start_server() -> std::result::Result<Child, LocalLlmError> {
    platform_support().map_err(LocalLlmError::Unsupported)?;
    let cli = locate_cli().ok_or(LocalLlmError::CliMissing)?;
    info!(cli = %cli.display(), "starting the local model server");
    let mut child = hidden(
        Command::new(&cli)
            .args(["serve", "--host", &format!("127.0.0.1:{MANAGED_PORT}")])
            .env("GENIEX_LOG", "info")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped()),
    )
    .spawn()
    .map_err(|error| LocalLlmError::StartFailed(error.to_string()))?;
    kill_with_parent(&child);
    {
        use std::os::windows::io::AsRawHandle;
        // The server drives the NPU from the CPU; a throttled server process
        // would feed it slowly.
        if !crate::power::opt_out_of_throttling(child.as_raw_handle() as _) {
            warn!(
                error = %std::io::Error::last_os_error(),
                "could not opt the local model server out of power throttling"
            );
        }
    }
    // Everything the server prints goes into OpenWritr's log, so a model
    // that fails or stalls can be diagnosed from one file.
    if let Some(stream) = child.stdout.take() {
        forward_output(stream, "stdout");
    }
    if let Some(stream) = child.stderr.take() {
        forward_output(stream, "stderr");
    }

    let started = Instant::now();
    while started.elapsed() < START_TIMEOUT {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(LocalLlmError::StartFailed(format!("exited with {status}")));
        }
        if server_answers(Duration::from_secs(2)) {
            info!(took = ?started.elapsed(), "local model server is ready");
            return Ok(child);
        }
        std::thread::sleep(Duration::from_millis(400));
    }
    let _ = child.kill();
    let _ = child.wait();
    Err(LocalLlmError::StartFailed("timed out".into()))
}

/// How loud a line of the server's output is in OpenWritr's log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LineLevel {
    /// Two lines per model step (hundreds per reply): only with
    /// `RUST_LOG=openwritr=debug`, where they time each part of the model.
    Debug,
    Info,
    /// The server or the NPU runtime reported an error.
    Warn,
}

fn server_line_level(text: &str) -> LineLevel {
    if text.contains("QnnGraph_execute") || text.contains("execution finished with result 0") {
        LineLevel::Debug
    } else if text.contains(" ERR ") || text.contains("<E>") {
        LineLevel::Warn
    } else {
        LineLevel::Info
    }
}

/// Copy a child's output into the log, one line at a time. Also keeps the
/// pipe drained so the server can never block on a full buffer.
fn forward_output(stream: impl Read + Send + 'static, name: &'static str) {
    let spawned = std::thread::Builder::new()
        .name(format!("geniex-{name}"))
        .spawn(move || {
            let mut reader = std::io::BufReader::new(stream);
            let mut line = Vec::new();
            loop {
                line.clear();
                match std::io::BufRead::read_until(&mut reader, b'\n', &mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let text = strip_ansi(&String::from_utf8_lossy(&line));
                        let text = text.trim();
                        if !text.is_empty() {
                            let text: String = text.chars().take(600).collect();
                            match server_line_level(&text) {
                                LineLevel::Debug => debug!(stream = name, "geniex: {text}"),
                                LineLevel::Info => info!(stream = name, "geniex: {text}"),
                                LineLevel::Warn => warn!(stream = name, "geniex: {text}"),
                            }
                        }
                    }
                }
            }
        });
    if let Err(error) = spawned {
        warn!(error = %error, "could not capture the local model server's output");
    }
}

fn server_answers(timeout: Duration) -> bool {
    reqwest::blocking::Client::builder()
        .timeout(timeout)
        .build()
        .and_then(|client| client.get(format!("{MANAGED_BASE_URL}/models")).send())
        .is_ok_and(|response| response.status().is_success())
}

fn reap_when_idle() {
    loop {
        std::thread::sleep(Duration::from_secs(30));
        let mut server = SERVER.lock();
        if server.child.is_none() {
            server.reaper_running = false;
            return;
        }
        if server.last_used.elapsed() >= IDLE_STOP {
            info!("local model server idle; stopping it");
            stop_locked(&mut server);
            server.reaper_running = false;
            return;
        }
    }
}

fn stop_locked(server: &mut Server) {
    if let Some(mut child) = server.child.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
    server.model_loaded = false;
    server.loaded_model = None;
    server.was_fast = false;
    server.slow_answers = 0;
}

/// After a request to the server failed without timing out: whether the
/// server HouWritr started has exited (waiting up to two seconds for a
/// process that is going down). Its exit status is logged, and the next
/// request starts a new server.
pub fn server_exited() -> bool {
    let Some(mut server) = SERVER.try_lock_for(Duration::from_secs(2)) else {
        return false;
    };
    let started = Instant::now();
    loop {
        let Some(child) = server.child.as_mut() else {
            return false;
        };
        match child.try_wait() {
            Ok(Some(status)) => {
                warn!(%status, "the local model server stopped by itself");
                server.child = None;
                stop_locked(&mut server);
                return true;
            }
            Ok(None) if started.elapsed() < Duration::from_secs(2) => {
                std::thread::sleep(Duration::from_millis(100));
            }
            _ => return false,
        }
    }
}

/// Stop the server OpenWritr started, now, so the NPU and its memory go back
/// to the rest of the system. The next request starts it again (and reloads
/// the model, about half a minute). Does nothing when OpenWritr did not start
/// a server, so a server the user runs is never touched. Also cancels
/// whatever the server is still generating, e.g. a reply nobody waits for
/// any more after a timeout.
///
/// Gives up quietly when another thread holds the server for longer than a
/// moment (it is loading the model for a new request); the idle stop then
/// releases it later.
pub fn release(reason: &str) {
    let Some(mut server) = SERVER.try_lock_for(Duration::from_secs(2)) else {
        info!(reason, "local model server is busy; leaving it running");
        return;
    };
    if server.child.is_some() {
        info!(reason, "stopping the local model server");
        stop_locked(&mut server);
    }
}

/// Writing speed (tokens per second) below which the built-in model counts as
/// slow. It normally writes about 18; in the slow state it wrote under 1.
pub(crate) const SLOW_WRITING_TOKENS_PER_SECOND: f64 = 4.0;

/// Whether a reply written at `tokens_per_second` should restart the server:
/// only when the same server wrote at normal speed before. That is the slow
/// state seen in practice (fast, then about 20 times slower for every later
/// request until restarted). A PC where the model is always slow gains
/// nothing from a restart, and each one costs a model load.
fn restart_for_speed(tokens_per_second: f64, was_fast: bool) -> bool {
    tokens_per_second < SLOW_WRITING_TOKENS_PER_SECOND && was_fast
}

/// Report how fast the built-in model wrote its last reply. Restarts the
/// server when it has fallen into the slow state, so the next request gets a
/// fresh one.
pub fn note_writing_speed(tokens_per_second: f64) {
    let Some(mut server) = SERVER.try_lock_for(Duration::from_secs(2)) else {
        return;
    };
    if server.child.is_none() {
        return;
    }
    if tokens_per_second >= SLOW_WRITING_TOKENS_PER_SECOND {
        server.was_fast = true;
        server.slow_answers = 0;
        forget_timeout();
        return;
    }
    server.slow_answers += 1;
    if restart_for_speed(tokens_per_second, server.was_fast) {
        warn!(
            tokens_per_second,
            "the local model has become slow; restarting it"
        );
        stop_locked(&mut server);
    } else {
        info!(
            tokens_per_second,
            "the local model writes slowly on this PC"
        );
    }
}

/// Slow answers in a row after which a long cleanup stops: at under 1 token a
/// second a 48 minute transcript would take hours, holding the NPU and
/// memory all that time. What is left is retried later.
const SLOW_ANSWERS_TO_GIVE_UP: u32 = 2;

/// Whether the built-in model has kept writing slowly (a long cleanup then
/// stops instead of grinding on).
pub fn keeps_writing_slowly() -> bool {
    TIMED_OUT.load(Ordering::Relaxed)
        || SERVER
            .try_lock_for(Duration::from_secs(2))
            .is_some_and(|server| server.slow_answers >= SLOW_ANSWERS_TO_GIVE_UP)
}

/// Set when the built-in model let a cleanup request time out. Kept apart
/// from the server state, which stopping the server (as a timeout does)
/// clears: a slice of a few hundred characters that takes over two minutes
/// means the model writes at about one token a second, so the rest of the
/// transcript would take hours.
static TIMED_OUT: AtomicBool = AtomicBool::new(false);

/// The built-in model did not answer a request in time.
pub fn note_timeout() {
    TIMED_OUT.store(true, Ordering::Relaxed);
}

/// A new cleanup starts: give the model another chance.
pub fn forget_timeout() {
    TIMED_OUT.store(false, Ordering::Relaxed);
}

/// Stop the server (called when OpenWritr exits or the model is removed).
pub fn shutdown() {
    // A model load can hold the lock for minutes; do not make quitting wait
    // for it (Windows ends the server with the app anyway).
    if let Some(mut server) = SERVER.try_lock_for(Duration::from_secs(2)) {
        stop_locked(&mut server);
    }
}

/// Tie the child's life to ours: when OpenWritr exits for any reason, even a
/// crash, Windows ends the server too, so the NPU is never left occupied.
fn kill_with_parent(child: &Child) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    // The job handle is intentionally never closed: it lives as long as the
    // process, and closing it is what kills the children.
    static JOB: Lazy<usize> = Lazy::new(|| unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return 0;
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let set = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        if set == 0 {
            return 0;
        }
        job as usize
    });

    let job = *JOB;
    if job == 0 {
        warn!("could not create a job object; the server will not stop with the app");
        return;
    }
    let assigned =
        unsafe { AssignProcessToJobObject(job as HANDLE, child.as_raw_handle() as HANDLE) };
    if assigned == 0 {
        warn!("could not attach the local model server to the app's job");
    }
}

// ---------------------------------------------------------------------------
// Test connection (used by Settings)
// ---------------------------------------------------------------------------

/// Models a server reports at `GET <base>/models` (OpenAI list format).
pub fn discover_models(base_url: &str) -> Result<Vec<String>> {
    let scope = EndpointScope::parse(base_url).map_err(|error| anyhow!(error.to_string()))?;
    if !scope.is_loopback() {
        bail!("only servers on this PC can be queried here");
    }
    ensure_running(&scope, None).map_err(|error| anyhow!(error.to_string()))?;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let value: serde_json::Value = client
        .get(scope.join("/models"))
        .send()?
        .error_for_status()?
        .json()?;
    Ok(parse_model_ids(&value))
}

/// What a test request measured.
#[derive(Clone, Copy, Debug, Default)]
pub struct TestResult {
    pub elapsed: Duration,
    /// Reading speed, tokens per second, as reported by the server.
    pub prompt_per_second: Option<f64>,
    /// Writing speed, tokens per second, as reported by the server.
    pub predicted_per_second: Option<f64>,
}

impl TestResult {
    /// One line for Settings.
    pub fn summary(&self) -> String {
        let mut text = format!("Works: answered in {:.1} s", self.elapsed.as_secs_f32());
        if let Some(rate) = self.prompt_per_second {
            text.push_str(&format!(" · reads {rate:.0} tok/s"));
        }
        if let Some(rate) = self.predicted_per_second {
            text.push_str(&format!(" · writes {rate:.1} tok/s"));
        }
        text.push('.');
        text
    }
}

/// Send one short chat request to a server on this PC and measure it. The
/// first request to the built-in model includes loading it onto the NPU, so
/// the timeout is generous; run the test twice to see the steady speed.
pub fn test_chat(base_url: &str, model: &str) -> Result<TestResult> {
    let scope = EndpointScope::parse(base_url).map_err(|error| anyhow!(error.to_string()))?;
    if !scope.is_loopback() {
        bail!("only servers on this PC can be tested here");
    }
    if model.trim().is_empty() {
        bail!("enter a model ID first");
    }
    ensure_running(&scope, Some(model)).map_err(|error| anyhow!(error.to_string()))?;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(240))
        .build()?;
    let mut body = serde_json::json!({
        "model": model.trim(),
        "messages": [{
            "role": "user",
            "content": "Write two short sentences about a quiet morning."
        }],
        "max_tokens": 48,
        "temperature": 0.0,
    });
    if is_managed(&scope) {
        add_request_options(&mut body, model);
    }
    let started = Instant::now();
    let response = client
        .post(scope.join("/chat/completions"))
        .json(&body)
        .send()
        .context("the server did not answer")?;
    if !response.status().is_success() {
        bail!("the server answered with {}", response.status());
    }
    let value: serde_json::Value = response.json().context("the answer was not valid JSON")?;
    let has_text = value
        .pointer("/choices/0/message/content")
        .and_then(|content| content.as_str())
        .is_some();
    if !has_text {
        bail!("the answer had no text in the OpenAI chat format");
    }
    Ok(TestResult {
        elapsed: started.elapsed(),
        prompt_per_second: value
            .pointer("/timings/prompt_per_second")
            .and_then(|rate| rate.as_f64())
            .filter(|rate| *rate > 0.0),
        predicted_per_second: value
            .pointer("/timings/predicted_per_second")
            .and_then(|rate| rate.as_f64())
            .filter(|rate| *rate > 0.0),
    })
}

fn parse_model_ids(value: &serde_json::Value) -> Vec<String> {
    let mut ids: Vec<String> = value
        .get("data")
        .and_then(|data| data.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("id").and_then(|id| id.as_str()))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    ids.sort();
    ids.dedup();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(raw: &str) -> EndpointScope {
        EndpointScope::parse(raw).unwrap()
    }

    #[test]
    fn each_choice_is_asked_to_run_where_it_should() {
        let mut body = serde_json::json!({"model": "x"});
        add_request_options(&mut body, "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_0@npu");
        assert_eq!(body["model"], QWEN3_4B_GGUF);
        assert_eq!(body["compute"], "npu");
        assert_eq!(body["ngl"], -1);
        assert_eq!(body["enable_think"], false);
        let mut body = serde_json::json!({"model": "x"});
        add_request_options(&mut body, "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_0@hybrid");
        assert_eq!(body["model"], QWEN3_4B_GGUF);
        assert_eq!(body["compute"], "hybrid");
        // A setting from before the list was cut down uses the first choice.
        let mut old = serde_json::json!({"model": "x"});
        add_request_options(&mut old, "ai-hub-models/Qwen3-4B-Instruct-2507");
        assert_eq!(old["model"], QWEN3_4B_GGUF);
        assert_eq!(old["compute"], "hybrid");
        assert_eq!(runtime("something-else"), Runtime::Hybrid);
        // Every choice is unique and shares one file.
        let mut ids: Vec<&str> = BUILT_IN_MODELS.iter().map(|model| model.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), BUILT_IN_MODELS.len());
        assert!(BUILT_IN_MODELS
            .iter()
            .all(|model| model.name == QWEN3_4B_GGUF));
    }

    #[test]
    fn downloads_never_wait_for_an_answer() {
        // A precision and a model type, or GenieX would ask for them.
        assert!(QWEN3_4B_GGUF.ends_with(":Q4_0"));
        for model in BUILT_IN_MODELS {
            assert_eq!(
                pull_arguments(model.id),
                ["pull", QWEN3_4B_GGUF, "--model-type", "llm"]
            );
        }
    }

    #[test]
    fn built_in_models_are_found_by_name_and_listed_apart() {
        assert_eq!(
            built_in(" unsloth/qwen3-4b-instruct-2507-gguf:q4_0@HYBRID ")
                .map(|model| model.runtime),
            Some(Runtime::Hybrid)
        );
        assert_eq!(built_in("ai-hub-models/Qwen3-4B-Instruct-2507"), None);
        let listing = r#"[{"name":"unsloth/Qwen3-4B-Instruct-2507-GGUF","precisions":["Q4_0"]}]"#;
        assert!(lists_model(listing, geniex_name(DEFAULT_MODEL)));
        assert!(!lists_model(
            listing,
            "ai-hub-models/Qwen3-4B-Instruct-2507"
        ));
        assert!(!lists_model(
            listing,
            "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_0"
        ));
    }

    #[test]
    fn only_the_private_loopback_endpoint_is_managed() {
        assert!(is_managed(&scope(MANAGED_BASE_URL)));
        assert!(is_managed(&scope("http://localhost:18651/v1/")));
        assert!(!is_managed(&scope("http://127.0.0.1:18181/v1")));
        assert!(!is_managed(&scope("http://127.0.0.1:11434/v1")));
        assert!(!is_managed(&scope("https://127.0.0.1:18651/v1")));
        assert!(!is_managed(&scope("http://192.168.1.5:18651/v1")));
        assert!(!is_managed(&scope("http://127.0.0.1:18651")));
    }

    #[test]
    fn platform_support_needs_arm64() {
        assert!(platform_support_for(true, None).is_ok());
        assert!(platform_support_for(false, Some("ARM64")).is_ok());
        assert!(platform_support_for(false, Some("arm64")).is_ok());
        assert!(platform_support_for(false, None).is_err());
        assert!(platform_support_for(false, Some("AMD64")).is_err());
    }

    #[test]
    fn the_genie_uninstaller_is_found_next_to_the_tool() {
        let temp = tempfile::tempdir().unwrap();
        let tool = temp.path().join("GenieX CLI");
        std::fs::create_dir_all(&tool).unwrap();
        let cli = tool.join("geniex.exe");
        std::fs::write(&cli, b"").unwrap();
        assert_eq!(find_uninstaller(&cli), None);
        std::fs::write(tool.join("unins000.exe"), b"").unwrap();
        std::fs::write(tool.join("unins000.dat"), b"").unwrap();
        assert_eq!(find_uninstaller(&cli), Some(tool.join("unins000.exe")));
    }

    #[test]
    fn cli_search_order_is_install_dir_then_program_files_then_path() {
        let path = std::env::join_paths(["tools", "bin"]).unwrap();
        let found = cli_candidates(
            Some(Path::new("local")),
            Some(Path::new("programs")),
            Some(&path),
        );
        assert_eq!(
            found,
            [
                Path::new("local").join("GenieX CLI").join("geniex.exe"),
                Path::new("programs").join("GenieX CLI").join("geniex.exe"),
                Path::new("tools").join("geniex.exe"),
                Path::new("bin").join("geniex.exe"),
            ]
        );
        assert!(cli_candidates(None, None, None).is_empty());
    }

    #[test]
    fn model_listing_matches_the_canonical_name_geniex_stores() {
        let listing =
            r#"[{"name":"qualcomm/Qwen3-4B-Instruct-2507","size":123,"precisions":["N/A"]}]"#;
        assert!(lists_model(listing, "ai-hub-models/Qwen3-4B-Instruct-2507"));
        assert!(lists_model(listing, "AI-HUB-MODELS/qwen3-4b-instruct-2507"));
        assert!(lists_model(listing, "qualcomm/Qwen3-4B-Instruct-2507:int4"));
        assert!(!lists_model(listing, "ai-hub-models/Other"));
        assert!(!lists_model(listing, "ai-hub-models/Qwen3-4B"));
        assert!(!lists_model(listing, "  "));
        assert!(!lists_model("[]", "ai-hub-models/Qwen3-4B-Instruct-2507"));
        // Not JSON: plain search for the bare name.
        assert!(lists_model(
            "qualcomm/Qwen3-4B-Instruct-2507  2.3 GiB",
            "ai-hub-models/Qwen3-4B-Instruct-2507"
        ));
    }

    #[test]
    fn download_bars_are_read_without_trusting_their_total() {
        // GenieX's bar at 100% while the download is still going.
        let line = "downloading 100% |██████████| (1.2 GB/310 MB, 25 MB/s) [48s:0s]";
        let bar = parse_pull_progress(line).unwrap();
        assert_eq!(bar.downloaded, 1.2e9);
        assert_eq!(bar.rate, Some(25e6));
        let (fraction, text) = describe_pull(&bar, Some(2_400_000_000));
        assert_eq!(fraction, Some(0.5));
        assert_eq!(text, "50% · 1.2 GB of 2.4 GB · 25.0 MB/s · about 48 s left");
        // Same unit for both: written once.
        let bar =
            parse_pull_progress("downloading  45% |████   | (1.1/2.4 GB, 9.5 MB/s) [1m0s:2m0s]")
                .unwrap();
        assert_eq!(bar.downloaded, 1.1e9);
        assert_eq!(bar.total, Some(2.4e9));
        // Unknown size: GenieX's total only when it is larger than what came.
        let (fraction, text) = describe_pull(&bar, None);
        assert!((fraction.unwrap() - 0.458).abs() < 0.01);
        assert_eq!(text, "46% · 1.1 GB of 2.4 GB · 9.5 MB/s · about 2 min left");
        // Never 100% before GenieX finishes.
        let (fraction, _) = describe_pull(&bar, Some(1_000_000_000));
        assert_eq!(fraction, Some(0.99));
        assert_eq!(parse_pull_progress("Detected model type: llm"), None);
        assert_eq!(parse_pull_progress("fetching available precisions"), None);
    }

    #[test]
    fn only_downloads_that_can_recover_are_retried() {
        assert!(worth_retrying(&anyhow!(
            "geniex pull failed (exit code: 1): connection reset"
        )));
        assert!(!worth_retrying(&anyhow!("cancelled")));
        assert!(!worth_retrying(&anyhow!(
            "geniex pull failed: 404 Not Found"
        )));
        assert!(!worth_retrying(&anyhow!(
            "geniex pull failed: no space left on device"
        )));
    }

    #[test]
    fn ansi_escapes_are_removed() {
        assert_eq!(strip_ansi("\u{1b}[32m✔  done\u{1b}[0m"), "✔  done");
        assert_eq!(strip_ansi("plain"), "plain");
    }

    #[test]
    fn progress_lines_split_on_carriage_returns() {
        let (tx, rx) = mpsc::channel();
        let data: Box<dyn Read + Send> =
            Box::new(std::io::Cursor::new(b"10%\r20%\r\n\nfinished".to_vec()));
        split_progress_lines(data, &tx);
        drop(tx);
        let lines: Vec<String> = rx.iter().collect();
        assert_eq!(lines, ["10%", "20%", "finished"]);
    }

    #[test]
    fn model_ids_are_read_sorted_and_deduplicated() {
        let value = serde_json::json!({
            "object": "list",
            "data": [{"id": "b"}, {"id": "a"}, {"id": "b"}, {"nope": 1}]
        });
        assert_eq!(parse_model_ids(&value), ["a", "b"]);
        assert!(parse_model_ids(&serde_json::json!({})).is_empty());
    }

    #[test]
    fn test_summary_lists_the_speeds_the_server_reported() {
        let full = TestResult {
            elapsed: Duration::from_millis(2500),
            prompt_per_second: Some(310.4),
            predicted_per_second: Some(14.26),
        };
        assert_eq!(
            full.summary(),
            "Works: answered in 2.5 s · reads 310 tok/s · writes 14.3 tok/s."
        );
        let bare = TestResult {
            elapsed: Duration::from_secs(1),
            ..TestResult::default()
        };
        assert_eq!(bare.summary(), "Works: answered in 1.0 s.");
    }

    #[test]
    fn server_output_is_logged_at_a_level_that_matches_it() {
        // Lines as GenieX printed them in a real run.
        let step = r"Oct 17 19:26:02.120 INF go\ml.go:233 [ML] [QNN]  <I> QnnGraph_execute started. graph = 0x4";
        let done = r"Oct 17 19:26:02.125 INF go\ml.go:233 [ML] [QNN]  <I> Graph token_ar1_cl512_1_of_4 execution finished with result 0";
        let error = r"Oct 17 19:25:34.739 ERR go\ml.go:237 [ML] [QNN]  <E> Failed to map buffer of size 538968064";
        let summary = r"Oct 17 19:26:02.625 INF handler\chat_response.go:185 Generation complete prompt_tokens=332 decode_tok_s=18.1";
        assert_eq!(server_line_level(step), LineLevel::Debug);
        assert_eq!(server_line_level(done), LineLevel::Debug);
        assert_eq!(server_line_level(error), LineLevel::Warn);
        assert_eq!(server_line_level(summary), LineLevel::Info);
        assert_eq!(
            server_line_level("Local hosting on http://127.0.0.1:18651/"),
            LineLevel::Info
        );
    }

    #[test]
    fn only_a_server_that_slowed_down_is_restarted() {
        assert!(restart_for_speed(0.9, true), "fast before, slow now");
        assert!(!restart_for_speed(0.9, false), "always slow here");
        assert!(!restart_for_speed(18.2, true));
        assert!(!restart_for_speed(SLOW_WRITING_TOKENS_PER_SECOND, true));
    }

    #[test]
    fn releasing_without_a_started_server_does_nothing() {
        release("test");
        assert!(SERVER.lock().child.is_none());
    }

    #[test]
    fn hex_is_lowercase_and_padded() {
        assert_eq!(hex(&[0, 15, 255]), "000fff");
    }

    #[test]
    fn unmanaged_endpoints_never_start_anything() {
        assert!(ensure_running(&scope("http://127.0.0.1:11434/v1"), Some("m")).is_ok());
        assert!(ensure_running(&scope("https://api.openai.com/v1"), None).is_ok());
    }
}
