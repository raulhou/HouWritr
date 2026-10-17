//! Helpers to make `ort` 2.0-rc.12 errors play nicely with `anyhow::Error`.
//!
//! `ort::Error<R>` carries a "recoverable" payload (e.g. SessionBuilder).
//! Those payloads contain raw `NonNull<...>` pointers that are not Send/Sync,
//! so the auto-conversion via `?` into `anyhow::Error` (which requires
//! Send + Sync) fails. We provide a small extension trait that drops the
//! payload and converts to a simple string-based anyhow error.

use anyhow::{anyhow, Context, Result};
use ort::session::{
    builder::{AutoDevicePolicy, GraphOptimizationLevel},
    Session,
};
use std::path::Path;

pub trait OrtResultExt<T> {
    fn ortx(self) -> anyhow::Result<T>;
}

impl<T, R> OrtResultExt<T> for std::result::Result<T, ort::Error<R>> {
    fn ortx(self) -> anyhow::Result<T> {
        self.map_err(|e| anyhow!("ort: {e}"))
    }
}

/// CPU worker threads for ONNX sessions: all cores up to 8.
pub fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(4)
}

/// A session pinned to the CPU execution provider.
///
/// Automatic device selection is pinned to CPU so that later session builds
/// stay CPU-only even after the QNN provider has been registered
/// process-wide.
pub fn build_cpu_session(path: &Path) -> Result<Session> {
    use ort::ep::CPU;
    Session::builder()
        .ortx()?
        .with_auto_device(AutoDevicePolicy::PreferCPU)
        .ortx()?
        .with_no_environment_execution_providers()
        .ortx()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .ortx()?
        .with_intra_threads(num_threads())
        .ortx()?
        .with_execution_providers([CPU::default().build()])
        .ortx()?
        .commit_from_file(path)
        .ortx()
        .with_context(|| format!("load {}", path.display()))
}

/// A CPU session that cannot end up on the NPU.
///
/// Once the QNN provider is registered (the Whisper NPU engine does that),
/// the plain [`build_cpu_session`] fails with "Could not determine default
/// backend path for device", and a session with no provider named at all was
/// seen being handed to the NPU provider ("Dynamic shape is not supported
/// yet"). Naming the CPU device explicitly avoids both. Each way of asking is
/// tried in turn and, if all fail, every error is reported.
pub fn build_plain_cpu_session(path: &Path) -> Result<Session> {
    let mut failures = Vec::new();
    match build_on_cpu_device(path) {
        Ok(session) => return Ok(session),
        Err(error) => failures.push(format!("explicit CPU device: {error:#}")),
    }
    match build_without_providers(path) {
        Ok(session) => return Ok(session),
        Err(error) => failures.push(format!("default providers: {error:#}")),
    }
    Err(anyhow!("load {}: {}", path.display(), failures.join(" | ")))
}

fn build_on_cpu_device(path: &Path) -> Result<Session> {
    use ort::memory::DeviceType;
    let environment = ort::environment::current().ortx()?;
    let devices: Vec<_> = environment
        .devices()
        .filter(|device| {
            device.ty() == DeviceType::CPU && matches!(device.ep(), Ok("CPUExecutionProvider"))
        })
        .collect();
    if devices.is_empty() {
        return Err(anyhow!("ONNX Runtime lists no CPU device"));
    }
    Session::builder()
        .ortx()?
        .with_no_environment_execution_providers()
        .ortx()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .ortx()?
        .with_intra_threads(num_threads())
        .ortx()?
        .with_devices(devices, None)
        .ortx()?
        .commit_from_file(path)
        .ortx()
}

fn build_without_providers(path: &Path) -> Result<Session> {
    use ort::ep::CPU;
    Session::builder()
        .ortx()?
        .with_no_environment_execution_providers()
        .ortx()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .ortx()?
        .with_intra_threads(num_threads())
        .ortx()?
        .with_execution_providers([CPU::default().build()])
        .ortx()?
        .commit_from_file(path)
        .ortx()
}
