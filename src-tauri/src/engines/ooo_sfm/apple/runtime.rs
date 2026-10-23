//! ooo-sfm 的进程内运行时（"伪进程"封装）。
//!
//! 自研后端在本进程内计算，但上层（`pipeline/runner.rs` 的 observer、`work/logs/*.log`
//! 日志文件、取消、错误文案）期待的是"一个子进程"的行为。本模块把二者缝合起来，
//! 复刻 `ProcessManager::run`（`src/process/mod.rs`）的**对外可观察行为**：
//!
//! | 行为 | 子进程版 | 本模块 |
//! |---|---|---|
//! | `Started { process_id }` | 子进程真实 PID | 本进程 PID（承载该阶段的进程） |
//! | `LogOpened { path, offset }` | `manager.lifecycle` | 同左（`manager.lifecycle_observer`） |
//! | `Heartbeat { elapsed_ms }` | 1 s 定时器 | 1 s 定时器 |
//! | `Exited { executable, exit_code }` | 子进程退出 | 任务结束（成功 `Some(0)`／失败 `Some(1)`） |
//! | `Line { stream, line }` | stdout/stderr 泵 | `ProgressSink::log` |
//! | `StageProgress { .. }` | —（子进程版不存在） | `ProgressSink::stage`（结构化进度） |
//! | 取消 | 杀进程组 | 协作式：`CancellationToken` + 分块点 `check_cancelled` |
//! | 错误 | `ProcessOutput::failure_detail()` 尾部 | 同一函数，喂入本次任务的日志尾部 |
//! | panic | 进程崩溃（外层可见） | `catch_unwind` 兜住 → `SplatError::Process` |
//!
//! 已知且刻意的差异：
//! - 取消不是"立即生效"，而是任务在下一个分块点退出（进程内无法 kill 线程）；
//!   因此所有计算循环**必须**在分块点调用 [`ProgressSink::check_cancelled`]。
//! - `Resources`（按 PID 采 GPU 指标）不产生：进程内没有独立子进程可采样
//!   （macOS 本就不支持该采样）。

use std::{
    io::Write,
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant},
};

use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

use crate::{
    error::{Result, SplatError},
    process::{ProcessManager, ProcessObserver, ProcessOutput, ProcessStream, ProcessUpdate},
};

/// 失败诊断保留的日志尾部长度，与 `ProcessOutput::failure_detail` 的上限一致。
const TAIL_CHARS: usize = 4_096;

/// 任务上报进度 / 日志 / 取消的句柄；可从计算线程使用（内部均为 `Arc`）。
#[derive(Clone)]
#[allow(dead_code)] // M1：契约先定，调用点在 M3–M6 逐个落地（届时移除本抑制）
pub struct ProgressSink {
    observer: Option<ProcessObserver>,
    cancel: CancellationToken,
    log: Arc<Mutex<std::fs::File>>,
    tail: Arc<Mutex<String>>,
}

#[allow(dead_code)] // M1：同上
impl ProgressSink {
    /// 结构化进度 → `ProcessUpdate::StageProgress`，同时写入任务日志。
    pub fn stage(
        &self,
        phase: &'static str,
        current: u64,
        total: Option<u64>,
        message: impl Into<String>,
    ) {
        let message = message.into();
        self.write_log(&message);
        if let Some(observer) = &self.observer {
            observer(ProcessUpdate::StageProgress {
                phase,
                current,
                total,
                message,
            });
        }
    }

    /// 一行人类可读日志 → 日志文件 + `ProcessUpdate::Line`。
    pub fn log(&self, line: impl AsRef<str>) {
        let line = line.as_ref();
        self.write_log(line);
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return;
        }
        if let Some(observer) = &self.observer {
            observer(ProcessUpdate::Line {
                stream: ProcessStream::Stdout,
                line: trimmed.to_string(),
            });
        }
    }

    /// 取消是协作式的：所有长循环必须在分块点调用它。
    pub fn check_cancelled(&self) -> Result<()> {
        if self.cancel.is_cancelled() {
            Err(SplatError::Cancelled)
        } else {
            Ok(())
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    fn write_log(&self, line: &str) {
        let _ = writeln!(lock(&self.log), "{line}");
        let mut tail = lock(&self.tail);
        tail.push_str(line);
        tail.push('\n');
        let excess = tail.chars().count().saturating_sub(TAIL_CHARS);
        if excess > 0 {
            *tail = tail.chars().skip(excess).collect();
        }
    }

    /// 本次任务日志的有界尾部（失败诊断用）。
    fn tail(&self) -> String {
        lock(&self.tail).clone()
    }
}

/// 测试用：无 observer 的 sink（日志写在临时文件里；`cancel` 由调用方决定）。
#[cfg(test)]
pub fn test_sink(cancel: CancellationToken) -> Result<ProgressSink> {
    let path = std::env::temp_dir().join(format!(
        "ooo-sfm-test-{}-{:?}.log",
        std::process::id(),
        std::thread::current().id()
    ));
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    Ok(ProgressSink {
        observer: None,
        cancel,
        log: Arc::new(Mutex::new(log)),
        tail: Arc::new(Mutex::new(String::new())),
    })
}

/// 占位阶段统一走这里：把失败原因写进任务日志，再返回错误
/// （与子进程版"stderr 进日志 + 错误里带日志尾部"一致）。
pub fn fail(sink: &ProgressSink, error: SplatError) -> Result<()> {
    sink.log(error.to_string());
    Err(error)
}

/// 任务结局（决定日志尾部、`Exited.exit_code` 与返回的错误）。
enum JobOutcome {
    Completed,
    Cancelled,
    Failed(String),
}

/// 在**本进程内**执行一个 SfM 阶段，并复刻子进程版的全套可观察行为。
///
/// - `executable`：上层传入的引擎路径；仅用于日志头部与 `Exited` 事件的引擎归类
///   （自研后端不会启动它）。
/// - `description`：写入日志"参数区"的可读描述（取代 CLI 参数列表）。
/// - `log_path`：与子进程版同一个 `work/logs/*.log`。
pub async fn run_job<F>(
    manager: &ProcessManager,
    executable: &Path,
    description: &[String],
    log_path: PathBuf,
    observer: Option<ProcessObserver>,
    job: F,
) -> Result<()>
where
    F: FnOnce(ProgressSink) -> Result<()> + Send + 'static,
{
    let cancel = manager.child_token();
    let lifecycle = manager.lifecycle_observer();
    let started = Instant::now();

    if let Some(observer) = &observer {
        // 进程内没有子进程；上报承载该阶段的本进程 PID。
        observer(ProcessUpdate::Started {
            process_id: std::process::id(),
        });
    }
    // 守卫：任何提前返回都会停掉心跳。
    let _heartbeat = Heartbeat::spawn(&observer, started);

    // 日志准备顺序与 `ProcessManager::run` 一致：建目录 → 打开 → LogOpened → 写头部。
    if let Some(parent) = log_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .await?;
    if let Some(lifecycle) = &lifecycle {
        lifecycle(ProcessUpdate::LogOpened {
            path: log_path.clone(),
            offset: file.metadata().await?.len(),
        });
    }
    file.write_all(
        format!(
            "engine: ooo-sfm (in-process)\nexecutable: {}\narguments:\n  {}\n",
            executable.display(),
            if description.is_empty() {
                "-".to_string()
            } else {
                description.join("\n  ")
            }
        )
        .as_bytes(),
    )
    .await?;
    let log = Arc::new(Mutex::new(file.into_std().await));

    let sink = ProgressSink {
        observer: observer.clone(),
        cancel: cancel.clone(),
        log: Arc::clone(&log),
        tail: Arc::new(Mutex::new(String::new())),
    };
    let job_sink = sink.clone();

    let result = tokio::task::spawn_blocking(move || {
        let outcome = match panic::catch_unwind(AssertUnwindSafe(|| job(job_sink))) {
            Ok(Ok(())) => JobOutcome::Completed,
            Ok(Err(SplatError::Cancelled)) => JobOutcome::Cancelled,
            Ok(Err(error)) => JobOutcome::Failed(error.to_string()),
            Err(payload) => JobOutcome::Failed(format!(
                "ooo-sfm 内部错误（panic）：{}",
                // 注意：必须显式 `.as_ref()`。`&Box<dyn Any + Send>` 会走 unsize
                // （Box 自身也是 `Any + Send`）而不是 deref，downcast 永远失败。
                panic_message(payload.as_ref())
            )),
        };
        // 取消具有最终话语权（与子进程版"取消即 Cancelled"一致）。
        let outcome = if cancel.is_cancelled() {
            JobOutcome::Cancelled
        } else {
            outcome
        };
        finish_log(&mut lock(&log), &outcome, started);
        outcome
    })
    .await;

    let outcome = match result {
        Ok(outcome) => outcome,
        Err(error) => JobOutcome::Failed(format!("ooo-sfm 工作线程不可用：{error}")),
    };

    let exit_code = match outcome {
        JobOutcome::Completed => Some(0),
        JobOutcome::Cancelled => None,
        JobOutcome::Failed(_) => Some(1),
    };
    // 与子进程版一致：取消路径不产生 `Exited` 事件。
    if !matches!(outcome, JobOutcome::Cancelled) {
        if let Some(lifecycle) = &lifecycle {
            lifecycle(ProcessUpdate::Exited {
                executable: executable.to_path_buf(),
                exit_code,
            });
        }
    }

    match outcome {
        JobOutcome::Completed => Ok(()),
        JobOutcome::Cancelled => Err(SplatError::Cancelled),
        JobOutcome::Failed(reason) => {
            let detail = ProcessOutput {
                success: false,
                exit_code,
                stdout: String::new(),
                stderr: sink.tail(),
            }
            .failure_detail();
            Err(SplatError::Process(format!(
                "ooo-sfm 阶段失败（{reason}）{}",
                if detail.is_empty() {
                    String::new()
                } else {
                    format!("\n{detail}")
                }
            )))
        }
    }
}

/// 1 s 心跳；drop 即停止（复刻 `ProcessManager::run` 的 heartbeat 定时器）。
struct Heartbeat(Option<tokio::task::JoinHandle<()>>);

impl Heartbeat {
    fn spawn(observer: &Option<ProcessObserver>, started: Instant) -> Self {
        let Some(observer) = observer.clone() else {
            return Self(None);
        };
        Self(Some(tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
                observer(ProcessUpdate::Heartbeat {
                    elapsed_ms: started.elapsed().as_millis() as u64,
                });
            }
        })))
    }
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        if let Some(handle) = &self.0 {
            handle.abort();
        }
    }
}

fn finish_log(file: &mut std::fs::File, outcome: &JobOutcome, started: Instant) {
    let elapsed_ms = started.elapsed().as_millis();
    let message = match outcome {
        JobOutcome::Completed => format!("exit_code: Some(0)\nelapsed_ms: {elapsed_ms}\n\n"),
        JobOutcome::Cancelled => format!("cancelled after {elapsed_ms} ms\n\n"),
        JobOutcome::Failed(reason) => format!("error: {reason}\nelapsed_ms: {elapsed_ms}\n\n"),
    };
    let _ = file.write_all(message.as_bytes());
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "未知 panic".into()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// 采集 observer 事件，便于断言事件序列。
    #[derive(Default)]
    struct Recorder {
        updates: Mutex<Vec<String>>,
    }

    impl Recorder {
        fn observer(self: &Arc<Self>) -> ProcessObserver {
            let recorder = Arc::clone(self);
            Arc::new(move |update: ProcessUpdate| {
                let label = match update {
                    ProcessUpdate::Started { process_id } => format!("started:{process_id}"),
                    ProcessUpdate::LogOpened { offset, .. } => format!("log-opened:{offset}"),
                    ProcessUpdate::Exited { exit_code, .. } => format!("exited:{exit_code:?}"),
                    ProcessUpdate::Heartbeat { .. } => "heartbeat".into(),
                    ProcessUpdate::Resources(_) => "resources".into(),
                    ProcessUpdate::StageProgress {
                        phase,
                        current,
                        total,
                        message,
                    } => format!("stage:{phase}:{current}/{total:?}:{message}"),
                    ProcessUpdate::Line { line, .. } => format!("line:{line}"),
                };
                lock(&recorder.updates).push(label);
            })
        }

        fn all(&self) -> Vec<String> {
            lock(&self.updates).clone()
        }

        fn contains(&self, needle: &str) -> bool {
            self.all().iter().any(|item| item.contains(needle))
        }
    }

    fn manager_with_lifecycle(recorder: &Arc<Recorder>) -> ProcessManager {
        ProcessManager::new().with_lifecycle(recorder.observer())
    }

    #[tokio::test]
    async fn runs_in_process_and_mirrors_process_observability() {
        let directory = tempfile::tempdir().unwrap();
        let log_path = directory.path().join("logs/sfm.log");
        let recorder = Arc::new(Recorder::default());
        let manager = manager_with_lifecycle(&recorder);

        run_job(
            &manager,
            Path::new("/engines/colmap"),
            &["stage: features".to_string()],
            log_path.clone(),
            Some(recorder.observer()),
            |sink| {
                sink.log("提取特征 1/2");
                sink.stage("features", 2, Some(2), "已完成 2 张");
                Ok(())
            },
        )
        .await
        .unwrap();

        let events = recorder.all();
        assert!(recorder.contains(&format!("started:{}", std::process::id())));
        assert!(recorder.contains("log-opened:0"));
        assert!(recorder.contains("line:提取特征 1/2"));
        assert!(recorder.contains("stage:features:2/Some(2)"));
        assert!(recorder.contains("exited:Some(0)"));

        let log = tokio::fs::read_to_string(&log_path).await.unwrap();
        assert!(log.starts_with("engine: ooo-sfm (in-process)\n"));
        assert!(log.contains("executable: /engines/colmap\n"));
        assert!(log.contains("arguments:\n  stage: features\n"));
        assert!(log.contains("提取特征 1/2\n"));
        assert!(log.contains("已完成 2 张\n"));
        assert!(log.contains("exit_code: Some(0)\n"));
        assert!(log.contains("elapsed_ms: "));
        assert!(events.len() >= 5);
    }

    #[tokio::test]
    async fn appends_to_an_existing_log_and_reports_its_offset() {
        let directory = tempfile::tempdir().unwrap();
        let log_path = directory.path().join("sfm.log");
        tokio::fs::write(&log_path, "previous run\n").await.unwrap();
        let recorder = Arc::new(Recorder::default());
        let manager = manager_with_lifecycle(&recorder);

        run_job(
            &manager,
            Path::new("/engines/colmap"),
            &[],
            log_path.clone(),
            Some(recorder.observer()),
            |_| Ok(()),
        )
        .await
        .unwrap();

        assert!(recorder.contains("log-opened:13"));
        let log = tokio::fs::read_to_string(&log_path).await.unwrap();
        assert!(log.starts_with("previous run\nengine: ooo-sfm"));
    }

    #[tokio::test]
    async fn cooperative_cancellation_stops_the_job_and_marks_the_log() {
        let directory = tempfile::tempdir().unwrap();
        let log_path = directory.path().join("sfm.log");
        let recorder = Arc::new(Recorder::default());
        let manager = manager_with_lifecycle(&recorder);
        let cancelling = manager.clone();
        let started = Arc::new(AtomicBool::new(false));

        let job_started = Arc::clone(&started);
        let run = run_job(
            &manager,
            Path::new("/engines/colmap"),
            &[],
            log_path.clone(),
            Some(recorder.observer()),
            move |sink| {
                job_started.store(true, Ordering::SeqCst);
                loop {
                    sink.check_cancelled()?;
                    std::thread::sleep(Duration::from_millis(5));
                }
            },
        );

        let trigger = tokio::spawn(async move {
            while !started.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            cancelling.cancel();
        });
        let result = run.await;
        trigger.await.unwrap();

        assert!(matches!(result, Err(SplatError::Cancelled)));
        let log = tokio::fs::read_to_string(&log_path).await.unwrap();
        assert!(log.contains("cancelled after "));
        assert!(!log.contains("exit_code:"));
        assert!(!recorder.contains("exited:"));
    }

    #[tokio::test]
    async fn failure_and_panic_are_mapped_to_process_errors_with_a_bounded_tail() {
        let directory = tempfile::tempdir().unwrap();
        let recorder = Arc::new(Recorder::default());
        let manager = manager_with_lifecycle(&recorder);

        let failed = run_job(
            &manager,
            Path::new("/engines/colmap"),
            &[],
            directory.path().join("failed.log"),
            None,
            |sink| {
                sink.log("注册失败：图像无法三角化");
                Err(SplatError::Process("mapper 未注册任何图像".into()))
            },
        )
        .await
        .unwrap_err();
        let message = failed.to_string();
        assert!(message.contains("ooo-sfm 阶段失败"));
        assert!(message.contains("mapper 未注册任何图像"));
        assert!(message.contains("注册失败：图像无法三角化"));

        let panicked = run_job(
            &manager,
            Path::new("/engines/colmap"),
            &[],
            directory.path().join("panicked.log"),
            None,
            |_| panic!("三角化越界"),
        )
        .await
        .unwrap_err();
        assert!(
            panicked.to_string().contains("三角化越界"),
            "got: {panicked}"
        );
        assert!(panicked.to_string().contains("panic"));

        let failed_log = tokio::fs::read_to_string(directory.path().join("failed.log"))
            .await
            .unwrap();
        assert!(failed_log.contains("error: "));
        assert!(failed_log.contains("elapsed_ms: "));
    }
}
