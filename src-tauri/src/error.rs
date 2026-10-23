use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum SplatError {
    /// 某个组件缺失（媒体 / 重建 / 训练）。
    #[error("找不到本地处理引擎：{0}")]
    ComponentMissing(String),
    /// 某个组件存在但起不来。
    #[error("本地处理引擎无法启动：{component}（{detail}）")]
    EngineStart { component: String, detail: String },
    #[error("视频无效：{0}")]
    InvalidVideo(String),
    #[error("不支持的文件路径：{0}")]
    InvalidPath(PathBuf),
    #[error("任务已取消")]
    Cancelled,
    #[error("外部进程执行失败：{0}")]
    Process(String),
    #[error("Brush 显存不足：{0}")]
    BrushOutOfMemory(String),
    #[error("Splat 训练期间显卡设备连接中断。请接通电源、关闭占用显卡的软件，并在 Windows 图形设置中将 brush_app.exe 设为高性能后重试。\n{0}")]
    BrushDeviceLost(String),
    /// 后端版本/形态不支持安全接入。
    #[error("当前引擎版本不支持安全接入：{0}")]
    UnsupportedBackend(String),
    #[error("文件读写失败：{0}")]
    Io(#[from] std::io::Error),
    #[error("无法解析引擎输出：{0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, SplatError>;

impl serde::Serialize for SplatError {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}
