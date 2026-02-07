# Human Pose Burn

[English Doc](README.md)

使用 Copper 任务图 + Burn 推理后端的 YOLOv8 Pose 演示。GStreamer
采集摄像头图像，Burn 执行姿态估计，Rerun 做可视化。

## 功能概览

- GStreamer 采集摄像头帧，输出到 Copper
- `GstToCuImage` 转换为 `CuImage`
- `YoloPoseBurn` 使用 Burn 后端推理 YOLOv8 Pose
- `RerunPoseViz` 在 Rerun Viewer 中显示图像、关键点、骨架与框

## 依赖项安装

### Rust

建议使用稳定版 Rust 工具链。

### GStreamer

项目使用 `cu-gstreamer` 进行视频采集，请先安装 GStreamer 及常用插件。

macOS (Homebrew):

```bash
brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav
```

Linux (Debian/Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav
```

### Rerun Viewer

可选，但强烈建议安装以可视化结果。

macOS (Homebrew):

```bash
brew install rerun-io/tap/rerun
```

Linux: 参考 https://www.rerun.io/docs/getting-started/installing-viewer

## 运行

默认启用 `wgpu` 后端：

```bash
cargo run --release
```

如果需要切换后端：

```bash
# CPU 后端
cargo run --release --no-default-features --features cpu

# CUDA 后端 (需要 CUDA 环境)
cargo run --release --no-default-features --features cuda
```

运行时会自动从 HuggingFace 下载 YOLOv8 pose 权重（模型仓库
`lmz/candle-yolo-v8`），并缓存到本地。

## 配置摄像头参数 (RON)

配置文件：

- macOS: [copperconfig.mac.ron](copperconfig.mac.ron)
- Linux: [copperconfig.linux.ron](copperconfig.linux.ron)

与摄像头相关的配置主要在以下字段：

- `tasks.camera.config.pipeline`：GStreamer pipeline
- `tasks.camera.config.caps`：appsink 输出 caps
- `tasks.gst_to_image.config.width/height/fourcc`：必须与 pipeline
  中的输出保持一致

**关键要求**

- `pipeline` 里的 `video/x-raw, format=...` 要与 `caps` 和 `gst_to_image` 的
  `fourcc` 对齐。
- `width/height` 在 `pipeline`、`caps`、`gst_to_image` 中必须一致。
- `framerate` 必须是 GStreamer 接受的分数形式，比如 `30/1` 或 `30000/1001`。

## 如何确定 width/height/framerate

### macOS (avfvideosrc)

1. 查看可用设备和支持的 caps:

```bash
gst-device-monitor-1.0 Video
```

2. 验证某组参数是否可用:

```bash
gst-launch-1.0 avfvideosrc device-index=0 ! \
  video/x-raw,width=480,height=360,framerate=30000/1001 ! \
  videoconvert ! fakesink
```

### Linux (v4l2src)

1. 查看设备支持的分辨率与帧率:

```bash
v4l2-ctl --list-formats-ext -d /dev/video0
```

2. 验证参数:

```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! \
  video/x-raw,format=NV12,width=640,height=360,framerate=30/1 ! \
  videoconvert ! fakesink
```

## 修改示例

假设你需要把分辨率改成 640x480、帧率 30fps，并输出 NV12：

- 更新 `pipeline`:

```text
... ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! video/x-raw, format=NV12 ! ...
```

- 更新 `caps`:

```text
video/x-raw, format=NV12, width=640, height=480
```

- 更新 `gst_to_image`:

```text
width: 640
height: 480
fourcc: "NV12"
```

## Copper-rs 与 Burn 说明

- Copper 负责任务图调度与消息传递，GStreamer 采集作为 `CuDefaultGStreamer`
  任务输入。
- `GstToCuImage` 把 `CuGstBuffer` 转成
  `CuImage<Vec<u8>>`，为推理和可视化提供统一的数据格式。
- `YoloPoseBurn` 使用 Burn 后端加载 safetensors 权重并执行推理，后端由 Cargo
  features 选择 (`wgpu`/`cpu`/`cuda`)。
- `RerunPoseViz` 使用 Rerun SDK 把图像、骨架和关键点实时显示。

## 常见问题

- 如果看不到画面，优先检查配置的 pipeline 部份，确认 `width/height/format`
  是否和摄像头支持的参数一致。
