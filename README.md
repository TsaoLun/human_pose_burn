# Human Pose Burn

[中文文档](README_zh.md)

YOLOv8 pose demo using a Copper task graph with a Burn inference backend.
GStreamer captures camera frames, Burn runs pose estimation, and Rerun
visualizes results.

## Overview

- Capture camera frames via GStreamer into Copper
- `GstToCuImage` converts to `CuImage`
- `YoloPoseBurn` runs YOLOv8 pose with a Burn backend
- `RerunPoseViz` renders image, keypoints, skeletons, and boxes in Rerun Viewer

## Dependencies

### Rust

Use a stable Rust toolchain.

### GStreamer

This project relies on `cu-gstreamer` for video capture. Install GStreamer and
common plugins.

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

Optional but highly recommended for visualization.

macOS (Homebrew):

```bash
brew install rerun-io/tap/rerun
```

Linux: see https://www.rerun.io/docs/getting-started/installing-viewer

## Run

Default backend is `wgpu`:

```bash
cargo run --release
```

Switch backends:

```bash
# CPU backend
cargo run --release --no-default-features --features cpu

# CUDA backend (requires a CUDA environment)
cargo run --release --no-default-features --features cuda
```

On first run, weights are downloaded from HuggingFace (repo
`lmz/candle-yolo-v8`) and cached locally.

## Camera Config (RON)

Config files:

- macOS: [copperconfig.mac.ron](copperconfig.mac.ron)
- Linux: [copperconfig.linux.ron](copperconfig.linux.ron)

Camera-related fields:

- `tasks.camera.config.pipeline`: GStreamer pipeline
- `tasks.camera.config.caps`: appsink output caps
- `tasks.gst_to_image.config.width/height/fourcc`: must match pipeline output

**Key rules**

- `video/x-raw, format=...` in `pipeline` must match `caps` and
  `gst_to_image.fourcc`.
- `width/height` must be consistent across `pipeline`, `caps`, and
  `gst_to_image`.
- `framerate` must be a GStreamer fraction, e.g. `30/1` or `30000/1001`.

## How to determine width/height/framerate

### macOS (avfvideosrc)

1. List devices and supported caps:

```bash
gst-device-monitor-1.0 Video
```

2. Validate a candidate setting:

```bash
gst-launch-1.0 avfvideosrc device-index=0 ! \
  video/x-raw,width=480,height=360,framerate=30000/1001 ! \
  videoconvert ! fakesink
```

### Linux (v4l2src)

1. List supported resolutions and frame rates:

```bash
v4l2-ctl --list-formats-ext -d /dev/video0
```

2. Validate a candidate setting:

```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! \
  video/x-raw,format=NV12,width=640,height=360,framerate=30/1 ! \
  videoconvert ! fakesink
```

## Example update

If you want 640x480 at 30fps with NV12 output:

- Update `pipeline`:

```text
... ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! video/x-raw, format=NV12 ! ...
```

- Update `caps`:

```text
video/x-raw, format=NV12, width=640, height=480
```

- Update `gst_to_image`:

```text
width: 640
height: 480
fourcc: "NV12"
```

## Copper-rs and Burn

- Copper handles task scheduling and message passing; GStreamer capture runs as
  a `CuDefaultGStreamer` task.
- `GstToCuImage` converts `CuGstBuffer` into `CuImage<Vec<u8>>` for inference
  and visualization.
- `YoloPoseBurn` loads safetensors weights and runs inference using the selected
  Burn backend (`wgpu`/`cpu`/`cuda`).
- `RerunPoseViz` uses the Rerun SDK to render images, skeletons, and keypoints
  in real time.

## Troubleshooting

- If you see no frames, check the pipeline configuration to ensure that `width`,
  `height`, and `format` match the supported parameters of your camera.

## Log Analysis

The application automatically logs structured data to `logs/human-pose.copper`.
Use the logreader to analyze logs:

### Extract text logs

```bash
cargo run --features logreader --bin human-pose-logreader -- \
  logs/human-pose.copper \
  extract-text-log \
  target/debug/cu29_log_index
```

### Extract CopperLists (JSON format)

```bash
cargo run --features logreader --bin human-pose-logreader -- \
  logs/human-pose.copper \
  extract-copperlists
```

### Analyze task latency (log statistics)

```bash
# macOS
cargo run --features logreader --bin human-pose-logreader -- \
  logs/human-pose.copper \
  log-stats --config copperconfig.mac.ron

# Linux
cargo run --features logreader --bin human-pose-logreader -- \
  logs/human-pose.copper \
  log-stats --config copperconfig.linux.ron
```

This generates `cu29_logstats.json` containing per-task latency statistics:

- Minimum, maximum, and mean latencies
- Standard deviation and jitter metrics
- End-to-end latency analysis

### Check log integrity

```bash
cargo run --features logreader --bin human-pose-logreader -- \
  logs/human-pose.copper \
  fsck
```
