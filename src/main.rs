mod image;
mod payloads;
mod tasks;
mod yolo;

use std::fs;
use std::path::{Path, PathBuf};

use cu29::prelude::*;
use cu29_helpers::basic_copper_setup;

const SLAB_SIZE: Option<usize> = Some(1024 * 1024 * 1024); // 1GB

#[cfg(target_os = "macos")]
#[copper_runtime(config = "copperconfig.mac.ron")]
struct YoloPoseDemoApplication {}

#[cfg(target_os = "linux")]
#[copper_runtime(config = "copperconfig.linux.ron")]
struct YoloPoseDemoApplication {}

fn main() {
    // Create temporary directory for Copper logs
    let logger_path = "logs/human-pose.copper";
    if let Some(parent) = Path::new(logger_path).parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).expect("Failed to create logs directory");
        }
    }

    // Initialize Copper context
    let copper_ctx = basic_copper_setup(&PathBuf::from(logger_path), SLAB_SIZE, true, None)
        .expect("Failed to set up Copper context");

    // Build the application from RON config
    let mut application = YoloPoseDemoApplicationBuilder::new()
        .with_context(&copper_ctx)
        .build()
        .expect("Failed to build application");

    println!("Starting YOLOv8 Pose Demo with Burn framework...");

    // Start all tasks
    application
        .start_all_tasks()
        .expect("Failed to start tasks");

    println!("Running... Press Ctrl+C to stop.");

    // Run the application
    if let Err(e) = application.run() {
        error!("Error during iteration: {}", e.to_string());
    }

    println!("Stopping tasks and flushing logs...");

    // Stop all tasks to ensure proper cleanup and log flushing
    application.stop_all_tasks().expect("Failed to stop tasks");

    println!("Application stopped successfully.");
}
