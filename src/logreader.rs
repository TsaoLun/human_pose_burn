mod payloads;

use crate::payloads::CuPoses;
use cu_gstreamer::CuGstBuffer;
use cu_sensor_payloads::CuImage;
use cu29::prelude::*;
use cu29_export::run_cli;
use cu29_export::serde_to_jsonschema::trace_type_to_jsonschema;

// This will create the CuStampedDataSet that is specific to your copper project.
#[cfg(target_os = "macos")]
gen_cumsgs!("copperconfig.mac.ron");

#[cfg(target_os = "linux")]
gen_cumsgs!("copperconfig.linux.ron");

// Implement PayloadSchemas for MCAP export with proper JSON schemas
impl PayloadSchemas for cumsgs::CuStampedDataSet {
    fn get_payload_schemas() -> Vec<(&'static str, String)> {
        vec![
            ("camera", trace_type_to_jsonschema::<CuGstBuffer>()),
            (
                "gst_to_image",
                trace_type_to_jsonschema::<CuImage<Vec<u8>>>(),
            ),
            ("yolo", trace_type_to_jsonschema::<CuPoses>()),
            // "viz" is a sink task with no output, so no schema needed
        ]
    }
}

#[cfg(feature = "logreader")]
fn main() {
    // Initialize GStreamer if available (needed for deserialization of CuGstBuffer)
    #[cfg(feature = "logreader")]
    {
        use std::sync::atomic::Ordering;
        if !gstreamer::INITIALIZED.load(Ordering::SeqCst) {
            if let Err(e) = gstreamer::init() {
                eprintln!("Warning: Failed to initialize GStreamer: {:?}", e);
                // Continue anyway, as logreader may not need GStreamer for all operations
            }
        }
    }

    run_cli::<CuStampedDataSet>().expect("Failed to run the export CLI");
}
