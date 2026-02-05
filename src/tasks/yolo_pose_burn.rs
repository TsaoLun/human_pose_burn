//! YOLOv8 Pose estimation Copper task with Burn backend

use crate::payloads::CuPoses;
use crate::yolo::{Multiples, YoloV8Pose, preprocess, postprocess};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Tensor};
use burn_store::{ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore, TensorSnapshot};
use cu_sensor_payloads::CuImage;
use cu29::prelude::*;
use std::marker::PhantomData;
use std::rc::Rc;

/// Adapter that upcasts any incoming F16/BF16 weights to F32 so the WGPU backend
/// (which expects 32-bit floats) won't hit validation errors.
#[derive(Debug, Clone, Default)]
struct CastToF32Adapter;

impl ModuleAdapter for CastToF32Adapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if snapshot.dtype == DType::F32 {
            return snapshot.clone();
        }

        let data_fn = snapshot.clone_data_fn();
        let shape = snapshot.shape.clone();
        let path_stack = snapshot.path_stack.clone().unwrap_or_default();
        let container_stack = snapshot.container_stack.clone().unwrap_or_default();
        let tensor_id = snapshot.tensor_id.unwrap_or_default();

        TensorSnapshot::from_closure(
            Rc::new(move || {
                let data = data_fn()?;
                Ok(data.convert_dtype(DType::F32))
            }),
            DType::F32,
            shape,
            path_stack,
            container_stack,
            tensor_id,
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Adapter that first applies PyTorch-to-Burn conversions (transpose, gamma/beta) then upcasts to F32.
#[derive(Debug, Clone, Default)]
struct PytorchF32Adapter {
    inner: PyTorchToBurnAdapter,
    caster: CastToF32Adapter,
}

impl ModuleAdapter for PytorchF32Adapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        let adapted = self.inner.adapt(snapshot);
        self.caster.adapt(&adapted)
    }

    fn get_alternative_param_name(&self, param_name: &str, container_type: &str) -> Option<String> {
        self.inner.get_alternative_param_name(param_name, container_type)
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// YOLOv8 Pose estimation task using Burn framework
///
/// Processes incoming camera images and outputs detected poses with keypoints.
/// Supports multiple backends: WGPU, CPU (NdArray), CUDA
pub struct YoloPoseBurn<B: Backend> {
    model: YoloV8Pose<B>,
    device: B::Device,
    conf_threshold: f32,
    iou_threshold: f32,
    _phantom: PhantomData<B>,
}

impl<B: Backend> Freezable for YoloPoseBurn<B> {}

impl<B: Backend> CuTask for YoloPoseBurn<B> {
    type Resources<'r> = ();
    type Input<'m> = input_msg!(CuImage<Vec<u8>>);
    type Output<'m> = output_msg!(CuPoses);

    fn new(config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        // Parse configuration
        let config = config.ok_or_else(|| CuError::from("YoloPoseBurn requires configuration"))?;

        let variant = config
            .get::<String>("variant")?
            .unwrap_or_else(|| "yolov8n-pose".to_string());

        let conf_threshold = config.get::<f64>("conf_threshold")?.unwrap_or(0.25) as f32;
        let iou_threshold = config.get::<f64>("iou_threshold")?.unwrap_or(0.7) as f32;

        // Determine model size from variant
        let multiples = match variant.as_str() {
            "yolov8n-pose" => Multiples::n(),
            "yolov8s-pose" => Multiples::s(),
            "yolov8m-pose" => Multiples::m(),
            "yolov8l-pose" => Multiples::l(),
            "yolov8x-pose" => Multiples::x(),
            _ => {
                return Err(CuError::from(format!(
                    "Unknown variant: {}. Use yolov8[n|s|m|l|x]-pose",
                    variant
                )));
            }
        };

        // Initialize backend device
        let device = B::Device::default();

        info!("Initializing YOLOv8 Pose model with variant: {}", &variant);
        
        // Initialize model
        let mut model = YoloV8Pose::new(multiples, 1, (17, 3), &device);

        // Download and load model weights from HuggingFace
        let weights_path = Self::load_model_weights(&variant)?;
        
        println!("Loading weights from: {}", weights_path.to_string_lossy());
        
        // Use Burn Store to load safetensors directly
        let mut store = SafetensorsStore::from_file(&weights_path)
            // Auto-convert Linear/BatchNorm naming AND upcast fp16/bf16 weights to f32 for WGPU
            .with_from_adapter(PytorchF32Adapter::default())
            // Remap numeric backbone indices from the checkpoint to our named fields
            .with_key_remapping(r"\.b1\.0\.", ".b1.stem1.")
            .with_key_remapping(r"\.b1\.1\.", ".b1.stem2.")
            .with_key_remapping(r"\.b2\.0\.", ".b2.c2f_small.")
            .with_key_remapping(r"\.b2\.1\.", ".b2.down.")
            .with_key_remapping(r"\.b2\.2\.", ".b2.c2f_big.")
            .with_key_remapping(r"\.b3\.0\.", ".b3.conv.")
            .with_key_remapping(r"\.b3\.1\.", ".b3.c2f.")
            .with_key_remapping(r"\.b4\.0\.", ".b4.conv.")
            .with_key_remapping(r"\.b4\.1\.", ".b4.c2f.")
            .with_key_remapping(r"\.b5\.0\.", ".b5.sppf.")
            .allow_partial(false)
            .validate(true);
        
        match model.load_from(&mut store) {
            Ok(result) => {
                let applied = result.applied.len();
                let missing = result.missing.len();
                let unused = result.unused.len();

                // Human-friendly summary
                info!("Weights loaded successfully:");
                info!("  - Applied: {} tensors", applied);
                if missing > 0 {
                    warning!("  - Missing: {} tensors: {:?}", missing, result.missing);
                }
                if unused > 0 {
                    warning!("  - Unused: {} tensors from file", unused);
                }

                // Also print to stdout so it's visible without log piping
                println!(
                    "Weights loaded successfully: applied={} missing={} unused={} partial_load=false validate=true variant={}",
                    applied,
                    missing,
                    unused,
                    variant
                );

                // Parse-friendly single-line summary for log collectors/grep.
                // Format: WEIGHTS_SUMMARY applied=<n> missing=<n> unused=<n> partial_load=false validate=true variant=<variant>
                info!(
                    "WEIGHTS_SUMMARY applied={} missing={} unused={} partial_load=false validate=true variant={}",
                    applied,
                    missing,
                    unused,
                    variant
                );

                // Quick sanity check: ensure BN params are f32 to satisfy WGPU expectations.
                let gamma_dtype = model.first_bn_dtype();
                println!("First BN gamma dtype: {:?}", gamma_dtype);
            }
            Err(e) => {
                // Mirror the error to stderr for visibility
                eprintln!("Failed to load weights: {}", e);
                return Err(CuError::from(format!("Failed to load weights: {}", e)));
            }
        }

        Ok(Self {
            model,
            device,
            conf_threshold,
            iou_threshold,
            _phantom: PhantomData,
        })
    }

    fn process(
        &mut self,
        _clock: &RobotClock,
        input: &Self::Input<'_>,
        output: &mut Self::Output<'_>,
    ) -> CuResult<()> {
        let Some(image) = input.payload() else {
            // No image received; don't emit poses to avoid clearing downstream overlays.
            output.clear_payload();
            return Ok(());
        };

        // Preprocess the image
        let (tensor, scaled_width, scaled_height) = preprocess(image, &self.device)
            .map_err(|e| CuError::from(format!("Failed to preprocess image: {}", e)))?;

        // Run inference
        let predictions = self.model.forward(tensor);
        
        // predictions is now [batch, channels, anchors] with shape like [1, 56, num_anchors]
        // We need to reshape to [56, num_anchors] for postprocess
        let [_batch, _channels, _anchors] = predictions.dims();
        let predictions: Tensor<B, 2> = predictions.squeeze(); // Remove batch dim

        // Postprocess predictions
        let poses = postprocess(
            predictions,
            image.format.width,
            image.format.height,
            scaled_width,
            scaled_height,
            self.conf_threshold,
            self.iou_threshold,
        );

        if !poses.is_empty() {
            info!("YoloPoseBurn: Detected {} poses", poses.len());
            for (i, pose) in poses.iter().enumerate() {
                let visible_kps = pose.keypoints.iter().filter(|kp| kp.confidence > 0.5).count();
                info!("  Pose {}: conf={:.2}, bbox=[{:.1}, {:.1}, {:.1}, {:.1}], visible_keypoints={}",
                    i, pose.confidence, pose.bbox[0], pose.bbox[1], pose.bbox[2], pose.bbox[3], visible_kps);
            }
        }

        output.set_payload(poses);
        output.tov = input.tov;

        Ok(())
    }
}

impl<B: Backend> YoloPoseBurn<B> {
    /// Load model weights, downloading from HuggingFace if necessary
    fn load_model_weights(variant: &str) -> CuResult<std::path::PathBuf> {
        // Map variant to model filename
        let size = match variant {
            "yolov8n-pose" => "n",
            "yolov8s-pose" => "s",
            "yolov8m-pose" => "m",
            "yolov8l-pose" => "l",
            "yolov8x-pose" => "x",
            _ => "n",
        };
        let filename = format!("yolov8{}-pose.safetensors", size);

        info!("Downloading model weights from HuggingFace: {}", &filename);

        // Try to download from HuggingFace Hub
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| CuError::new_with_cause("Failed to create HuggingFace API", e))?;

        let repo = api.model("lmz/candle-yolo-v8".to_string());

        let path = repo
            .get(&filename)
            .map_err(|e| CuError::new_with_cause("Failed to download model weights", e))?;

        info!("Model weights cached at: {}", path.to_string_lossy());
        Ok(path)
    }
}
