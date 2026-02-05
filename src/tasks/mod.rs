pub mod gst_to_image;
pub mod yolo_pose_burn;
pub mod rerun_viz;

pub use gst_to_image::GstToCuImage;
pub use rerun_viz::RerunPoseViz;

#[cfg(feature = "wgpu")]
pub type YoloPoseBurn = yolo_pose_burn::YoloPoseBurn<burn::backend::wgpu::Wgpu>;

#[cfg(feature = "cpu")]
pub type YoloPoseBurn = yolo_pose_burn::YoloPoseBurn<burn::backend::ndarray::NdArray>;

#[cfg(feature = "cuda")]
pub type YoloPoseBurn = yolo_pose_burn::YoloPoseBurn<burn::backend::Cuda>;