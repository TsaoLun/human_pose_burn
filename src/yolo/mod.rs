pub mod model;
pub mod preprocess;
pub mod postprocess;

pub use model::{Multiples, YoloV8Pose};
pub use preprocess::preprocess;
pub use postprocess::postprocess;
