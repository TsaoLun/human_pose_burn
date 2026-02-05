//! YOLOv8 Pose model implementation using Burn framework
//!
//! Adapted from Candle implementation to Burn 0.20.1
//! Supports WGPU, CPU (NdArray), and CUDA backends

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d};
use burn::tensor::backend::Backend;
use burn::tensor::module::max_pool2d;
use burn::tensor::{activation, DType, Tensor};

/// Model size multipliers for different YOLOv8 variants
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Multiples {
    pub depth: f64,
    pub width: f64,
    pub ratio: f64,
}

impl Multiples {
    pub fn n() -> Self {
        Self {
            depth: 0.33,
            width: 0.25,
            ratio: 2.0,
        }
    }

    pub fn s() -> Self {
        Self {
            depth: 0.33,
            width: 0.50,
            ratio: 2.0,
        }
    }

    pub fn m() -> Self {
        Self {
            depth: 0.67,
            width: 0.75,
            ratio: 1.5,
        }
    }

    pub fn l() -> Self {
        Self {
            depth: 1.00,
            width: 1.00,
            ratio: 1.0,
        }
    }

    pub fn x() -> Self {
        Self {
            depth: 1.00,
            width: 1.25,
            ratio: 1.0,
        }
    }

    pub fn filters(&self) -> (usize, usize, usize) {
        let f1 = (256. * self.width) as usize;
        let f2 = (512. * self.width) as usize;
        let f3 = (512. * self.width * self.ratio) as usize;
        (f1, f2, f3)
    }
}

/// Upsampling layer using nearest neighbor interpolation
#[derive(Module, Debug, Clone)]
pub struct Upsample {
    scale_factor: usize,
}

impl Upsample {
    pub fn new(scale_factor: usize) -> Self {
        Self { scale_factor }
    }

    pub fn forward<B: Backend>(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_b, _c, _h, _w] = xs.dims();
        
        // Simple 2x nearest neighbor upsampling by repeating each pixel
        // Reshape to [b, c, h, 1, w], repeat on dim 3 to [b, c, h, 2, w], then flatten
        let xs: Tensor<B, 5> = xs.unsqueeze_dim(3); // [b, c, h, 1, w]
        let xs = xs.repeat(&[1, 1, 1, 2, 1]); // [b, c, h, 2, w]
        let xs: Tensor<B, 4> = xs.flatten(2, 3); // [b, c, h*2, w]
        let xs: Tensor<B, 5> = xs.unsqueeze_dim(4); // [b, c, h*2, w, 1]
        let xs = xs.repeat(&[1, 1, 1, 1, 2]); // [b, c, h*2, w, 2]
        xs.flatten(3, 4) // [b, c, h*2, w*2]
    }
}

/// Convolutional block: Conv2d + BatchNorm + SiLU activation
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(
        c1: usize,
        c2: usize,
        k: usize,
        stride: usize,
        padding: Option<usize>,
        device: &B::Device,
    ) -> Self {
        let padding = padding.unwrap_or(k / 2);
        
        let conv = Conv2dConfig::new([c1, c2], [k, k])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(padding, padding))
            .with_bias(false)
            .init(device);
        
        let bn = BatchNormConfig::new(c2)
            .with_epsilon(1e-3)
            .with_momentum(0.03)
            .init(device);
        
        Self { conv, bn }
    }

    pub fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        let xs = self.conv.forward(xs);
        let xs = self.bn.forward(xs);
        activation::silu(xs)
    }
}

/// Bottleneck block with optional residual connection
#[derive(Module, Debug)]
pub struct Bottleneck<B: Backend> {
    cv1: ConvBlock<B>,
    cv2: ConvBlock<B>,
    residual: bool,
}

impl<B: Backend> Bottleneck<B> {
    pub fn new(
        c1: usize,
        c2: usize,
        shortcut: bool,
        device: &B::Device,
    ) -> Self {
        let channel_factor = 1.0;
        let c_ = (c2 as f64 * channel_factor) as usize;
        
        let cv1 = ConvBlock::new(c1, c_, 3, 1, None, device);
        let cv2 = ConvBlock::new(c_, c2, 3, 1, None, device);
        let residual = c1 == c2 && shortcut;
        
        Self { cv1, cv2, residual }
    }

    pub fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        let ys = self.cv2.forward(self.cv1.forward(xs.clone()));
        if self.residual {
            xs + ys
        } else {
            ys
        }
    }
}

/// C2f module: CSP Bottleneck with 2 convolutions
/// Bottleneck count is dynamic so we can match depth-scaled YOLO variants.
#[derive(Module, Debug)]
pub struct C2f<B: Backend> {
    cv1: ConvBlock<B>,
    cv2: ConvBlock<B>,
    bottleneck: Vec<Bottleneck<B>>,
}

impl<B: Backend> C2f<B> {
    pub fn new(
        c1: usize,
        c2: usize,
        num_blocks: usize,
        shortcut: bool,
        device: &B::Device,
    ) -> Self {
        let c = (c2 as f64 * 0.5) as usize;
        let num_blocks = usize::max(1, num_blocks); // Ensure at least one block like reference implementation
        
        let cv1 = ConvBlock::new(c1, 2 * c, 1, 1, None, device);
        let cv2 = ConvBlock::new((2 + num_blocks) * c, c2, 1, 1, None, device);
        
        let bottleneck = (0..num_blocks)
            .map(|_| Bottleneck::new(c, c, shortcut, device))
            .collect();
        
        Self { cv1, cv2, bottleneck }
    }

    pub fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        let ys = self.cv1.forward(xs);
        
        // Split into chunks
        let mut chunks = ys.chunk(2, 1);
        
        // Apply bottlenecks sequentially
        for bottleneck in &self.bottleneck {
            let last = chunks.last().unwrap().clone();
            chunks.push(bottleneck.forward(last));
        }
        
        // Concatenate all chunks
        let zs = Tensor::cat(chunks, 1);
        self.cv2.forward(zs)
    }
}

/// SPPF: Spatial Pyramid Pooling - Fast
#[derive(Module, Debug)]
pub struct Sppf<B: Backend> {
    cv1: ConvBlock<B>,
    cv2: ConvBlock<B>,
    k: usize,
}

impl<B: Backend> Sppf<B> {
    pub fn new(c1: usize, c2: usize, k: usize, device: &B::Device) -> Self {
        let c_ = c1 / 2;
        let cv1 = ConvBlock::new(c1, c_, 1, 1, None, device);
        let cv2 = ConvBlock::new(c_ * 4, c2, 1, 1, None, device);
        
        Self { cv1, cv2, k }
    }

    pub fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        let xs1 = self.cv1.forward(xs);
        
        // Apply max pooling multiple times as in SPPF
        let y1 = max_pool2d(
            xs1.clone(),
            [self.k, self.k],
            [1, 1],
            [self.k / 2, self.k / 2],
            [1, 1],
            false,
        );
        let y2 = max_pool2d(
            y1.clone(),
            [self.k, self.k],
            [1, 1],
            [self.k / 2, self.k / 2],
            [1, 1],
            false,
        );
        let y3 = max_pool2d(
            y2.clone(),
            [self.k, self.k],
            [1, 1],
            [self.k / 2, self.k / 2],
            [1, 1],
            false,
        );
        
        let concatenated = Tensor::cat(vec![xs1, y1, y2, y3], 1);
        self.cv2.forward(concatenated)
    }
}

/// DarkNet backbone (CSPDarknet53)
/// YOLOv8n uses: n=3 for first C2f, n=6 for middle ones
#[derive(Module, Debug)]
struct B1<B: Backend> {
    stem1: ConvBlock<B>,
    stem2: ConvBlock<B>,
}

#[derive(Module, Debug)]
struct B2<B: Backend> {
    c2f_small: C2f<B>,
    down: ConvBlock<B>,
    c2f_big: C2f<B>,
}

#[derive(Module, Debug)]
struct B3<B: Backend> {
    conv: ConvBlock<B>,
    c2f: C2f<B>,
}

#[derive(Module, Debug)]
struct B4<B: Backend> {
    conv: ConvBlock<B>,
    c2f: C2f<B>,
}

#[derive(Module, Debug)]
struct B5<B: Backend> {
    sppf: Sppf<B>,
}

#[derive(Module, Debug)]
pub struct DarkNet<B: Backend> {
    b1: B1<B>, // b1.0, b1.1
    b2: B2<B>, // b2.0, b2.1, b2.2
    b3: B3<B>, // b3.0, b3.1
    b4: B4<B>, // b4.0, b4.1
    b5: B5<B>, // b5.0
}

impl<B: Backend> DarkNet<B> {
    pub fn new(m: Multiples, device: &B::Device) -> Self {
        let w = m.width;
        let r = m.ratio;
        let d = m.depth;
        let n3d = (3.0 * d).round() as usize;
        let n6d = (6.0 * d).round() as usize;
        
        let b1 = B1 {
            stem1: ConvBlock::new(3, (64.0 * w) as usize, 3, 2, Some(1), device),
            stem2: ConvBlock::new((64.0 * w) as usize, (128.0 * w) as usize, 3, 2, Some(1), device),
        };
        let b2 = B2 {
            c2f_small: C2f::new((128.0 * w) as usize, (128.0 * w) as usize, n3d, true, device),
            down: ConvBlock::new((128.0 * w) as usize, (256.0 * w) as usize, 3, 2, Some(1), device),
            c2f_big: C2f::new((256.0 * w) as usize, (256.0 * w) as usize, n6d, true, device),
        };
        let b3 = B3 {
            conv: ConvBlock::new((256.0 * w) as usize, (512.0 * w) as usize, 3, 2, Some(1), device),
            c2f: C2f::new((512.0 * w) as usize, (512.0 * w) as usize, n6d, true, device),
        };
        let b4 = B4 {
            conv: ConvBlock::new((512.0 * w) as usize, (512.0 * w * r) as usize, 3, 2, Some(1), device),
            c2f: C2f::new((512.0 * w * r) as usize, (512.0 * w * r) as usize, n3d, true, device),
        };
        let b5 = B5 {
            sppf: Sppf::new((512.0 * w * r) as usize, (512.0 * w * r) as usize, 5, device),
        };
        
        Self {
            b1, b2, b3, b4, b5,
        }
    }

    pub fn forward(&self, xs: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let x1 = self.b1.stem2.forward(self.b1.stem1.forward(xs));
        let x2 = self.b2.c2f_big.forward(self.b2.down.forward(self.b2.c2f_small.forward(x1)));
        let x3 = self.b3.c2f.forward(self.b3.conv.forward(x2.clone()));
        let x4 = self.b4.c2f.forward(self.b4.conv.forward(x3.clone()));
        let x5 = self.b5.sppf.forward(x4);
        (x2, x3, x5)
    }
}

/// YOLOv8 Neck (Feature Pyramid Network)
#[derive(Module, Debug)]
pub struct YoloV8Neck<B: Backend> {
    up: Upsample,
    n1: C2f<B>,
    n2: C2f<B>,
    n3: ConvBlock<B>,
    n4: C2f<B>,
    n5: ConvBlock<B>,
    n6: C2f<B>,
}

impl<B: Backend> YoloV8Neck<B> {
    pub fn new(m: Multiples, device: &B::Device) -> Self {
        let up = Upsample::new(2);
        let (w, r, d) = (m.width, m.ratio, m.depth);
        let n = (3.0 * d).round() as usize;
        
        let n1 = C2f::new((512.0 * w * (1.0 + r)) as usize, (512.0 * w) as usize, n, false, device);
        let n2 = C2f::new((768.0 * w) as usize, (256.0 * w) as usize, n, false, device);
        let n3 = ConvBlock::new((256.0 * w) as usize, (256.0 * w) as usize, 3, 2, Some(1), device);
        let n4 = C2f::new((768.0 * w) as usize, (512.0 * w) as usize, n, false, device);
        let n5 = ConvBlock::new((512.0 * w) as usize, (512.0 * w) as usize, 3, 2, Some(1), device);
        let n6 = C2f::new((512.0 * w * (1.0 + r)) as usize, (512.0 * w * r) as usize, n, false, device);
        
        Self { up, n1, n2, n3, n4, n5, n6 }
    }

    pub fn forward(
        &self,
        p3: Tensor<B, 4>,
        p4: Tensor<B, 4>,
        p5: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let x = self.n1.forward(Tensor::cat(vec![self.up.forward(p5.clone()), p4], 1));
        let head_1 = self.n2.forward(Tensor::cat(vec![self.up.forward(x.clone()), p3], 1));
        let head_2 = self.n4.forward(Tensor::cat(vec![self.n3.forward(head_1.clone()), x], 1));
        let head_3 = self.n6.forward(Tensor::cat(vec![self.n5.forward(head_2.clone()), p5], 1));
        (head_1, head_2, head_3)
    }
}

/// Distribution Focal Loss layer for bounding box prediction
#[derive(Module, Debug)]
pub struct Dfl<B: Backend> {
    conv: Conv2d<B>,
    num_classes: usize,
}

impl<B: Backend> Dfl<B> {
    pub fn new(num_classes: usize, device: &B::Device) -> Self {
        let conv = Conv2dConfig::new([num_classes, 1], [1, 1])
            .with_bias(false)
            .init(device);
        Self { conv, num_classes }
    }

    pub fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, _channels, anchors] = xs.dims();
        let xs = xs.reshape([b, 4, self.num_classes, anchors]);
        let xs = xs.swap_dims(2, 1); // transpose(2, 1)
        let xs = activation::softmax(xs, 1);
        let xs = self.conv.forward(xs);
        xs.reshape([b, 4, anchors])
    }
}

/// Pose detection head combining detection and keypoint prediction
#[derive(Module, Debug)]
pub struct PoseHead<B: Backend> {
    dfl: Dfl<B>,
    cv2: [(ConvBlock<B>, ConvBlock<B>, Conv2d<B>); 3],
    cv3: [(ConvBlock<B>, ConvBlock<B>, Conv2d<B>); 3],
    cv4: [(ConvBlock<B>, ConvBlock<B>, Conv2d<B>); 3],
    ch: usize,
    no: usize,
    kpt_shape: (usize, usize),
}

impl<B: Backend> PoseHead<B> {
    pub fn new(m: Multiples, nc: usize, kpt_shape: (usize, usize), device: &B::Device) -> Self {
        let filters = m.filters();
        let ch = 16;
        let dfl = Dfl::new(ch, device);
        let c1 = usize::max(filters.0, nc);
        let c2 = usize::max(filters.0 / 4, ch * 4);
        let nk = kpt_shape.0 * kpt_shape.1;
        let c4 = usize::max(filters.0 / 4, nk);

        let cv3 = [
            Self::new_cv3(c1, nc, filters.0, device),
            Self::new_cv3(c1, nc, filters.1, device),
            Self::new_cv3(c1, nc, filters.2, device),
        ];

        let cv2 = [
            Self::new_cv2(c2, ch, filters.0, device),
            Self::new_cv2(c2, ch, filters.1, device),
            Self::new_cv2(c2, ch, filters.2, device),
        ];

        let cv4 = [
            Self::new_cv4(c4, nk, filters.0, device),
            Self::new_cv4(c4, nk, filters.1, device),
            Self::new_cv4(c4, nk, filters.2, device),
        ];

        let no = nc + ch * 4;

        Self { dfl, cv2, cv3, cv4, ch, no, kpt_shape }
    }

    fn new_cv3(c1: usize, nc: usize, filter: usize, device: &B::Device) -> (ConvBlock<B>, ConvBlock<B>, Conv2d<B>) {
        let block0 = ConvBlock::new(filter, c1, 3, 1, None, device);
        let block1 = ConvBlock::new(c1, c1, 3, 1, None, device);
        let conv = Conv2dConfig::new([c1, nc], [1, 1]).init(device);
        (block0, block1, conv)
    }

    fn new_cv2(c2: usize, ch: usize, filter: usize, device: &B::Device) -> (ConvBlock<B>, ConvBlock<B>, Conv2d<B>) {
        let block0 = ConvBlock::new(filter, c2, 3, 1, None, device);
        let block1 = ConvBlock::new(c2, c2, 3, 1, None, device);
        let conv = Conv2dConfig::new([c2, 4 * ch], [1, 1]).init(device);
        (block0, block1, conv)
    }

    fn new_cv4(c1: usize, nc: usize, filter: usize, device: &B::Device) -> (ConvBlock<B>, ConvBlock<B>, Conv2d<B>) {
        let block0 = ConvBlock::new(filter, c1, 3, 1, None, device);
        let block1 = ConvBlock::new(c1, c1, 3, 1, None, device);
        let conv = Conv2dConfig::new([c1, nc], [1, 1]).init(device);
        (block0, block1, conv)
    }

    pub fn forward(
        &self,
        xs0: Tensor<B, 4>,
        xs1: Tensor<B, 4>,
        xs2: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        // Detection branch (boxes + classes)
        let forward_cv = |xs: Tensor<B, 4>, i: usize| -> Tensor<B, 4> {
            let xs_2 = self.cv2[i].0.forward(xs.clone());
            let xs_2 = self.cv2[i].1.forward(xs_2);
            let xs_2 = self.cv2[i].2.forward(xs_2);

            let xs_3 = self.cv3[i].0.forward(xs);
            let xs_3 = self.cv3[i].1.forward(xs_3);
            let xs_3 = self.cv3[i].2.forward(xs_3);
            
            Tensor::cat(vec![xs_2, xs_3], 1)
        };

        let xs0_det = forward_cv(xs0.clone(), 0);
        let xs1_det = forward_cv(xs1.clone(), 1);
        let xs2_det = forward_cv(xs2.clone(), 2);

        let (anchors, strides) = make_anchors(&xs0_det, &xs1_det, &xs2_det, (8, 16, 32), 0.5);
        let anchors = anchors.swap_dims(0, 1);  // [hw_total, 2]
        let strides = strides.swap_dims(0, 1);  // [hw_total, 1]

        let reshape = |xs: Tensor<B, 4>| -> Tensor<B, 3> {
            let [d, _, h, w] = xs.dims();
            xs.reshape([d, self.no, h * w])
        };
        
        let ys0 = reshape(xs0_det);
        let ys1 = reshape(xs1_det);
        let ys2 = reshape(xs2_det);

        let x_cat = Tensor::cat(vec![ys0, ys1, ys2], 2);
        
        // Split box and class predictions
        let [_b, _no, _hw] = x_cat.dims();
        let box_end = self.ch * 4;
        
        let box_ = x_cat.clone().narrow(1, 0, box_end);
        let cls = x_cat.narrow(1, box_end, self.no - box_end);

        let dbox = dist2bbox(self.dfl.forward(box_), anchors.clone());
        let dbox = dbox.mul(strides.clone().unsqueeze_dim(0));
        let cls_sigmoid = activation::sigmoid(cls);
        let pred = Tensor::cat(vec![dbox, cls_sigmoid], 1);

        // Keypoint branch
        let forward_cv_kpt = |xs: Tensor<B, 4>, i: usize| -> Tensor<B, 3> {
            let [b, _, h, w] = xs.dims();
            let xs = self.cv4[i].0.forward(xs);
            let xs = self.cv4[i].1.forward(xs);
            let xs = self.cv4[i].2.forward(xs);
            xs.reshape([b, self.kpt_shape.0 * self.kpt_shape.1, h * w])
        };

        let xs0_k = forward_cv_kpt(xs0, 0);
        let xs1_k = forward_cv_kpt(xs1, 1);
        let xs2_k = forward_cv_kpt(xs2, 2);

        let xs = Tensor::cat(vec![xs0_k, xs1_k, xs2_k], 2);
        let [b, _nk, hw] = xs.dims();
        let xs = xs.reshape([b, self.kpt_shape.0, self.kpt_shape.1, hw]);

        let xs_xy = xs.clone().narrow(2, 0, 2);
        let xs_conf = xs.narrow(2, 2, 1);

        let anchors_step1: Tensor<B, 3> = anchors.clone().unsqueeze_dim(0);  // [1, hw_total, 2]
        let anchors_expanded: Tensor<B, 4> = anchors_step1.unsqueeze_dim(0);  // [1, 1, hw_total, 2]

        let strides_step1: Tensor<B, 3> = strides.unsqueeze_dim(0);  // [1, hw_total, 1]
        let strides_expanded: Tensor<B, 4> = strides_step1.unsqueeze_dim(0);  // [1, 1, hw_total, 1]

        let ys01 = xs_xy.mul_scalar(2.0).add(anchors_expanded)
            .sub_scalar(0.5).mul(strides_expanded);
        let ys2 = activation::sigmoid(xs_conf);
        let ys = Tensor::cat(vec![ys01, ys2], 2).flatten(1, 2);

        Tensor::cat(vec![pred, ys], 1)
    }
}

/// Create anchor points and stride tensors for multi-scale detection
fn make_anchors<B: Backend>(
    xs0: &Tensor<B, 4>,
    xs1: &Tensor<B, 4>,
    xs2: &Tensor<B, 4>,
    (s0, s1, s2): (usize, usize, usize),
    grid_cell_offset: f32,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let device = xs0.device();
    let mut anchor_points: Vec<Tensor<B, 2>> = vec![];
    let mut stride_tensors: Vec<Tensor<B, 1>> = vec![];
    
    for (xs, stride) in [(xs0, s0), (xs1, s1), (xs2, s2)] {
        let [_b, _c, h, w] = xs.dims();
        
        // Create grid coordinates using IntTensor::arange then convert to float
        use burn::tensor::Int;
        let sx = Tensor::<B, 1, Int>::arange(0..w as i64, &device).float() + grid_cell_offset;
        let sy = Tensor::<B, 1, Int>::arange(0..h as i64, &device).float() + grid_cell_offset;
        
        let sx: Tensor<B, 2> = sx.reshape([1, w]).repeat(&[h, 1]);
        let sx: Tensor<B, 1> = sx.flatten(0, 1);
        let sy: Tensor<B, 2> = sy.reshape([h, 1]).repeat(&[1, w]);
        let sy: Tensor<B, 1> = sy.flatten(0, 1);
        
        let anchor: Tensor<B, 2> = Tensor::stack(vec![sx, sy], 1);
        anchor_points.push(anchor);
        stride_tensors.push(Tensor::ones([h * w], &device).mul_scalar(stride as f32));
    }
    
    let anchor_points = Tensor::cat(anchor_points, 0);
    let stride_tensor: Tensor<B, 2> = Tensor::cat(stride_tensors, 0).unsqueeze_dim(1);
    (anchor_points, stride_tensor)
}

/// Convert distance predictions to bounding boxes
fn dist2bbox<B: Backend>(distance: Tensor<B, 3>, anchor_points: Tensor<B, 2>) -> Tensor<B, 3> {
    let chunks = distance.chunk(2, 1);
    let lt = &chunks[0];
    let rb = &chunks[1];
    
    let anchor_expanded = anchor_points.unsqueeze_dim(0);
    let x1y1 = anchor_expanded.clone().sub(lt.clone());
    let x2y2 = anchor_expanded.clone().add(rb.clone());
    let c_xy = x1y1.clone().add(x2y2.clone()).mul_scalar(0.5);
    let wh = x2y2.sub(x1y1);
    
    Tensor::cat(vec![c_xy, wh], 1)
}

/// Complete YOLOv8 Pose model
#[derive(Module, Debug)]
pub struct YoloV8Pose<B: Backend> {
    net: DarkNet<B>,
    fpn: YoloV8Neck<B>,
    head: PoseHead<B>,
}

impl<B: Backend> YoloV8Pose<B> {
    pub fn new(
        multiples: Multiples,
        nc: usize,
        kpt_shape: (usize, usize),
        device: &B::Device,
    ) -> Self {
        let net = DarkNet::new(multiples, device);
        let fpn = YoloV8Neck::new(multiples, device);
        let head = PoseHead::new(multiples, nc, kpt_shape, device);
        
        Self { net, fpn, head }
    }

    pub fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 3> {
        let (x2, x3, x5) = self.net.forward(xs);
        let (head_1, head_2, head_3) = self.fpn.forward(x2, x3, x5);
        self.head.forward(head_1, head_2, head_3)
    }

    /// Helper for debugging: report dtype of the very first BatchNorm gamma.
    pub fn first_bn_dtype(&self) -> DType {
        self.net.b1.stem1.bn.gamma.val().dtype()
    }
}
