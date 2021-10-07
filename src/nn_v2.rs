#![allow(dead_code)]
use nalgebra as na;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
#[derive(Clone)]
pub struct Dual<T> {
    pub val: T,
    pub grad: T,
}
type MatrixXf = na::DMatrix<f32>;
type VectorXf = na::DVector<f32>;
// pub trait Activation: Send + Sync {
//     fn new() -> Self
//     where
//         Self: Sized;
//     fn forward(&self, x: &MatrixXf) -> MatrixXf;
//     fn backward(&self, x: &MatrixXf, out: Dual<&MatrixXf>) -> MatrixXf;
// }
struct Relu {}
impl Relu {
    fn forward(x: &mut MatrixXf) {
        // x.map(|v| v.max(0.0))
        x.iter_mut().for_each(|x| *x = x.max(0.0))
    }
    fn backward(x: &MatrixXf, out: Dual<&MatrixXf>) -> MatrixXf {
        MatrixXf::from_fn(x.nrows(), x.ncols(), |r, c| {
            if x[(r, c)] > 0.0 {
                out.grad[(r, c)]
            } else {
                0.0
            }
        })
    }
}
struct Sigmoid {}
impl Sigmoid {
    fn forward(x: &mut MatrixXf) {
        x.iter_mut().for_each(|x| *x = 1.0 / (1.0 + (-*x).exp()))
    }
    fn backward(x: &MatrixXf, out: Dual<&MatrixXf>) -> MatrixXf {
        MatrixXf::from_fn(x.nrows(), x.ncols(), |r, c| {
            let s = out.val[(r, c)];
            s * (1.0 - s) * out.grad[(r, c)]
        })
    }
}
// struct Exp {}
// impl Exp {
//     fn forward(x: &MatrixXf) -> MatrixXf {
//         x.map(|v| v.exp())
//     }
//     fn backward(x: &MatrixXf, out: Dual<&MatrixXf>) -> MatrixXf {
//         MatrixXf::from_fn(x.nrows(), x.ncols(), |r, c| {
//             let s = out.val[(r, c)];
//             s * out.grad[(r, c)]
//         })
//     }
// }
// struct Sin {}
// impl Sin {
//     fn forward(x: &MatrixXf) -> MatrixXf {
//         x.map(|v| v.sin())
//     }
//     fn backward(x: &MatrixXf, out: Dual<&MatrixXf>) -> MatrixXf {
//         MatrixXf::from_fn(x.nrows(), x.ncols(), |r, c| {
//             let s = x[(r, c)];
//             s.cos() * out.grad[(r, c)]
//         })
//     }
// }
#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    Relu,
    Sigmoid,
    // Exp,
    // Sin,
}
impl Activation {
    pub fn forward(&self, x: &mut MatrixXf) {
        match self {
            Self::Relu => Relu::forward(x),
            Self::Sigmoid => Sigmoid::forward(x),
            // Self::Exp => Exp::forward(x),
            // Self::Sin => Sin::forward(x),
        }
    }
    pub fn backward(&self, x: &MatrixXf, out: Dual<&MatrixXf>) -> MatrixXf {
        match self {
            Self::Relu => Relu::backward(x, out),
            Self::Sigmoid => Sigmoid::backward(x, out),
            // Self::Exp => Exp::backward(x, out),
            // Self::Sin => Exp::backward(x, out),
        }
    }
}
pub struct Layer {
    pub inputs: usize,
    pub outputs: usize,
    pub activation: Option<Activation>,
    pub bias: bool,
}
impl Layer {
    pub fn new(act: Option<Activation>, inputs: usize, outputs: usize, bias: bool) -> Self {
        Layer {
            inputs,
            outputs,
            activation: act,
            bias,
        }
    }
}
pub trait Optimizer: Clone {
    fn create(&self, n: usize) -> Box<dyn OptimizerImpl>;
}
pub trait OptimizerImpl: Send + Sync {
    fn step(&mut self, val: &mut [f32], grad: &[f32]);
}

#[derive(Copy, Clone)]
pub struct SGDParams {
    pub learning_rate: f32,
}
impl Default for SGDParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.003,
        }
    }
}
#[derive(Clone)]
pub struct SGD {
    params: Arc<RwLock<SGDParams>>,
}
impl SGD {
    pub fn new(params: SGDParams) -> Self {
        Self {
            params: Arc::new(RwLock::new(params)),
        }
    }
}
struct SGDImpl {
    params: Arc<RwLock<SGDParams>>,
}
impl Optimizer for SGD {
    fn create(&self, _n: usize) -> Box<dyn OptimizerImpl> {
        Box::new(SGDImpl {
            params: self.params.clone(),
        })
    }
}
impl OptimizerImpl for SGDImpl {
    fn step(&mut self, val: &mut [f32], grad: &[f32]) {
        let lr = self.params.read().unwrap().learning_rate;
        assert!(val.len() == grad.len());
        for i in 0..val.len() {
            val[i] -= lr * grad[i].min(1000.0).max(-1000.0);
        }
    }
}
#[derive(Clone, Copy)]
pub struct AdamParams {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
}
impl Default for AdamParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
        }
    }
}
#[derive(Clone)]
pub struct Adam {
    params: Arc<RwLock<AdamParams>>,
}
impl Adam {
    pub fn new(params: AdamParams) -> Self {
        Self {
            params: Arc::new(RwLock::new(params)),
        }
    }
}
impl Optimizer for Adam {
    fn create(&self, n: usize) -> Box<dyn OptimizerImpl> {
        Box::new(AdamImpl {
            params: self.params.clone(),
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        })
    }
}
struct AdamImpl {
    params: Arc<RwLock<AdamParams>>,
    m: Vec<f32>,
    v: Vec<f32>,
    t: i32,
}
impl OptimizerImpl for AdamImpl {
    fn step(&mut self, val: &mut [f32], grad: &[f32]) {
        let params = self.params.read().unwrap();
        let lr = params.learning_rate;
        let beta1 = params.beta1;
        let beta2 = params.beta2;
        self.t += 1;

        for i in 0..val.len() {
            let grad = grad[i].min(1000.0).max(-1000.0);
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * grad;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * grad * grad;
        }
        let b1 = beta1.powi(self.t);
        let b2 = beta2.powi(self.t);
        for i in 0..val.len() {
            let m_tilde = self.m[i] / (1.0 - b1);
            let v_tilde = self.v[i] / (1.0 - b2);
            val[i] -= lr * (m_tilde / (v_tilde.sqrt() + 1e-8f32)) as f32;
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
struct LayerData {
    weights: MatrixXf,
    bias: VectorXf,
    activation: Option<Activation>,
    use_bias: bool,
}
struct LayerOptimizer {
    weights: Box<dyn OptimizerImpl>,
    bias: Box<dyn OptimizerImpl>,
}
use rand::distributions::Distribution;
use statrs::distribution::Normal;
struct LayerOutput {
    linear_out: MatrixXf,
    out: MatrixXf,
}
impl LayerData {
    fn num_inputs(&self) -> usize {
        self.weights.ncols()
    }
    fn num_outputs(&self) -> usize {
        self.weights.nrows()
    }
    fn new(inputs: usize, outputs: usize, activation: Option<Activation>, use_bias: bool) -> Self {
        let mut rng = rand::thread_rng();
        let n = Normal::new(0.0, (2.0 / (inputs + outputs) as f64).sqrt()).unwrap();
        let weights = MatrixXf::from_fn(outputs, inputs, |_r, _c| n.sample(&mut rng) as f32);
        let bias = if use_bias {
            VectorXf::from_fn(outputs, |_r, _c| 0.0f32)
        } else {
            VectorXf::zeros(0)
        };
        Self {
            weights,
            bias,
            activation,
            use_bias,
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct MLPData {
    layers: Vec<LayerData>,
}
pub struct MLP {
    layers: Vec<LayerData>,
    opts: Vec<LayerOptimizer>,
}

pub enum Loss {
    L2,
    RelativeL2,
}
impl MLP {
    pub fn from_data_opt<O: Optimizer>(data: MLPData, opt: O) -> Self {
        let layers = data.layers;
        let mut opts = vec![];
        for l in &layers {
            opts.push(LayerOptimizer {
                weights: opt.create(l.num_inputs() * l.num_outputs()),
                bias: opt.create(l.num_outputs()),
            })
        }
        Self { layers, opts }
    }
    pub fn from_data(data: MLPData) -> Self {
        Self {
            layers: data.layers,
            opts: vec![],
        }
    }
    pub fn data(self) -> MLPData {
        MLPData {
            layers: self.layers,
        }
    }
    pub fn new<O: Optimizer>(desc: Vec<Layer>, opt: O) -> Self {
        let mut layers = vec![];
        let mut opts = vec![];
        for d in desc {
            layers.push(LayerData::new(d.inputs, d.outputs, d.activation, d.bias));
            opts.push(LayerOptimizer {
                weights: opt.create(d.inputs * d.outputs),
                bias: opt.create(d.outputs),
            })
        }
        Self { layers, opts }
    }
    fn forward(&self, mut x: MatrixXf, tmp: &mut Vec<LayerOutput>, training: bool) -> MatrixXf {
        for layer in &self.layers {
            x = &layer.weights * x;
            if layer.use_bias {
                // x = MatrixXf::from_fn(x.nrows(), x.ncols(), |r, c| x[(r, c)] + layer.bias[r]);
                for c in 0..x.ncols() {
                    for r in 0..x.nrows() {
                        x[(r, c)] += layer.bias[r];
                    }
                }
            }
            let linear_out = if training { Some(x.clone()) } else { None };
            if let Some(f) = &layer.activation {
                // x = f.forward(&x);
                f.forward(&mut x);
            }
            if training {
                tmp.push(LayerOutput {
                    linear_out: linear_out.unwrap(),
                    out: x.clone(),
                })
            }
        }
        x
    }
    pub fn infer(&self, x: MatrixXf) -> MatrixXf {
        let mut tmp = vec![];
        self.forward(x, &mut tmp, false)
    }
    pub fn train(&mut self, x: MatrixXf, target: &MatrixXf, loss_fn: Loss) -> f32 {
        assert!(
            !self.opts.is_empty(),
            "model is not configured for training!"
        );
        let mut tmp = vec![];
        let y = self.forward(x.clone(), &mut tmp, true);
        let (loss, dy) = match loss_fn {
            Loss::L2 => {
                let loss = (&y - target).map(|v| v * v).mean();
                let dy = 0.5 * (&y - target) / (y.ncols() * y.nrows()) as f32;
                (loss, dy)
            }
            Loss::RelativeL2 => {
                let loss = y
                    .iter()
                    .zip(target.iter())
                    .map(|(y, target)| -> f32 { (*y - *target).powi(2) / (*y * *y + 0.01) })
                    .sum::<f32>()
                    / (y.ncols() * y.nrows()) as f32;
                let dy = na::DMatrix::from_fn(y.nrows(), y.ncols(), |r, c| -> f32 {
                    0.5 * (y[(r, c)] - target[(r, c)]) / (y[(r, c)] * y[(r, c)] + 0.01)
                }) / (y.ncols() * y.nrows()) as f32;
                // let dy = 0.5 * (&y - target).component_div(y.component_muml(y)) / (y.ncols() * y.nrows()) as f32;
                (loss, dy)
            }
        };

        let mut dout = dy;
        for i in (0..self.layers.len()).rev() {
            let layer = &self.layers[i];
            let use_bias = layer.use_bias;
            let out = &tmp[i].out;
            let linear_out = &tmp[i].linear_out;
            if let Some(f) = &layer.activation {
                let gradf = f.backward(
                    linear_out,
                    Dual {
                        val: out,
                        grad: &dout,
                    },
                );
                // out = linear_out;
                dout = gradf;
            }
            let input = if i == 0 { &x } else { &tmp[i - 1].out };
            let (dw, dbias, dx) = {
                let dbias = dout.column_sum();
                let dw = &dout * input.transpose();
                let dx = &layer.weights.transpose() * &dout;
                (dw, dbias, dx)
            };
            let opt = &mut self.opts[i];
            opt.weights
                .as_mut()
                .step(self.layers[i].weights.as_mut_slice(), dw.as_slice());

            if use_bias {
                opt.bias
                    .as_mut()
                    .step(self.layers[i].bias.as_mut_slice(), dbias.as_slice());
            }
            dout = dx;
        }
        loss
    }
}
#[macro_export]
macro_rules! position_encoding_func_v2 {
    // N: #inputs need mapping
    // E: #encoders
    // R: #inputs don't need mapping
    // input is [N; R]
    ($name:ident, $N:expr, $E:expr, $M:expr, $R: expr) => {
        fn $name(v: &na::DMatrix<f32>) -> na::DMatrix<f32> {
            assert!(v.nrows() == $N + $R);
            let mut u: na::DMatrix<f32> = na::DMatrix::zeros($N * $E * 2 + $N + $R, v.ncols());
            for c in 0..v.ncols() {
                for i in 0..$N {
                    for j in 0..$E {
                        let feq = 2.0f32.powf($M as f32 * (j as f32) / ($E - 1) as f32);
                        u[(i * $E + j, c)] = (v[(i, c)] * feq).sin();
                        u[(i * $E + j + $N * $E, c)] = (v[(i, c)] * feq).cos();
                    }
                    u[($N * $E * 2 + i, c)] = v[(i, c)];
                }
                for i in 0..$R {
                    u[($N * $E * 2 + $N + i, c)] = v[($N + i, c)];
                }
            }
            u
        }
    };
}
#[macro_export]
macro_rules! position_encoding_func_v3 {
    ($name:ident, $N:expr, $E:expr, $F:expr) => {
        fn $name(v: &na::DMatrix<f32>) -> na::DMatrix<f32> {
            assert!(v.nrows() == $N);
            let mut u: na::DMatrix<f32> = na::DMatrix::zeros($N * $E * 2 + $N, v.ncols());
            for c in 0..v.ncols() {
                for i in 0..$N {
                    for j in 0..$E {
                        let feq = 2.0f32.powf((j as f32 / $E as f32) * $F);
                        u[(i * $E + j, c)] = (v[(i, c)] * feq).sin();
                        u[(i * $E + j + $N * $E, c)] = (v[(i, c)] * feq).cos();
                    }
                    u[($N * $E * 2 + i, c)] = v[(i, c)];
                }
            }
            u
        }
    };
}
