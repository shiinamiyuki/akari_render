use util::RobustSum;

use crate::*;
pub trait Function1D: Send + Sync {
    fn evaluate(&self, x: f32) -> f32;
    fn domain(&self) -> (f32, f32);
    fn inner_product<F: Function1D, G: Function1D>(f: &F, g: &G, dx: f32) -> f32 {
        let f_dom = f.domain();
        let g_dom = g.domain();
        let x_min = f_dom.0.max(g_dom.0);
        let x_max = f_dom.1.min(g_dom.1);
        let mut x = x_min;
        let mut s = RobustSum::new(0.0);
        while x < x_max {
            s.add(f.evaluate(x) * g.evaluate(x));
            x += dx;
        }
        s.sum() * dx
    }
}

#[derive(Clone)]
pub struct Dense1D {
    domain: (f32, f32),
    ys: Vec<f32>,
}
impl Dense1D {
    pub fn new(domain: (f32, f32), ys: Vec<f32>) -> Self {
        Self { domain, ys }
    }
}
impl Function1D for Dense1D {
    fn domain(&self) -> (f32, f32) {
        self.domain
    }
    fn evaluate(&self, x: f32) -> f32 {
        if x > self.domain.1 || x < self.domain.0 {
            return 0.0;
        }
        let t = (x - self.domain.0) / (self.domain.1 - self.domain.0);
        let i = (t as usize).min(self.ys.len() - 2);
        let off = t.fract();
        lerp(self.ys[i], self.ys[i + 1], off)
    }
}
#[derive(Clone)]
pub struct DenseSlice1D {
    domain: (f32, f32),
    ys: &'static [f32],
}
impl DenseSlice1D {
    pub const fn new(domain: (f32, f32), ys: &'static [f32]) -> Self {
        Self { domain, ys }
    }
}
impl Function1D for DenseSlice1D {
    fn domain(&self) -> (f32, f32) {
        self.domain
    }
    fn evaluate(&self, x: f32) -> f32 {
        if x > self.domain.1 || x < self.domain.0 {
            return 0.0;
        }
        let t = (x - self.domain.0) / (self.domain.1 - self.domain.0);
        let i = (t as usize).min(self.ys.len() - 2);
        let off = t.fract();
        lerp(self.ys[i], self.ys[(i + 1).min(self.ys.len() - 1)], off)
    }
}
#[derive(Clone)]
pub struct PiecewiseLinear1D {
    ys: Vec<f32>,
    xs: Vec<f32>,
}
impl PiecewiseLinear1D {
    pub fn from_interleaved(yx: &[f32]) -> Self {
        assert!(yx.len() % 2 == 0);
        let mut xs: Vec<f32> = vec![];
        let mut ys: Vec<f32> = vec![];
        for i in 0..yx.len() / 2 {
            xs[i] = yx[2 * i + 1];
            ys[i] = yx[2 * i];
        }
        Self { xs, ys }
    }
}

impl Function1D for PiecewiseLinear1D {
    fn domain(&self) -> (f32, f32) {
        (self.xs[0], *self.xs.last().unwrap())
    }
    fn evaluate(&self, x: f32) -> f32 {
        let i = find_largest(&self.xs, |x_| *x_ <= x);
        lerp(
            self.ys[i],
            self.ys[i + 1],
            (x - self.xs[i]) / (self.xs[i + 1] - self.xs[i]),
        )
    }
}
