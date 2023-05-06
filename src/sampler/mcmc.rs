use std::f32::consts::PI;

use crate::{sampler::*, sampling::sample_gaussian, *};

use super::IndependentSampler;
pub struct PrimarySample {
    pub values: VLArrayExpr<f32>,
}
pub struct Proposal {
    pub sample: PrimarySample,
    pub log_prob: Expr<f32>,
}
pub trait Mutation {
    fn weight(&self) -> f32;
    fn proposal(&self, s: &PrimarySample, sampler: &dyn IndependentSampler) -> Proposal;
}
pub struct CombinedPSSMutation {
    pub mutations: Vec<Box<dyn Mutation>>,
}
impl CombinedPSSMutation {
    pub fn with_mutation<R: Aggregate>(&self, i: Expr<u32>, f: impl Fn(&dyn Mutation) -> R) -> R {
        let mut sw = SwitchBuilder::new(i.int());
        for (i, m) in self.mutations.iter().enumerate() {
            sw = sw.case(i as i32, || f(m.as_ref()));
        }
        sw.finish()
    }
    pub fn sample_mutation(&self, u: Expr<f32>) -> Expr<u32> {
        let mut sum = 0.0;
        for m in &self.mutations {
            sum += m.weight();
        }
        let mut u = u * sum;
        let ret = var!(u32, 0);
        for (i, m) in self.mutations.iter().enumerate() {
            let last = i == self.mutations.len() - 1;
            if_!(u.cmplt(m.weight()) | last, {
                ret.store(i as u32);
            });
            u -= m.weight();
        }
        ret.load()
    }
}
pub struct AnisotropicDiagonalGaussianMutation {
    pub sigma: VLArrayExpr<f32>,
    pub weight: f32,
}
impl Mutation for AnisotropicDiagonalGaussianMutation {
    fn weight(&self) -> f32 {
        self.weight
    }
    fn proposal(&self, s: &PrimarySample, sampler: &dyn IndependentSampler) -> Proposal {
        let values = VLArrayVar::<f32>::zero(s.values.static_len());
        let log_prob = var!(f32, 0.0f32);
        let det_sigma = var!(f32, 0.0);
        for_range(const_(0)..values.len().int(), |i| {
            let i = i.uint();
            let cur = s.values.read(i);
            let sigma = self.sigma.read(i);
            let x = sample_gaussian(sampler.next_1d());
            det_sigma.store(det_sigma.load() - sigma);
            log_prob.store(log_prob.load() + (-0.5f32 * x * x / sigma));
            let new = cur + x * sigma.sqrt();
            let new = new - new.floor();
            values.write(i, new);
        });
        let log_prob = log_prob.load()
            + det_sigma.load() * 0.5
            + values.len().float() * 0.5 * (2.0f32 * PI).ln();
        Proposal {
            sample: PrimarySample {
                values: values.load(),
            },
            log_prob,
        }
    }
}
pub struct IsotropicGaussianMutation {
    pub sigma: Expr<f32>,
    pub weight: f32,
}
impl Mutation for IsotropicGaussianMutation {
    fn weight(&self) -> f32 {
        self.weight
    }
    fn proposal(&self, s: &PrimarySample, sampler: &dyn IndependentSampler) -> Proposal {
        let values = VLArrayVar::<f32>::zero(s.values.static_len());
        let log_prob = var!(f32, 0.0f32);
        for_range(const_(0)..values.len().int(), |i| {
            let i = i.uint();
            let cur = s.values.read(i);
            let x = sample_gaussian(sampler.next_1d());
            log_prob.store(log_prob.load() + (-0.5f32 * x * x / self.sigma));
            let new = cur + x * self.sigma.sqrt();
            let new = new - new.floor();
            values.write(i, new);
        });
        let det_sigma = self.sigma * values.len().float() * -0.5;
        let log_prob =
            log_prob.load() + det_sigma + values.len().float() * 0.5 * (2.0f32 * PI).ln();
        Proposal {
            sample: PrimarySample {
                values: values.load(),
            },
            log_prob,
        }
    }
}
pub struct LargeStepMutation {
    pub weight: f32,
}
impl Mutation for LargeStepMutation {
    fn weight(&self) -> f32 {
        self.weight
    }
    fn proposal(&self, s: &PrimarySample, sampler: &dyn IndependentSampler) -> Proposal {
        let values = VLArrayVar::<f32>::zero(s.values.static_len());
        for_range(const_(0)..values.len().int(), |i| {
            let i = i.uint();
            values.write(i, sampler.next_1d());
        });
        Proposal {
            sample: PrimarySample {
                values: values.load(),
            },
            log_prob: const_(0.0f32),
        }
    }
}

pub fn acceptance_prob(
    log_cur_f: Expr<f32>,
    cur: &Proposal,
    log_proposal_f: Expr<f32>,
    proposal: &Proposal,
) -> Expr<f32> {
    (log_proposal_f - log_cur_f + proposal.log_prob - cur.log_prob)
        .exp()
        .clamp(0.0, 1.0)
}

pub struct MetroplisSampler<S: IndependentSampler> {
    pub base: S,
    pub sample: PrimarySample,
    pub cur_dim: Var<u32>,
}
impl<S: IndependentSampler> MetroplisSampler<S> {
    pub fn new(base: S, sample: PrimarySample) -> Self {
        Self {
            base,
            sample,
            cur_dim: var!(u32, 0),
        }
    }
}
impl<S: IndependentSampler> Sampler for MetroplisSampler<S> {
    fn next_1d(&self) -> Float {
        if_!(self.cur_dim.load().cmplt(self.sample.values.len()), {
            let ret = self.sample.values.read(self.cur_dim.load());
            self.cur_dim.store(self.cur_dim.load() + 1);
            ret
        }, else {
            self.base.next_1d()
        })
    }
    fn start(&self) {
        self.cur_dim.store(0);
    }
}
