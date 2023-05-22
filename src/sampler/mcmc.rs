use std::f32::consts::PI;

use crate::{
    sampler::*,
    sampling::{log_gaussian_pdf, sample_gaussian},
};

use super::IndependentSampler;
#[derive(Clone, Copy, Aggregate)]
pub struct Proposal {
    pub sample: PrimarySample,
    pub log_pdf: Expr<f32>,
}
pub trait Mutation {
    // fn weight(&self) -> f32;
    fn log_pdf(&self, cur: &PrimarySample, proposal: &PrimarySample) -> Expr<f32>;
    fn mutate(&self, s: &PrimarySample, sampler: &IndependentSampler) -> Proposal;
}
pub struct AnisotropicDiagonalGaussianMutation {
    pub sigma: VLArrayExpr<f32>,
}
impl Mutation for AnisotropicDiagonalGaussianMutation {
    fn log_pdf(&self, cur: &PrimarySample, proposal: &PrimarySample) -> Expr<f32> {
        assert_eq!(cur.values.static_len(), proposal.values.static_len());
        let log_prob = var!(f32, 0.0);
        for_range(const_(0)..cur.values.len().int(), |i| {
            let i = i.uint();
            let cur = cur.values.read(i);
            let proposal = proposal.values.read(i);
            let x = proposal - cur;
            log_prob.store(log_prob.load() + log_gaussian_pdf(x, self.sigma.read(i)));
        });
        log_prob.load()
    }
    fn mutate(&self, s: &PrimarySample, sampler: &IndependentSampler) -> Proposal {
        let values = VLArrayVar::<f32>::zero(s.values.static_len());
        for_range(const_(0)..values.len().int(), |i| {
            let i = i.uint();
            let cur = s.values.read(i);
            let x = sample_gaussian(sampler.next_1d());
            let new = cur + x * self.sigma.read(i);
            values.write(i, new);
        });
        let proposal = PrimarySample { values };
        let log_pdf = self.log_pdf(&s, &proposal);
        Proposal {
            sample: proposal,
            log_pdf,
        }
    }
}

pub struct IsotropicGaussianMutation {
    pub sigma: Expr<f32>,
}
impl Mutation for IsotropicGaussianMutation {
    fn log_pdf(&self, cur: &PrimarySample, proposal: &PrimarySample) -> Expr<f32> {
        assert_eq!(cur.values.static_len(), proposal.values.static_len());
        let log_prob = var!(f32, 0.0);
        for_range(const_(0)..cur.values.len().int(), |i| {
            let i = i.uint();
            let cur = cur.values.read(i);
            let proposal = proposal.values.read(i);
            let x = proposal - cur;
            log_prob.store(log_prob.load() + log_gaussian_pdf(x, self.sigma));
        });
        log_prob.load()
    }
    fn mutate(&self, s: &PrimarySample, sampler: &IndependentSampler) -> Proposal {
        let values = VLArrayVar::<f32>::zero(s.values.static_len());
        for_range(const_(0)..values.len().int(), |i| {
            let i = i.uint();
            let cur = s.values.read(i);
            let x = sample_gaussian(sampler.next_1d());
            let new = cur + x * self.sigma;
            values.write(i, new);
        });
        let proposal = PrimarySample { values };
        let log_pdf = self.log_pdf(&s, &proposal);
        Proposal {
            sample: proposal,
            log_pdf,
        }
    }
}
pub struct LargeStepMutation {}
impl Mutation for LargeStepMutation {
    fn log_pdf(&self, _cur: &PrimarySample, _proposal: &PrimarySample) -> Expr<f32> {
        const_(0.0f32)
    }
    fn mutate(&self, s: &PrimarySample, sampler: &IndependentSampler) -> Proposal {
        let values = VLArrayVar::<f32>::zero(s.values.static_len());
        for_range(const_(0)..values.len().int(), |i| {
            let i = i.uint();
            values.write(i, sampler.next_1d());
        });
        Proposal {
            sample: PrimarySample { values },
            log_pdf: const_(0.0f32),
        }
    }
}

pub fn acceptance_prob(
    log_cur_f: Expr<f32>,
    log_cur_pdf: Expr<f32>,
    log_proposal_f: Expr<f32>,
    log_proposal_pdf: Expr<f32>,
) -> Expr<f32> {
    (log_proposal_f - log_cur_f + log_proposal_pdf - log_cur_pdf)
        .exp()
        .clamp(0.0, 1.0)
}
