use crate::{
    sampler::*,
    sampling::{log_gaussian_pdf, sample_gaussian},
};
use lazy_static::lazy_static;

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
    pub drift: VLArrayExpr<f32>,
}
impl Mutation for AnisotropicDiagonalGaussianMutation {
    fn log_pdf(&self, cur: &PrimarySample, proposal: &PrimarySample) -> Expr<f32> {
        assert_eq!(cur.values.static_len(), proposal.values.static_len());
        let log_prob = var!(f32, 0.0);
        for_range(const_(0)..cur.values.len().int(), |i| {
            let i = i.uint();
            let cur = cur.values.read(i);
            let proposal = proposal.values.read(i);
            let x = proposal - cur - self.drift.read(i);
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
            let new = cur + self.drift.read(i) + x * self.sigma.read(i);
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
pub struct IsotropicExponentialMutation {
    mutation_size_low: Expr<f32>,
    mutation_size_high: Expr<f32>,
    log_ratio: Expr<f32>,
    compute_log_pdf: bool,
    image_mutation_size: Option<f32>,
    res: Expr<Float2>,
    image_mutation: Expr<bool>,
}
impl IsotropicExponentialMutation {
    pub fn new_default(
        compute_log_pdf: bool,
        image_mutation: Expr<bool>,
        image_mutation_size: Option<f32>,
        res: Expr<Float2>,
    ) -> Self {
        Self::new(
            const_(1.0 / 1024.0f32),
            const_(1.0 / 64.0f32),
            compute_log_pdf,
            image_mutation,
            image_mutation_size,
            res,
        )
    }
    pub fn new(
        mutation_size_low: Expr<f32>,
        mutation_size_high: Expr<f32>,
        compute_log_pdf: bool,
        image_mutation: Expr<bool>,
        image_mutation_size: Option<f32>,
        res: Expr<Float2>,
    ) -> Self {
        let log_ratio = -(mutation_size_high / mutation_size_low).ln();
        Self {
            mutation_size_low,
            mutation_size_high,
            log_ratio,
            compute_log_pdf,
            image_mutation_size,
            res,
            image_mutation,
        }
    }
}
#[derive(Clone, Copy, Value, Debug)]
#[repr(C)]
pub struct KelemenMutationRecord {
    pub cur: f32,
    pub u: f32,
    pub mutation_size_low: f32,
    pub mutation_size_high: f32,
    pub log_ratio: f32,
    pub mutated: f32,
}
lazy_static! {
    pub static ref KELEMEN_MUTATE: Callable<(Var<KelemenMutationRecord>,), ()> =
        create_static_callable::<(Var<KelemenMutationRecord>,), ()>(
            |record: Var<KelemenMutationRecord>| {
                let cur = record.cur().load();
                let u = record.u().load();
                let (u, add) = if_!(u.cmplt(0.5), {
                    (u * 2.0, const_(true))
                }, else {
                    ((u - 0.5) * 2.0, const_(false))
                });
                let dv = record.mutation_size_high().load() * (record.log_ratio().load() * u).exp();
                let new = if_!(add, {
                    let new = cur + dv;
                    select(new.cmpgt(1.0), new - 1.0, new)
                }, else {
                    let new = cur - dv;
                    select(new.cmplt(0.0), new + 1.0, new)
                });
                record.set_mutated(new);
            }
        );
}
impl Mutation for IsotropicExponentialMutation {
    fn log_pdf(&self, _cur: &PrimarySample, _proposal: &PrimarySample) -> Expr<f32> {
        todo!()
    }
    fn mutate(&self, s: &PrimarySample, sampler: &IndependentSampler) -> Proposal {
        let values = VLArrayVar::<f32>::zero(s.values.static_len());

        let begin = if self.image_mutation_size.is_some() {
            let image_mutation_size = self.image_mutation_size.unwrap();
            mutate_image_space(s, &values, sampler, const_(image_mutation_size), self.res);
            2
        } else {
            0
        };
        for_range(const_(begin)..values.len().int(), |i| {
            let i = i.uint();
            let cur = s.values.read(i);
            let new = if_!(!self.image_mutation | i.cmplt(2), {
                let u = sampler.next_1d();
                let record = var!(
                    KelemenMutationRecord,
                    KelemenMutationRecordExpr::new(
                        cur,
                        u,
                        self.mutation_size_low,
                        self.mutation_size_high,
                        self.log_ratio,
                        0.0
                    )
                );
                KELEMEN_MUTATE.call(record);
                record.mutated().load()
            }, else {
                cur
            });
            values.write(i, new);
        });
        let proposal = PrimarySample { values };
        let log_pdf = if self.compute_log_pdf {
            self.log_pdf(&s, &proposal)
        } else {
            const_(0.0f32)
        };
        Proposal {
            sample: proposal,
            log_pdf,
        }
    }
}
pub struct IsotropicGaussianMutation {
    pub sigma: Expr<f32>,
    pub compute_log_pdf: bool,
    pub image_mutation_size: Option<f32>,
    pub res: Expr<Float2>,
    pub image_mutation: Expr<bool>,
}

// mutates the image space coordinate within range [0, mutation_size]
fn mutate_image_space(
    s: &PrimarySample,
    values: &VLArrayVar<f32>,
    sampler: &IndependentSampler,
    mutation_size: Expr<f32>,
    res: Expr<Float2>,
) {
    for i in 0..2 {
        let cur = s.values.read(i);
        let u = sampler.next_1d();
        let (u, add) = if_!(u.cmplt(0.5), {
            (u * 2.0, const_(true))
        }, else {
            ((u - 0.5) * 2.0, const_(false))
        });
        let offset = u * mutation_size;
        let offset = select(add, offset, -offset);
        let new = cur + offset / res.at(i as usize);
        values.write(i, new - new.floor());
    }
}
impl Mutation for IsotropicGaussianMutation {
    fn log_pdf(&self, cur: &PrimarySample, proposal: &PrimarySample) -> Expr<f32> {
        // assert_eq!(cur.values.static_len(), proposal.values.static_len());
        // let log_prob = var!(f32, 0.0);
        // for_range(const_(0)..cur.values.len().int(), |i| {
        //     let i = i.uint();
        //     let cur = cur.values.read(i);
        //     let proposal = proposal.values.read(i);
        //     let x = proposal - cur;
        //     log_prob.store(log_prob.load() + log_gaussian_pdf(x, self.sigma));
        // });
        // log_prob.load()
        todo!()
    }
    fn mutate(&self, s: &PrimarySample, sampler: &IndependentSampler) -> Proposal {
        let values = VLArrayVar::<f32>::zero(s.values.static_len());
        let begin = if self.image_mutation_size.is_some() {
            let image_mutation_size = self.image_mutation_size.unwrap();
            mutate_image_space(s, &values, sampler, const_(image_mutation_size), self.res);
            2
        } else {
            0
        };

        for_range(const_(begin)..s.values.len().int(), |i| {
            let i = i.uint();
            let cur = s.values.read(i);
            let new = if_!(!self.image_mutation | i.cmplt(2), {
                let x = sample_gaussian(sampler.next_1d());
                cur + x * self.sigma
            }, else {
                cur
            });
            values.write(i, new);
        });

        let proposal = PrimarySample { values };
        let log_pdf = if self.compute_log_pdf {
            self.log_pdf(&s, &proposal)
        } else {
            const_(0.0f32)
        };
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
