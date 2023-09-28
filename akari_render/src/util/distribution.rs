use std::collections::VecDeque;

use rand::Rng;

use crate::*;

use crate::sampling::{uniform_discrete_choice_and_remap, weighted_discrete_choice2_and_remap};
#[derive(Clone, Copy, Debug, Default, Value)]
#[repr(C)]
pub struct AliasTableEntry {
    pub j: u32,
    pub t: f32,
}

pub struct AliasTable(pub Buffer<AliasTableEntry>, pub Buffer<f32>);
pub struct BindlessAliasTableVar(
    pub BindlessBufferVar<AliasTableEntry>,
    pub BindlessBufferVar<f32>,
);
impl BindlessAliasTableVar {
    pub fn pdf(&self, i: Expr<u32>) -> Expr<f32> {
        self.1.read(i)
    }
    pub fn sample_and_remap(&self, u: Expr<f32>) -> (Expr<u32>, Expr<f32>, Expr<f32>) {
        let (idx, u) = uniform_discrete_choice_and_remap(self.0.len_expr().as_u32(), u);
        let entry = self.0.read(idx);
        let (idx, u) = weighted_discrete_choice2_and_remap(entry.t, idx, entry.j, u);
        let pdf = self.1.read(idx);
        (idx, pdf, u)
    }
}
impl AliasTable {
    pub fn new(device: Device, weights: &[f32]) -> Self {
        assert!(weights.len() >= 1);
        let sum: f32 = weights.iter().map(|x| *x).sum::<f32>();
        let mut prob: Vec<_> = weights
            .iter()
            .map(|x| *x / sum * (weights.len() as f32))
            .collect();
        let mut small = VecDeque::new();
        let mut large = VecDeque::new();
        for (i, p) in prob.iter().enumerate() {
            if *p >= 1.0 {
                large.push_back(i);
            } else {
                small.push_back(i);
            }
        }
        let mut table = vec![AliasTableEntry::default(); weights.len()];
        while !small.is_empty() && !large.is_empty() {
            let l = small.pop_front().unwrap();
            let g = large.pop_front().unwrap();
            table[l].t = prob[l];
            table[l].j = g as u32;
            prob[g] = (prob[g] + prob[l]) - 1.0;
            if prob[g] < 1.0 {
                small.push_back(g);
            } else {
                large.push_back(g);
            }
        }
        while !large.is_empty() {
            let g = large.pop_front().unwrap();
            table[g].t = 1.0;
            table[g].j = g as u32;
        }
        while !small.is_empty() {
            let l = small.pop_front().unwrap();
            table[l].t = 1.0;
            table[l].j = l as u32;
        }
        Self(
            device.create_buffer_from_slice(&table),
            device.create_buffer_from_fn(prob.len(), |i| weights[i] / sum),
        )
    }
    pub fn pdf(&self, i: Expr<u32>) -> Expr<f32> {
        self.1.var().read(i)
    }
    pub fn sample_and_remap(&self, u: Expr<f32>) -> (Expr<u32>, Expr<f32>, Expr<f32>) {
        // let idx = (u.x * self.0.var().len().cast_f32()).cast_u32();
        // let idx = idx.min(self.0.var().len() - 1);
        // let entry = self.0.var().read(idx);
        // let idx = select(u.y.ge(entry.t()), entry.j(), idx);
        // let pdf = self.1.var().read(idx);
        // (idx, pdf)
        // lc_assert!((self.0.len() as u32).expr().eq(self.0.var().len_expr().as_u32()));
        // let (idx, u) = uniform_discrete_choice_and_remap((self.0.len() as u32).expr(), u);
        let (idx, u) = uniform_discrete_choice_and_remap(self.0.var().len_expr().as_u32(), u);
        let entry = self.0.var().read(idx);
        let (idx, u) = weighted_discrete_choice2_and_remap(entry.t, idx, entry.j, u);
        let pdf = self.1.var().read(idx);
        (idx, pdf, u)
    }
}

/// let (sum, samples) = resample_on_cpu(&weights, count);
pub fn resample_with_f64(weights: &[f32], count: usize) -> (f64, Vec<u32>) {
    let weights = weights.iter().map(|x| *x as f64).collect::<Vec<_>>();
    let sum = weights.par_iter().sum::<f64>();
    let mut cdf = vec![];
    for i in 0..weights.len() {
        let p = weights[i] / sum;
        if i == 0 {
            cdf.push(p);
        } else {
            cdf.push(cdf[i - 1] + p);
        }
    }

    let mut rng = rand::thread_rng();

    let resampled = (0..count)
        .map(|_| {
            let u = rng.gen::<f64>();
            let i = cdf.partition_point(|x| u >= *x) as u32;
            i.min(weights.len() as u32 - 1)
        })
        .collect();
    (sum, resampled)
}
#[cfg(test)]
mod test {

    use std::env::current_exe;

    use rand::{thread_rng, Rng};

    use super::*;

    #[test]
    fn alias_table() {
        let ctx = luisa::Context::new(current_exe().unwrap());
        let mut rng = thread_rng();
        let mut weights = (0..100).map(|_| rng.gen::<f32>()).collect::<Vec<_>>();
        let sum = weights.iter().sum::<f32>();
        weights.iter_mut().for_each(|x| *x /= sum);
        let device = ctx.create_cpu_device();
        let table = AliasTable::new(device.clone(), &weights);
        let entries = table.0.copy_to_vec();
        let mut h = vec![0.0f32; weights.len()];
        for (i, e) in entries.iter().enumerate() {
            let t = e.t;
            let j = e.j as usize;
            h[i] += t;
            h[j] += 1.0 - t;
        }
        h.iter_mut().for_each(|x| *x /= weights.len() as f32);
        for (a, b) in h.iter().zip(weights.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
    }
}
