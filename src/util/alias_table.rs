use std::collections::VecDeque;

use crate::*;
#[derive(Clone, Copy, Debug, Default, Value)]
#[repr(C)]
pub struct AliasTableEntry {
    pub j: u32,
    pub t: f32,
}

pub struct AliasTable(Buffer<AliasTableEntry>, Buffer<f32>);

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
            device.create_buffer_from_slice(&table).unwrap(),
            device
                .create_buffer_from_fn(prob.len(), |i| weights[i] / sum)
                .unwrap(),
        )
    }
    pub fn pdf(&self, i: Uint32) -> Float32 {
        self.1.var().read(i)
    }
    pub fn sample(&self, u: Expr<Vec2>) -> (Uint32, Float32) {
        let idx = (u.x() * self.0.len() as f32).uint();
        let idx = idx.min(self.0.len() as u32 - 1);
        let entry = self.0.var().read(idx);
        let idx = select(u.y().cmpge(entry.t()), entry.j(), idx);
        let pdf = self.1.var().read(idx);
        (idx, pdf)
    }
}