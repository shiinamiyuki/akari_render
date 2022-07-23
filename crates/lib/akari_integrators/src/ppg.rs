use std::sync::atomic::AtomicU64;

use crate::*;
use akari_common::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct QTreeNode {
    pub(crate) sum: [AtomicFloat; 4],
    pub(crate) children: [Option<u32>; 4],
}
impl QTreeNode {
    fn new() -> Self {
        Self {
            sum: [
                AtomicFloat::new(0.25),
                AtomicFloat::new(0.25),
                AtomicFloat::new(0.25),
                AtomicFloat::new(0.25),
            ],
            children: [None; 4],
        }
    }
}
fn determine_child(p: &Vec2) -> usize {
    let x = p.x > 0.5;
    let y = p.y > 0.5;
    (x as usize) + (y as usize) * 2
}
fn remap(x: f32) -> f32 {
    (if x > 0.5 { (x - 0.5) * 2.0 } else { x * 2.0 }).clamp(0.0, 1.0)
}
fn remap2(u: Vec2) -> Vec2 {
    vec2(remap(u.x), remap(u.y))
}
// fn unmap(u: &Vec2, p: Vec2) -> Vec2 {
//     fn f(x: f32, p: f32) -> f32 {
//         (if x > 0.5 { p * 0.5 + 0.5 } else { p * 0.5 }).clamp(0.0, 1.0)
//     }
//     vec2(f(u.x, p.x), f(u.y, p.y))
// }
#[derive(Clone, Serialize, Deserialize)]
pub struct QTree {
    pub(crate) nodes: Vec<QTreeNode>,
}

impl QTree {
    fn sample_recursive(&self, mut u: Vec2, idx: usize) -> (Vec2, f32) {
        let node = &self.nodes[idx];
        let sum0 = node.sum[0].load(Ordering::Relaxed) + node.sum[1].load(Ordering::Relaxed);
        let sum1 = node.sum[2].load(Ordering::Relaxed) + node.sum[3].load(Ordering::Relaxed);
        let total = sum0 + sum1;
        let (child_idx, pdf) = if total == 0.0 {
            let i = determine_child(&u);
            u = remap2(u);
            (i, 1.0)
        } else {
            let x;
            let y;
            let sum;
            let mut pdf = 1.0;
            if u.y < sum0 / total {
                y = 0;
                u.y *= total / sum0;
                sum = sum0;
                pdf *= sum0 / total;
            } else {
                y = 1;
                u.y = (u.y - sum0 / total) * total / sum1;
                sum = sum1;
                pdf *= sum1 / total;
            };
            let total = sum;
            let sum0 = node.sum[0 + 2 * y as usize].load(Ordering::Relaxed);
            let sum1 = node.sum[1 + 2 * y as usize].load(Ordering::Relaxed);
            if u.x < sum0 / total {
                x = 0;
                u.x *= total / sum0;
                pdf *= sum0 / total;
            } else {
                x = 1;
                u.x = (u.x - sum0 / total) * total / sum1;
                pdf *= sum1 / total;
            };
            ((x + 2 * y) as usize, pdf)
        };

        if let Some(child) = node.children[child_idx] {
            let (p, pdf1) = self.sample_recursive(u, child as usize);
            let x = child_idx & 1;
            let y = child_idx >> 1;
            (
                vec2(p.x * 0.5 + x as f32 * 0.5, p.y * 0.5 + y as f32 * 0.5),
                pdf1 * pdf,
            )
        } else {
            (u, pdf)
        }
    }
    pub fn sample(&self, u: &Vec2) -> (Vec2, f32) {
        self.sample_recursive(*u, 0)
    }
    fn pdf_recursive(&self, d: &Vec2, idx: usize) -> f32 {
        let node = &self.nodes[idx];
        let sum0 = node.sum[0].load(Ordering::Relaxed) + node.sum[1].load(Ordering::Relaxed);
        let sum1 = node.sum[2].load(Ordering::Relaxed) + node.sum[3].load(Ordering::Relaxed);
        let total = sum0 + sum1;
        let child_idx = determine_child(d);
        node.sum[child_idx].load(Ordering::Relaxed) / total
            * if let Some(child) = node.children[child_idx] {
                self.pdf_recursive(&remap2(*d), child as usize)
            } else {
                1.0
            }
    }
    fn pdf(&self, d: &Vec2) -> f32 {
        self.pdf_recursive(d, 0)
    }
    fn compute_sum_recursive(&mut self, idx: usize) -> f32 {
        let mut sum = 0.0;
        for i in 0..4 {
            let node = &mut self.nodes[idx];
            if let Some(child) = node.children[idx] {
                std::mem::drop(node);
                let s = self.compute_sum_recursive(child as usize);
                let node = &mut self.nodes[idx];
                node.sum[idx].store(s, Ordering::Relaxed);
                sum += s;
            } else {
                sum += node.sum[i].load(Ordering::Relaxed);
            }
        }
        sum
    }
    fn compute_sum(&mut self) -> f32 {
        self.compute_sum_recursive(0)
    }
    fn sum(&self, idx: usize) -> f32 {
        self.nodes[idx]
            .sum
            .iter()
            .map(|x| x.load(Ordering::Relaxed))
            .sum()
    }
    fn subdivide_leaf(&self, fraction: f32, other: &mut QTree, other_idx: usize) {
        let total: f32 = self.sum(0);
        for i in 0..4 {
            if other.nodes[other_idx].sum[i].load(Ordering::Relaxed) / total > fraction {
                let new_node = other.nodes.len() as usize;
                let node = QTreeNode::new();
                for i in 0..4 {
                    node.sum[i].store(
                        other.nodes[other_idx].sum[i].load(Ordering::Relaxed) * 0.25,
                        Ordering::Relaxed,
                    );
                }
                other.nodes.push(node);
                {
                    let other_node = &mut other.nodes[other_idx];
                    other_node.children[i] = Some(new_node as u32);
                }
                self.subdivide_leaf(fraction, other, new_node)
            }
        }
    }
    fn subdivide_recursive(&self, fraction: f32, idx: usize, other: &mut QTree, other_idx: usize) {
        let total: f32 = self.sum(0);
        {
            let other_node = &mut other.nodes[other_idx];
            for i in 0..4 {
                other_node.sum[i] = self.nodes[idx].sum[i].clone();
            }
        }
        for i in 0..4 {
            let sum = self.nodes[other_idx].sum[i].load(Ordering::Relaxed);
            if sum / total > fraction {
                if let Some(child) = self.nodes[idx].children[i] {
                    let new_node = other.nodes.len() as usize;
                    other.nodes.push(QTreeNode::new());
                    self.subdivide_recursive(fraction, child as usize, other, new_node);
                } else {
                    let new_node = other.nodes.len() as usize;
                    other.nodes.push(QTreeNode::new());
                    for i in 0..4 {
                        other.nodes[new_node].sum[i] = AtomicFloat::new(sum * 0.25);
                    }
                    self.subdivide_leaf(fraction, other, new_node);
                }
            }
        }
    }
    pub fn subdivide(&mut self, fraction: f32) -> Self {
        let _ = self.compute_sum();
        let mut other = QTree {
            nodes: vec![QTreeNode::new()],
        };
        self.subdivide_recursive(fraction, 0, &mut other, 0);
        other
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct DTree {
    pub(crate) sampling: QTree,
    pub(crate) building: QTree,
}

impl DTree {
    pub fn sample(&self, u: &Vec2) -> (Vec3A, f32) {
        let (w, pdf) = self.sampling.sample(u);
        (cylindrical_to_xyz(&w), pdf)
    }
    pub fn pdf(&self, w: &Vec3A) -> f32 {
        self.sampling.pdf(&xyz_to_cylindrical(w))
    }
}
#[derive(Serialize, Deserialize)]
pub struct STreeNode {
    pub(crate) dtree: DTree,
    pub(crate) axis: usize,
    pub(crate) n_samples: AtomicU64,
    pub(crate) children: [Option<usize>; 2],
}
fn xyz_to_cylindrical(xyz: &Vec3A) -> Vec2 {
    let mut phi = xyz.z.atan2(xyz.x);
    while phi < 0.0 {
        phi += 2.0 * PI;
    }
    vec2(phi, (xyz.y.clamp(-1.0, 1.0) + 1.0) / 2.0)
}
fn cylindrical_to_xyz(d: &Vec2) -> Vec3A {
    let phi = d.x * 2.0 * PI;
    let c = 2.0 * d.y - 1.0;
    let s = (1.0 - c * c).sqrt();
    vec3a(s * phi.cos(), c, s * phi.sin())
}
impl STreeNode {}
#[derive(Serialize, Deserialize)]
pub struct STree {
    nodes: STreeNode,
}
