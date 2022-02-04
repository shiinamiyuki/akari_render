use std::{convert::TryInto, fmt};

use glam::{BVec3, BVec3A, BVec4, BVec4A};

use super::{
    bvh::{BvhAccel, BvhData, BvhNode},
    Accel, TopLevelBvhData,
};
use crate::shape::Shape;
use crate::shape::SurfaceInteraction;
use crate::texture::ShadingPoint;
use crate::*;
use std::sync::Arc;
const INVALID_CHILD: u8 = u8::MAX;
#[derive(Clone, Copy)]
struct QBvhNode {
    min: [Vec4; 3],
    max: [Vec4; 3],
    children: UVec4,
    count: [u8; 4],
    // axes: [u32; 3],
}
impl Default for QBvhNode {
    fn default() -> Self {
        Self {
            min: [Vec4::ZERO; 3],
            max: [Vec4::ZERO; 3],
            children: UVec4::ZERO,
            count: [INVALID_CHILD; 4],
        }
    }
}
impl std::fmt::Debug for QBvhNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QBvhNode")
            .field("children", &<[u32; 4]>::from(self.children))
            .field("count", &self.count)
            .finish()
    }
}
mod test {
    #[test]
    fn test_size() {
        use super::*;
        assert_eq!(std::mem::size_of::<QBvhNode>(), 128);
    }
}
#[inline(always)]
fn bool4_to_u32(x: [bool; 4]) -> u32 {
    (x[0] as u32) | ((x[1] as u32) << 1) | ((x[2] as u32) << 2) | ((x[3] as u32) << 3)
}
impl QBvhNode {
    // #[inline(always)]
    // fn prim_start(children: UVec4) -> UVec4 {
    //     Self::child_indices(children)
    // }
    #[inline]
    fn leaf_mask(&self) -> u32 {
        bool4_to_u32([
            self.count[0] != 0 && self.count[0] != INVALID_CHILD,
            self.count[1] != 0 && self.count[1] != INVALID_CHILD,
            self.count[2] != 0 && self.count[2] != INVALID_CHILD,
            self.count[3] != 0 && self.count[3] != INVALID_CHILD,
        ])
    }
    #[inline]
    fn children_mask(&self) -> u32 {
        bool4_to_u32([
            self.count[0] == 0,
            self.count[1] == 0,
            self.count[2] == 0,
            self.count[3] == 0,
        ])
    }
    fn intersect(
        &self,
        ray: &Ray,
        o: [Vec4; 3],
        invd: [Vec4; 3],
        // sign: [BVec4A; 3],
    ) -> (BVec4A, UVec4) {
        let t10 = (self.min[0] - o[0]) * invd[0];
        let t11 = (self.min[1] - o[1]) * invd[1];
        let t12 = (self.min[2] - o[2]) * invd[2];
        let t00 = (self.max[0] - o[0]) * invd[0];
        let t01 = (self.max[1] - o[1]) * invd[1];
        let t02 = (self.max[2] - o[2]) * invd[2];

        let tmin0 = t10.min(t00);
        let tmax0 = t10.max(t00);
        let tmin1 = t11.min(t01);
        let tmax1 = t11.max(t01);
        let tmin2 = t12.min(t02);
        let tmax2 = t12.max(t02);

        let tmin = tmin0.max(tmin1).max(tmin2).max(Vec4::splat(ray.tmin));
        let tmax = tmax0.min(tmax1).min(tmax2).min(Vec4::splat(ray.tmax));

        // (tmin.cmple(tmax), tmin)
        let mask = tmin.cmple(tmax);

        let mut t: [f32; 4] = tmin.into();
        let mut indices: [u32; 4] = [0, 1, 2, 3];
        macro_rules! cmp_swap {
            ($a:expr, $b:expr) => {
                if t[$a] < t[$b] {
                    t.swap($a, $b);
                    indices.swap($a, $b);
                }
            };
        }
        cmp_swap!(0, 1);
        cmp_swap!(2, 3);
        cmp_swap!(0, 2);
        cmp_swap!(1, 3);
        cmp_swap!(2, 3);
        (mask, UVec4::from(indices))
        // let  = tmin0.min(tmax0);
        // let t0 = (aabb.min - o) * invd;
        // let t1 = (aabb.max - o) * invd;
        // let min = t0.min(t1);
        // let max = t0.max(t1);
        // let tmin = min.max_element().max(ray.tmin);
        // let tmax = max.min_element().min(ray.tmax);
        // if tmin <= tmax {
        //     tmin
        // } else {
        //     -1.0
        // }
    }
}

pub struct QBvhAccel<T: BvhData> {
    pub(crate) data: T,
    nodes: Vec<QBvhNode>,
    references: Vec<u32>,
    pub(crate) aabb: Aabb,
}

impl<T: BvhData> QBvhAccel<T> {
    #[allow(unused_mut)]
    pub fn traverse4<F: FnMut(&mut Ray4, BVec4A, u32) -> BVec4A>(
        &self,
        _ray4: &mut Ray4,
        mut _active_mask: BVec4A,
        mut _f: F,
    ) {
        todo!()
    }

    pub fn traverse<F: FnMut(&mut Ray, u32) -> bool>(&self, mut ray: Ray, mut f: F) {
        unsafe {
            let inv_d = Vec3::ONE / ray.d;
            let inv_d = [
                Vec4::splat(inv_d.x),
                Vec4::splat(inv_d.y),
                Vec4::splat(inv_d.z),
            ];
            let o = [
                Vec4::splat(ray.o.x),
                Vec4::splat(ray.o.y),
                Vec4::splat(ray.o.z),
            ];
            let mut stack: [u32; 32] = [0; 32];
            let mut sp = 1;
            while sp > 0 {
                sp -= 1;
                let node = self.nodes.get_unchecked(stack[sp] as usize);

                let (mask, indices) = node.intersect(&ray, o, inv_d);
                let leaf_mask = node.leaf_mask();
                let children = node.children;
                if mask.any() {
                    let mask = mask.bitmask();
                    let hit_leaves = leaf_mask & mask;
                    let hit_children = node.children_mask() & mask;
                    for j in 0..4 {
                        let i = indices[j];
                        if 0 != (hit_leaves & (1 << i)) {
                            let start = children[i as usize] as usize;
                            let count = node.count[i as usize] as usize;
                            for p in start..(start + count) {
                                if !f(&mut ray, *self.references.get_unchecked(p)) {
                                    return;
                                }
                            }
                        } else {
                            if 0 != (hit_children & (1 << i)) {
                                stack[sp] = children[i as usize];
                                sp += (0 != (hit_children & (1 << i))) as usize;
                            }
                        }
                    }
                }
            }
        }
    }
    // pub fn new()
}

pub struct QBvhAccelBuilder<T: BvhData> {
    data: T,
    references: Vec<u32>,
    bvh_nodes: Vec<BvhNode>,
    qbvh_nodes: Vec<QBvhNode>,
    aabb: Aabb,
}
#[derive(Clone, Copy)]
enum BvhNodeChild {
    Child {
        aabb: Aabb,
        idx: usize,
    },
    Leaf {
        aabb: Aabb,
        first: usize,
        count: usize,
    },
    Empty,
}
impl<T: BvhData> QBvhAccelBuilder<T> {
    fn collapse2(&self, bvh_node: usize) -> [BvhNodeChild; 2] {
        let node = &self.bvh_nodes[bvh_node];
        // assert!(!node.is_leaf(), "{} {}", bvh_node, self.bvh_nodes.len());
        if node.is_leaf() {
            return [
                BvhNodeChild::Leaf {
                    aabb: node.aabb,
                    first: node.left_or_first_primitive as usize,
                    count: node.count as usize,
                },
                BvhNodeChild::Empty,
            ];
        }
        let left = node.left() as usize;
        let right = node.right() as usize;
        let f = |i: usize| {
            let node = &self.bvh_nodes[i];
            if node.is_leaf() {
                BvhNodeChild::Leaf {
                    aabb: node.aabb,
                    first: node.left_or_first_primitive as usize,
                    count: node.count as usize,
                }
            } else {
                BvhNodeChild::Child {
                    aabb: node.aabb,
                    idx: i,
                }
            }
        };
        [f(left), f(right)]
    }
    fn collapse4(&self, bvh_node: usize) -> [BvhNodeChild; 4] {
        let node = &self.bvh_nodes[bvh_node];
        assert!(!node.is_leaf(), "{} {}", bvh_node, self.bvh_nodes.len());
        let left = self.collapse2(node.left() as usize);
        let right = self.collapse2(node.right() as usize);
        [left[0], left[1], right[0], right[1]]
    }
    fn create_qbvh_node(&self, bvh_node: usize) -> QBvhNode {
        let children = self.collapse4(bvh_node);
        let mut min = [Vec4::ZERO; 3];
        let mut max = [Vec4::ZERO; 3];
        for j in 0..3 {
            for i in 0..4 {
                min[j][i] = match children[i] {
                    BvhNodeChild::Child { aabb, .. } => aabb.min[j],
                    BvhNodeChild::Leaf { aabb, .. } => aabb.min[j],
                    BvhNodeChild::Empty => 0.0,
                };
                max[j][i] = match children[i] {
                    BvhNodeChild::Child { aabb, .. } => aabb.max[j],
                    BvhNodeChild::Leaf { aabb, .. } => aabb.max[j],
                    BvhNodeChild::Empty => 0.0,
                };
            }
        }
        let mut count = [0; 4];
        for i in 0..4 {
            count[i] = match children[i] {
                BvhNodeChild::Leaf { count, .. } => count.try_into().unwrap(),
                BvhNodeChild::Empty => INVALID_CHILD,
                _ => 0,
            };
        }
        QBvhNode {
            min,
            max,
            count,
            children: UVec4::ZERO,
        }
    }
    pub fn recursive_build(&mut self, bvh_node: usize, qbvh_idx: usize) {
        let children = self.collapse4(bvh_node);
        let n_children = children
            .iter()
            .map(|c| -> usize {
                {
                    match c {
                        BvhNodeChild::Child { .. } => 1,
                        _ => 0,
                    }
                }
            })
            .sum::<usize>();
        let qbvh_node = self.create_qbvh_node(bvh_node);
        self.qbvh_nodes[qbvh_idx] = qbvh_node;
        let base = self.qbvh_nodes.len();
        for _ in 0..n_children {
            self.qbvh_nodes.push(Default::default());
        }
        let mut cnt = 0;
        for (i, c) in children.iter().enumerate() {
            match c {
                BvhNodeChild::Child { idx, .. } => {
                    self.qbvh_nodes[qbvh_idx].children[i] = (base + cnt) as u32;
                    self.recursive_build(*idx, base + cnt);
                    cnt += 1;
                }
                BvhNodeChild::Leaf { first, .. } => {
                    self.qbvh_nodes[qbvh_idx].children[i] = *first as u32;
                }
                _ => {}
            }
        }
    }
    pub fn build(mut self) -> QBvhAccel<T> {
        if self.bvh_nodes[0].is_leaf() {
            let root = self.bvh_nodes[0];
            self.qbvh_nodes.push(QBvhNode {
                min: [
                    Vec4::splat(root.aabb.min[0]),
                    Vec4::splat(root.aabb.min[1]),
                    Vec4::splat(root.aabb.min[2]),
                ],
                max: [
                    Vec4::splat(root.aabb.max[0]),
                    Vec4::splat(root.aabb.max[1]),
                    Vec4::splat(root.aabb.max[2]),
                ],
                children: UVec4::ZERO,
                count: [root.count, INVALID_CHILD, INVALID_CHILD, INVALID_CHILD],
            })
        } else {
            self.qbvh_nodes.push(Default::default());
            self.recursive_build(0, 0);
        }
        log::info!(
            "QBVH: {} refs {} BVH nodes -> {} QBVH nodes",
            self.references.len(),
            self.bvh_nodes.len(),
            self.qbvh_nodes.len()
        );
        // println!("{:?}", self.qbvh_nodes);
        let references = std::mem::replace(&mut self.references, vec![]);
        let nodes = std::mem::replace(&mut self.qbvh_nodes, vec![]);
        let aabb = self.aabb;
        QBvhAccel {
            data: self.data,
            references,
            nodes,
            aabb,
        }
    }
    pub fn new(bvh: BvhAccel<T>) -> Self {
        Self {
            aabb: bvh.aabb,
            data: bvh.data,
            references: bvh.references,
            bvh_nodes: bvh.nodes,
            qbvh_nodes: vec![],
        }
    }
}

impl_bvh_accel!(QBvhAccel<TopLevelBvhData>);
