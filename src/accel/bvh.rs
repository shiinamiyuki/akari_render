use crate::bsdf::*;
use crate::shape::*;
use crate::*;
use ordered_float::OrderedFloat;
use std::sync::Mutex;

use super::*;
#[derive(Clone, Copy, Debug, Default)]
pub struct BVHNode {
    // pub axis: u8,
    pub aabb: Bounds3f,
    pub left_or_first_primitive: u32,
    pub count: u32,
}
impl BVHNode {
    pub fn is_leaf(&self) -> bool {
        self.count > 0
    }
    pub fn left(&self) -> u32 {
        self.left_or_first_primitive
    }
    pub fn right(&self) -> u32 {
        self.left_or_first_primitive + 1
    }
}
pub trait BVHData {
    fn intersect<'a>(&'a self, idx: u32, ray: &Ray) -> Option<Intersection<'a>>;
    fn occlude(&self, idx: u32, ray: &Ray) -> bool;
    fn bsdf<'a>(&'a self, idx: u32) -> Option<&'a dyn Bsdf>;
    fn aabb(&self, idx: u32) -> Bounds3f;
}
pub struct BVHAccelerator<T: BVHData> {
    pub data: T,
    pub references: Vec<u32>,
    pub nodes: Vec<BVHNode>,
    pub aabb: Bounds3f,
}

pub struct SweepSAHBuilder<T: BVHData> {
    pub data: T,
    pub nodes: Mutex<Option<Vec<BVHNode>>>,
    pub aabb: Mutex<Bounds3f>,
}
impl<T> SweepSAHBuilder<T>
where
    T: BVHData + Sync + 'static,
{
    pub fn build(data: T, mut references: Vec<u32>) -> BVHAccelerator<T> {
        let n_prims = references.len();
        let (data, nodes, aabb) = {
            let builder = Self {
                data,
                nodes: Mutex::new(Some(vec![BVHNode::default()])),
                aabb: Mutex::new(Bounds3f::default()),
            };

            builder.recursive_build(0, 0, references.len() as u32, 0, &mut references);
            // bvh.references = references;

            let nodes = {
                let mut nodes = builder.nodes.lock().unwrap();
                nodes.replace(vec![]).unwrap()
            };
            let aabb = *builder.aabb.lock().unwrap();
            (builder.data, nodes, aabb)
        };

        let bvh = BVHAccelerator {
            data,
            nodes,
            references,
            aabb,
        };
        log::info!(
            "built bvh for {} references; nodes: {}",
            n_prims,
            bvh.nodes.len()
        );
        bvh
    }
    fn recursive_build(
        &self,
        node_idx: u32,
        begin: u32,
        end: u32,
        depth: u32,
        references: &mut Vec<u32>,
    ) {
        assert!(end - begin > 0);
        // println!("building {}..{}", begin, end);
        let mut aabb = Bounds3f::default();
        for i in begin..end {
            aabb.insert_box(self.data.aabb(references[i as usize]));
        }
        for i in 0..3 {
            if aabb.size()[i] == 0.0 {
                aabb.max[i] += 0.001;
            }
        }
        if depth == 0 {
            let mut self_aabb = self.aabb.lock().unwrap();
            *self_aabb = aabb;
        }
        if end - begin <= 4 || depth >= 40 {
            if end - begin == 0 {
                panic!("");
            }
            let node = BVHNode {
                // axis: 0,
                aabb,
                left_or_first_primitive: begin,
                count: end - begin,
            };
            let mut nodes = self.nodes.lock().unwrap();
            let nodes = nodes.as_mut().unwrap();
            nodes[node_idx as usize] = node;
        } else {
            // let try_split_with_axis = |axis| {
            macro_rules! try_split_with_axis {
                ($axis:expr) => {{
                    let mut surface_area_fwd = vec![0.0f64; (end - begin) as usize];
                    let mut surface_area_rev = vec![0.0f64; (end - begin) as usize];

                    references[begin as usize..end as usize].par_sort_unstable_by(|a, b| {
                        let box_a = self.data.aabb(*a);
                        let box_b = self.data.aabb(*b);
                        OrderedFloat::<f32>(box_a.centroid()[$axis])
                            .partial_cmp(&OrderedFloat::<f32>(box_b.centroid()[$axis]))
                            .unwrap()
                    });
                    {
                        let mut aabb = Bounds3f::default();
                        for i in begin..end {
                            aabb.insert_box(self.data.aabb(references[i as usize]));
                            surface_area_fwd[(i - begin) as usize] = aabb.surface_area() as f64;
                        }
                    }
                    {
                        let mut aabb = Bounds3f::default();
                        for i in (begin..end).rev() {
                            aabb.insert_box(self.data.aabb(references[i as usize]));
                            surface_area_rev[(i - begin) as usize] = aabb.surface_area() as f64;
                        }
                    }
                    let mut min_cost = f64::INFINITY;
                    let mut split = 0;
                    for i in begin..(end - 1) {
                        let cost = (i - begin + 1) as f64 * surface_area_fwd[(i - begin) as usize]
                            + (end - i - 1) as f64 * surface_area_rev[(i - begin + 1) as usize];
                        if cost < min_cost {
                            split = i;
                            min_cost = cost;
                        }
                    }
                    let left: HashSet<u32> = references[begin as usize..=split as usize]
                        .iter()
                        .map(|x| *x)
                        .collect();
                    ($axis, split as usize, min_cost, left)
                }};
            }

            let best_split = {
                let mut splits = [
                    try_split_with_axis!(0),
                    try_split_with_axis!(1),
                    try_split_with_axis!(2),
                ];
                let best_split = splits.iter().fold((0usize, std::f64::INFINITY), |a, b| {
                    if a.1 < b.2 {
                        a
                    } else {
                        (b.0, b.2)
                    }
                });
                let mut left = HashSet::new();
                // let mut right = vec![];
                std::mem::swap(&mut left, &mut splits[best_split.0].3);
                // std::mem::swap(&mut right, &mut splits[best_split.0].4);
                (
                    splits[best_split.0].0,
                    splits[best_split.0].1,
                    splits[best_split.0].2,
                    left,
                    // right,
                )
            };

            // partition
            {
                let predicate = |idx: u32| best_split.3.contains(&idx);
                let mut first = (|| {
                    for i in begin..end {
                        if !predicate(references[i as usize]) {
                            return i;
                        }
                    }
                    end
                })();
                let mut mid: u32 = (|| {
                    if first == end {
                        return first;
                    }
                    for i in first + 1..end {
                        if predicate(references[i as usize]) {
                            references.swap(first as usize, i as usize);
                            first += 1;
                        }
                    }
                    return first;
                })();
                if mid == begin || mid == end {
                    // println!("{:?} {:?}", costs, buckets);
                    if end - begin > 12 {
                        eprintln!(
                            "cannot split at depth {} with {} references",
                            depth,
                            end - begin
                        );
                    }
                    mid = (end + begin) / 2;
                }
                let p_ref = UnsafePointer::new(references as *mut Vec<u32>);
                let child_idx = {
                    let mut nodes = self.nodes.lock().unwrap();
                    let nodes = nodes.as_mut().unwrap();
                    let dummy = BVHNode::default();
                    let child_idx = nodes.len() as u32;
                    nodes.push(dummy);
                    nodes.push(dummy);
                    nodes[node_idx as usize] = BVHNode {
                        // axis: axis as u8,
                        aabb,
                        left_or_first_primitive: child_idx,
                        count: 0,
                    };
                    child_idx
                };
                assert!(child_idx % 2 == 1);
                // we know the two parts are disjoint

                let num_threads = rayon::current_num_threads();
                if end - begin >= 128 * 1024
                    && depth <= 1 + (num_threads as f64).log2().ceil() as u32
                {
                    rayon::join(
                        || {
                            self.recursive_build(child_idx, begin, mid, depth + 1, unsafe {
                                p_ref.as_mut().unwrap()
                            })
                        },
                        || {
                            self.recursive_build(child_idx + 1, mid, end, depth + 1, unsafe {
                                p_ref.as_mut().unwrap()
                            })
                        },
                    );
                } else {
                    self.recursive_build(child_idx, begin, mid, depth + 1, references);
                    self.recursive_build(child_idx + 1, mid, end, depth + 1, references);
                }
            }
        }
    }
}

impl<T> BVHAccelerator<T>
where
    T: BVHData,
{
    pub fn optimize_layout(mut self) -> Self {
        let nodes = std::mem::replace(&mut self.nodes, vec![]);
        assert!(nodes.len() % 2 == 1);
        let pair_count = (nodes.len() - 1) / 2;
        let mut pair_areas: Vec<_> = (0..pair_count)
            .into_par_iter()
            .map(|i| {
                let mut aabb = nodes[2 * i + 1].aabb;
                aabb.insert_box(nodes[2 * i + 2].aabb);
                (i, OrderedFloat(aabb.surface_area()))
            })
            .collect();
        pair_areas.par_sort_unstable_by(|a, b| b.1.cmp(&a.1));
        let mut map: Vec<_> = vec![usize::MAX; pair_count];
        for (i, p) in pair_areas.iter().enumerate() {
            map[p.0] = i;
        }
        self.nodes = vec![BVHNode::default(); nodes.len()];
        self.nodes[0] = nodes[0];

        for i in 0..pair_count {
            let j = pair_areas[i].0;
            self.nodes[1 + 2 * i] = nodes[1 + 2 * j];
            self.nodes[1 + 2 * i + 1] = nodes[1 + 2 * j + 1];
        }
        for i in 0..self.nodes.len() {
            if !self.nodes[i].is_leaf() {
                self.nodes[i].left_or_first_primitive =
                    map[((self.nodes[i].left_or_first_primitive - 1) / 2) as usize] as u32 * 2 + 1;
            }
        }
        self
    }
    fn intersect_aabb(aabb: &Bounds3f, ray: &Ray, o: Vec3A, invd: Vec3A) -> f32 {
        let t0 = (aabb.min - o) * invd;
        let t1 = (aabb.max - o) * invd;
        let min = t0.min(t1);
        let max = t0.max(t1);
        let tmin = min.max_element().max(ray.tmin);
        let tmax = max.min_element().min(ray.tmax);
        if tmin <= tmax {
            tmin
        } else {
            -1.0
        }
    }
    fn intersect_leaf<'a>(&'a self, node: &BVHNode, ray: &mut Ray) -> Option<Intersection<'a>> {
        let first = node.left_or_first_primitive;
        let last = node.left_or_first_primitive + node.count;
        let mut ret = None;
        for i in first..last {
            if let Some(isct) = self.data.intersect(self.references[i as usize], ray) {
                ray.tmax = isct.t;
                ret = Some(isct);
            }
        }
        ret
    }
    fn occlude_leaf<'a>(&'a self, node: &BVHNode, ray: &mut Ray) -> bool {
        let first = node.left_or_first_primitive;
        let last = node.left_or_first_primitive + node.count;
        for i in first..last {
            if let Some(_) = self.data.intersect(self.references[i as usize], ray) {
                return true;
            }
        }
        false
    }
    pub fn intersect<'a>(&'a self, original_ray: &Ray) -> Option<Intersection<'a>> {
        let mut stack: [u32; 32] = [0; 32];
        let mut sp = 0;
        let mut p = Some(&self.nodes[0]);
        let mut ray = *original_ray;
        let invd: Vec3A = Vec3A::ONE / Vec3A::from(ray.d);
        let o = ray.o.into();
        let mut isct = None;
        if self.nodes[0].is_leaf() {
            return self.intersect_leaf(&self.nodes[0], &mut ray);
        }

        while p.is_some() {
            let node = p.unwrap();
            if node.is_leaf() {
                if let Some(hit) = self.intersect_leaf(node, &mut ray) {
                    isct = Some(hit);
                }
                if sp > 0 {
                    sp -= 1;
                    p = Some(&self.nodes[stack[sp] as usize]);
                } else {
                    p = None;
                }
            } else {
                let left = &self.nodes[node.left() as usize];
                let right = &self.nodes[node.right() as usize];
                let t_left = Self::intersect_aabb(&left.aabb, &ray, o, invd);
                let t_right = Self::intersect_aabb(&right.aabb, &ray, o, invd);
                if t_left < 0.0 && t_right < 0.0 {
                    if sp > 0 {
                        sp -= 1;
                        p = Some(&self.nodes[stack[sp] as usize]);
                    } else {
                        p = None;
                    }
                    continue;
                }
                if t_left < 0.0 {
                    p = Some(right);
                } else if t_right < 0.0 {
                    p = Some(left);
                } else {
                    if t_left < t_right {
                        stack[sp] = node.right();
                        sp += 1;
                        p = Some(left);
                    } else {
                        stack[sp] = node.left();
                        sp += 1;
                        p = Some(right);
                    }
                }
            }
        }
        isct
    }
    pub fn occlude(&self, original_ray: &Ray) -> bool {
        let mut stack: [u32; 32] = [0; 32];
        let mut sp = 0;
        let mut p = Some(&self.nodes[0]);
        let mut ray = *original_ray;
        let invd: Vec3A = Vec3A::ONE / Vec3A::from(ray.d);
        let o = ray.o.into();
        while p.is_some() {
            let node = p.unwrap();
            let t = Self::intersect_aabb(&node.aabb, &ray, o, invd);
            if t < 0.0 {
                if sp > 0 {
                    sp -= 1;
                    p = Some(&self.nodes[stack[sp] as usize]);
                } else {
                    p = None;
                }
                continue;
            }
            if node.is_leaf() {
                if self.occlude_leaf(node, &mut ray) {
                    return true;
                }
                if sp > 0 {
                    sp -= 1;
                    p = Some(&self.nodes[stack[sp] as usize]);
                } else {
                    p = None;
                }
            } else {
                stack[sp] = node.right();
                sp += 1;
                p = Some(&self.nodes[node.left() as usize]);
            }
        }
        false
    }
}
