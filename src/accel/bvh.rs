use crate::shape::*;
use crate::texture::ShadingPoint;

use ordered_float::OrderedFloat;
use parking_lot::Mutex;

use super::*;
#[derive(Clone, Copy, Debug, Default)]
pub struct BvhNode {
    pub aabb: Bounds3f,
    pub left_or_first_primitive: u32,
    pub count: u8,
    pub axis: u8,
}
impl BvhNode {
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
pub trait BvhData: Send + Sync {
    fn aabb(&self, idx: u32) -> Bounds3f;
}
pub struct BvhAccel<T: BvhData> {
    pub(crate) data: T,
    pub(crate) references: Vec<u32>,
    pub(crate) nodes: Vec<BvhNode>,
    pub(crate) aabb: Bounds3f,
}

pub struct SweepSAHBuilder<T: BvhData> {
    pub data: T,
    pub nodes: Mutex<Option<Vec<BvhNode>>>,
    pub aabb: Mutex<Bounds3f>,
}
impl<T> SweepSAHBuilder<T>
where
    T: BvhData + Sync + 'static,
{
    pub fn build(data: T, mut references: Vec<u32>) -> BvhAccel<T> {
        assert!(!references.is_empty());
        let n_prims = references.len();
        let ((data, nodes, aabb), t) = profile_fn(|| {
            let builder = Self {
                data,
                nodes: Mutex::new(Some(vec![BvhNode::default()])),
                aabb: Mutex::new(Bounds3f::default()),
            };

            builder.recursive_build(0, 0, references.len() as u32, 0, &mut references);
            // bvh.references = references;

            let nodes = {
                let mut nodes = builder.nodes.lock();
                nodes.replace(vec![]).unwrap()
            };
            let aabb = *builder.aabb.lock();
            (builder.data, nodes, aabb)
        });

        let bvh = BvhAccel {
            data,
            nodes,
            references,
            aabb,
        };
        log::info!(
            "BVH built in {}s, refs: {}, nodes:{}",
            t,
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
            let mut self_aabb = self.aabb.lock();
            *self_aabb = aabb;
        }
        if end - begin <= 1 || depth >= 40 {
            if end - begin == 0 {
                panic!("");
            }
            let node = BvhNode {
                axis: 0,
                aabb,
                left_or_first_primitive: begin,
                count: (end - begin) as u8,
            };
            let mut nodes = self.nodes.lock();
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
                    let mut nodes = self.nodes.lock();
                    let nodes = nodes.as_mut().unwrap();
                    let dummy = BvhNode::default();
                    let child_idx = nodes.len() as u32;
                    nodes.push(dummy);
                    nodes.push(dummy);
                    nodes[node_idx as usize] = BvhNode {
                        axis: best_split.0 as u8,
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

impl<T> BvhAccel<T>
where
    T: BvhData,
{
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
    fn intersect_aabb4(
        aabb: &Bounds3f,
        ray: &Ray4,
        active_mask: BVec4A,
        inv_dx: Vec4,
        inv_dy: Vec4,
        inv_dz: Vec4,
    ) -> (BVec4A, Vec4) {
        let t0x = (aabb.min[0] - ray.o[0]) * inv_dx;
        let t0y = (aabb.min[1] - ray.o[1]) * inv_dy;
        let t0z = (aabb.min[2] - ray.o[2]) * inv_dz;

        let t1x = (aabb.max[0] - ray.o[0]) * inv_dx;
        let t1y = (aabb.max[1] - ray.o[1]) * inv_dy;
        let t1z = (aabb.max[2] - ray.o[2]) * inv_dz;

        let tminx = t0x.min(t1x);
        let tminy = t0y.min(t1y);
        let tminz = t0z.min(t1z);

        let tmaxx = t0x.max(t1x);
        let tmaxy = t0y.max(t1y);
        let tmaxz = t0z.max(t1z);

        let tmin = tminx.max(tminy).max(tminz).max(ray.tmin);
        let tmax = tmaxx.min(tmaxy).min(tmaxz).min(ray.tmax);

        let mask = tmin.cmplt(tmax) & active_mask;

        (mask, tmin)
    }
    fn traverse_leaf<F: FnMut(&mut Ray, u32) -> bool>(
        &self,
        node: &BvhNode,
        ray: &mut Ray,
        f: &mut F,
    ) -> bool {
        unsafe {
            let first = node.left_or_first_primitive;
            let last = node.left_or_first_primitive + node.count as u32;
            for i in first..last {
                if f(ray, *self.references.get_unchecked(i as usize)) {
                    return true;
                }
            }
            false
        }
    }
    // break when f returns true
    // pub fn traverse<F: FnMut(&mut Ray, u32) -> bool>(&self, mut ray: Ray, mut f: F) {
    //     let inv_d = Vec3A::ONE / Vec3A::from(ray.d);
    //     let mut stack: [u32; 32] = [0; 32];
    //     let mut sp = 0;
    //     let mut p = Some(&self.nodes[0]);
    //     let o = ray.o.into();
    //     while p.is_some() {
    //         let node = p.unwrap();
    //         let t = Self::intersect_aabb(&node.aabb, &ray, o, inv_d);
    //         if t < 0.0 {
    //             if sp > 0 {
    //                 sp -= 1;
    //                 p = Some(&self.nodes[stack[sp] as usize]);
    //             } else {
    //                 p = None;
    //             }
    //         } else {
    //             if node.is_leaf() {
    //                 if self.traverse_leaf(node, &mut ray, &mut f) {
    //                     break;
    //                 }
    //                 if sp > 0 {
    //                     sp -= 1;
    //                     p = Some(&self.nodes[stack[sp] as usize]);
    //                 } else {
    //                     p = None;
    //                 }
    //             } else {
    //                 let left = &self.nodes[node.left() as usize];
    //                 let right = &self.nodes[node.right() as usize];
    //                 let axis = node.axis as usize;
    //                 if ray.d[axis] < 0.0 {
    //                     p = Some(right);
    //                     stack[sp] = node.left();
    //                     sp += 1;
    //                 } else {
    //                     p = Some(left);
    //                     stack[sp] = node.right();
    //                     sp += 1;
    //                 }
    //             }
    //         }
    //     }
    // }
    // break when f returns false
    pub fn traverse4<F: FnMut(&mut Ray4, BVec4A, u32) -> BVec4A>(
        &self,
        ray4: &mut Ray4,
        mut active_mask: BVec4A,
        mut f: F,
    ) {
        unsafe {
            let inv_dx = Vec4::ONE / ray4.o[0];
            let inv_dy = Vec4::ONE / ray4.o[1];
            let inv_dz = Vec4::ONE / ray4.o[2];

            let mut stack: [u32; 32] = [0; 32];
            let mut sp = 0;
            let mut p = Some(&self.nodes[0]);
            while let Some(node) = p {
                if !active_mask.any() {
                    break;
                }
                if node.is_leaf() {
                    let first = node.left_or_first_primitive;
                    let last = node.left_or_first_primitive + node.count as u32;
                    for i in first..last {
                        active_mask &= f(
                            ray4,
                            active_mask,
                            *self.references.get_unchecked(i as usize),
                        );
                    }
                    if sp > 0 {
                        sp -= 1;
                        p = Some(self.nodes.get_unchecked(stack[sp] as usize));
                    } else {
                        p = None;
                    }
                } else {
                    let left = self.nodes.get_unchecked(node.left() as usize);
                    let right = self.nodes.get_unchecked(node.right() as usize);

                    let (left_mask, left_t) = Self::intersect_aabb4(
                        &left.aabb,
                        ray4,
                        active_mask,
                        inv_dx,
                        inv_dy,
                        inv_dz,
                    );
                    let (right_mask, right_t) = Self::intersect_aabb4(
                        &right.aabb,
                        ray4,
                        active_mask,
                        inv_dx,
                        inv_dy,
                        inv_dz,
                    );
                    if !left_mask.any() && !right_mask.any() {
                        if sp > 0 {
                            sp -= 1;
                            p = Some(self.nodes.get_unchecked(stack[sp] as usize));
                        } else {
                            p = None;
                        }
                        continue;
                    }
                    if !left_mask.any() {
                        p = Some(right);
                    } else if !right_mask.any() {
                        p = Some(left);
                    } else {
                        if (left_t.cmplt(right_t) & left_mask & right_mask).any() {
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
        }
    }
    // break when f returns false
    pub fn traverse<F: FnMut(&mut Ray, u32) -> bool>(&self, mut ray: Ray, mut f: F) {
        unsafe {
            let inv_d = Vec3A::ONE / Vec3A::from(ray.d);
            let mut stack: [u32; 32] = [0; 32];
            let mut sp = 0;
            let mut p = Some(&self.nodes[0]);
            let o = ray.o.into();
            while let Some(node) = p {
                if node.is_leaf() {
                    if self.traverse_leaf(node, &mut ray, &mut f) {
                        break;
                    }
                    if sp > 0 {
                        sp -= 1;
                        p = Some(self.nodes.get_unchecked(stack[sp] as usize));
                    } else {
                        p = None;
                    }
                } else {
                    let left = self.nodes.get_unchecked(node.left() as usize);
                    let right = self.nodes.get_unchecked(node.right() as usize);
                    let t_left = Self::intersect_aabb(&left.aabb, &ray, o, inv_d);
                    let t_right = Self::intersect_aabb(&right.aabb, &ray, o, inv_d);
                    if t_left < 0.0 && t_right < 0.0 {
                        if sp > 0 {
                            sp -= 1;
                            p = Some(self.nodes.get_unchecked(stack[sp] as usize));
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
        }
    }

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
        self.nodes = vec![BvhNode::default(); nodes.len()];
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
}
#[macro_export]
macro_rules! impl_bvh_accel {
    ($t:ty) => {
        impl Accel for $t {
            fn hit_to_iteraction<'a>(&'a self, hit: RayHit) -> SurfaceInteraction<'a> {
                let uv = hit.uv;
                let ng = hit.ng;
                let shape = self.data.shapes[hit.geom_id as usize].as_ref();
                let triangle = shape.shading_triangle(hit.prim_id);
                let ns = triangle.ns(uv);
                let texcoord = triangle.texcoord(uv);
                SurfaceInteraction::<'a> {
                    shape,
                    bsdf: triangle.bsdf,
                    triangle,
                    t: hit.t,
                    uv,
                    ng,
                    ns,
                    sp: ShadingPoint { texcoord },
                    texcoord,
                }
            }
            fn intersect(&self, ray: &Ray) -> Option<RayHit> {
                let mut hit = None;
                self.traverse(*ray, |ray, geom_id| {
                    if let Some(hit_) = self.data.shapes[geom_id as usize].intersect(ray) {
                        hit = Some(RayHit { geom_id, ..hit_ });
                        ray.tmax = hit_.t;
                    }
                    true
                });
                hit
            }
            fn intersect4(&self, ray: &[Ray; 4], mask: [bool; 4]) -> [Option<RayHit>; 4] {
                todo!()
                // let mut hits = [None; 4];
                // let ray4 = Ray4::from(*ray);
                // let mask = BVec4A::new(mask[0], mask[1], mask[2], mask[3]);
                // self.traverse4(ray4, mask, |ray4, mask, geom_id| {
                //     let hits_ = self.data.shapes[geom_id as usize].intersect4(ray, mask);
                //     for i in 0..4 {
                //         if let Some(hit_) = hits_[i] {
                //             hits[i] = Some(RayHit { geom_id, ..hit_ });
                //             ray[i].tmax = hit_.t;
                //         }
                //     }
                //     true
                // });
                // hits
            }
            fn occlude4(&self, ray: &[Ray; 4], mask: [bool; 4]) -> [bool; 4] {
                todo!()
                // let ray4 = Ray4::from(*ray);
                // let mut mask = BVec4A::new(mask[0], mask[1], mask[2], mask[3]);
                // self.traverse4(ray4, mask, |ray4, mask, geom_id| {
                //     let occluded = self.data.shapes[geom_id as usize].occlude4(ray4, mask);
                //     mask &= !BVec4A::new(occluded[0], occluded[1], occluded[2], occluded[3]);
                //     mask
                // });
                // let mask = mask.bitmask();
                // [
                //     (mask & 1) == 0,
                //     (mask & 2) == 0,
                //     (mask & 4) == 0,
                //     (mask & 8) == 0,
                // ]
            }

            fn occlude(&self, ray: &Ray) -> bool {
                let mut occluded = false;
                self.traverse(*ray, |ray, geom_id| {
                    if self.data.shapes[geom_id as usize].occlude(ray) {
                        occluded = true;
                        return false;
                    }
                    true
                });
                occluded
            }
            fn shapes(&self) -> Vec<Arc<dyn Shape>> {
                self.data.shapes.clone()
            }
        }
    };
}

impl_bvh_accel!(BvhAccel<TopLevelBvhData>);
