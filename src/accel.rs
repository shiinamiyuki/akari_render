use std::collections::HashSet;
use std::pin::Pin;

use futures::future;
use futures::Future;

use crate::bsdf::*;
use crate::shape::*;
use crate::*;
pub mod bvh {

    use std::sync::Mutex;

    use ordered_float::OrderedFloat;

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
    #[derive(Default, Clone, Copy, Debug)]
    struct Bucket {
        count: usize,
        aabb: Bounds3f,
    }
    // pub struct BinnedSAHBuilder<T: BVHData> {
    //     pub data: T,
    //     pub nodes: Mutex<Option<Vec<BVHNode>>>,
    //     pub aabb: Mutex<Bounds3f>,
    // }
    // impl<T> BinnedSAHBuilder<T>
    // where
    //     T: BVHData + Sync + 'static,
    // {
    //     pub fn build(data: T, mut references: Vec<u32>) -> BVHAccelerator<T> {
    //         let n_prims = references.len();
    //         let (data, nodes, aabb) = {
    //             let builder = Self {
    //                 data,
    //                 nodes: Mutex::new(Some(vec![])),
    //                 aabb: Mutex::new(Bounds3f::default()),
    //             };

    //             builder.recursive_build::<16>(0, references.len() as u32, 0, &mut references);
    //             // bvh.references = references;

    //             let nodes = {
    //                 let mut nodes = builder.nodes.lock().unwrap();
    //                 nodes.replace(vec![]).unwrap()
    //             };
    //             let aabb = *builder.aabb.lock().unwrap();
    //             (builder.data, nodes, aabb)
    //         };

    //         let bvh = BVHAccelerator {
    //             data,
    //             nodes,
    //             references,
    //             aabb,
    //         };
    //         log::info!(
    //             "built bvh for {} references; nodes: {}",
    //             n_prims,
    //             bvh.nodes.len()
    //         );
    //         bvh
    //     }
    //     fn recursive_build<const N_BUCKETS: usize>(
    //         &self,
    //         begin: u32,
    //         end: u32,
    //         depth: u32,
    //         references: &mut Vec<u32>,
    //     ) -> u32 {
    //         // println!("building {}..{}", begin, end);
    //         let mut aabb = Bounds3f::default();
    //         for i in begin..end {
    //             aabb.insert_box(&self.data.aabb(references[i as usize]));
    //         }
    //         for i in 0..3 {
    //             if aabb.size()[i] == 0.0 {
    //                 aabb.max[i] += 0.001;
    //             }
    //         }
    //         if depth == 0 {
    //             let mut self_aabb = self.aabb.lock().unwrap();
    //             *self_aabb = aabb;
    //         }
    //         if end - begin <= 4 || depth >= 30 {
    //             if end - begin == 0 {
    //                 panic!("");
    //             }
    //             let node = BVHNode {
    //                 // axis: 0,
    //                 aabb,
    //                 first: begin,
    //                 count: end - begin,
    //                 left: 0,
    //                 right: 0,
    //             };
    //             let mut nodes = self.nodes.lock().unwrap();
    //             let nodes = nodes.as_mut().unwrap();
    //             nodes.push(node);
    //             return (nodes.len() - 1) as u32;
    //         } else {
    //             let try_split_with_axis = |axis| {
    //                 let mut buckets = [Bucket::default(); N_BUCKETS];
    //                 for i in begin..end {
    //                     let p_aabb = self.data.aabb(references[i as usize]);
    //                     let b = (N_BUCKETS - 1).min(
    //                         (aabb.offset(&p_aabb.centroid())[axis] * N_BUCKETS as Float) as usize,
    //                     );
    //                     buckets[b].count += 1;
    //                     buckets[b].aabb.insert_box(&p_aabb);
    //                 }
    //                 let mut costs = vec![0.0; N_BUCKETS - 1];
    //                 for i in 0..N_BUCKETS - 1 {
    //                     let mut b0 = Bounds3f::default();
    //                     let mut b1 = Bounds3f::default();
    //                     let mut count0 = 0;
    //                     let mut count1 = 0;
    //                     for j in 0..=i {
    //                         b0.insert_box(&buckets[j].aabb);
    //                         count0 += buckets[j].count;
    //                     }
    //                     for j in i + 1..N_BUCKETS {
    //                         b1.insert_box(&buckets[j].aabb);
    //                         count1 += buckets[j].count;
    //                     }
    //                     costs[i] = 0.125
    //                         + if count0 == 0 {
    //                             0.0
    //                         } else {
    //                             count0 as Float * b0.surface_area()
    //                         }
    //                         + if count1 == 0 {
    //                             0.0
    //                         } else {
    //                             count1 as Float * b1.surface_area()
    //                         };
    //                     if costs[i].is_infinite() {
    //                         println!(
    //                             "{} {:?} {:?} {} {} {:?}",
    //                             i, b0, b1, count0, count1, buckets
    //                         );
    //                     }
    //                 }
    //                 let mut split_buckets = 0;
    //                 let mut min_cost = Float::INFINITY;
    //                 for i in 0..N_BUCKETS - 1 {
    //                     if costs[i] < min_cost {
    //                         min_cost = costs[i];
    //                         split_buckets = i;
    //                     }
    //                 }
    //                 (axis, split_buckets, min_cost)
    //             };
    //             let splits: Vec<_> = (0..3usize).map(try_split_with_axis).collect();
    //             let best_split =
    //                 splits
    //                     .iter()
    //                     .fold((0usize, 0usize, std::f64::INFINITY as Float), |a, b| {
    //                         if a.2 < b.2 {
    //                             a
    //                         } else {
    //                             *b
    //                         }
    //                     });
    //             let axis = best_split.0 as usize;
    //             let split_bucket = best_split.1;
    //             // partition
    //             {
    //                 let predicate = |idx: u32| {
    //                     let b = {
    //                         let b = (aabb.offset(&self.data.aabb(idx).centroid())[axis]
    //                             * N_BUCKETS as Float) as usize;
    //                         b.min(N_BUCKETS - 1)
    //                     };
    //                     b <= split_bucket
    //                 };
    //                 let mut first = (|| {
    //                     for i in begin..end {
    //                         if !predicate(references[i as usize]) {
    //                             return i;
    //                         }
    //                     }
    //                     end
    //                 })();
    //                 let mut mid: u32 = (|| {
    //                     if first == end {
    //                         return first;
    //                     }
    //                     for i in first + 1..end {
    //                         if predicate(references[i as usize]) {
    //                             references.swap(first as usize, i as usize);
    //                             first += 1;
    //                         }
    //                     }
    //                     return first;
    //                 })();
    //                 if mid == begin || mid == end {
    //                     // println!("{:?} {:?}", costs, buckets);
    //                     if end - begin > 12 {
    //                         eprintln!(
    //                             "cannot split at depth {} with {} references",
    //                             depth,
    //                             end - begin
    //                         );
    //                     }
    //                     mid = (end + begin) / 2;
    //                 }
    //                 let ret = {
    //                     let mut nodes = self.nodes.lock().unwrap();
    //                     let nodes = nodes.as_mut().unwrap();
    //                     let ret = nodes.len();
    //                     nodes.push(BVHNode {
    //                         // axis: axis as u8,
    //                         aabb,
    //                         first: 0,
    //                         count: 0,
    //                         left: 0,
    //                         right: 0,
    //                     });
    //                     ret
    //                 };
    //                 let p_ref = UnsafePointer::new(references as *mut Vec<u32>);
    //                 // we know the two parts are disjoint
    //                 let (left, right) = {
    //                     if end - begin >= 128 * 1024 {
    //                         rayon::join(
    //                             || {
    //                                 self.recursive_build::<N_BUCKETS>(
    //                                     begin,
    //                                     mid,
    //                                     depth + 1,
    //                                     unsafe { p_ref.as_mut().unwrap() },
    //                                 )
    //                             },
    //                             || {
    //                                 self.recursive_build::<N_BUCKETS>(mid, end, depth + 1, unsafe {
    //                                     p_ref.as_mut().unwrap()
    //                                 })
    //                             },
    //                         )
    //                     } else {
    //                         let left = self.recursive_build::<N_BUCKETS>(
    //                             begin,
    //                             mid,
    //                             depth + 1,
    //                             references,
    //                         );

    //                         let right =
    //                             self.recursive_build::<N_BUCKETS>(mid, end, depth + 1, references);
    //                         (left, right)
    //                     }
    //                 };

    //                 {
    //                     let mut nodes = self.nodes.lock().unwrap();
    //                     let nodes = nodes.as_mut().unwrap();
    //                     nodes[ret].left = left;
    //                     nodes[ret].right = right;
    //                 }
    //                 return ret as u32;
    //             }
    //         }
    //     }
    // }
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
                aabb.insert_box(&self.data.aabb(references[i as usize]));
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

                        references[begin as usize..end as usize].sort_unstable_by(|a, b| {
                            let box_a = self.data.aabb(*a);
                            let box_b = self.data.aabb(*b);
                            OrderedFloat::<Float>(box_a.centroid()[$axis])
                                .partial_cmp(&OrderedFloat::<Float>(box_b.centroid()[$axis]))
                                .unwrap()
                        });
                        {
                            let mut aabb = Bounds3f::default();
                            for i in begin..end {
                                aabb.insert_box(&self.data.aabb(references[i as usize]));
                                surface_area_fwd[(i - begin) as usize] = aabb.surface_area() as f64;
                            }
                        }
                        {
                            let mut aabb = Bounds3f::default();
                            for i in (begin..end).rev() {
                                aabb.insert_box(&self.data.aabb(references[i as usize]));
                                surface_area_rev[(i - begin) as usize] = aabb.surface_area() as f64;
                            }
                        }
                        let mut min_cost = f64::INFINITY;
                        let mut split = 0;
                        for i in begin..(end - 1) {
                            let cost = (i - begin + 1) as f64
                                * surface_area_fwd[(i - begin) as usize]
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
                        // let right: Vec<_> = references[split as usize..end as usize]
                        //     .iter()
                        //     .map(|x| *x)
                        //     .collect();
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
                    aabb.insert_box(&nodes[2 * i + 2].aabb);
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
                        map[((self.nodes[i].left_or_first_primitive - 1) / 2) as usize] as u32 * 2
                            + 1;
                }
            }
            self
        }
        fn intersect_aabb(aabb: &Bounds3f, ray: &Ray, invd: &Vec3) -> Float {
            let t0 = (aabb.min - ray.o).component_mul(&invd);
            let t1 = (aabb.max - ray.o).component_mul(&invd);
            let min = glm::min2(&t0, &t1);
            let max = glm::max2(&t0, &t1);
            let tmin = glm::comp_max(&min).max(ray.tmin);
            let tmax = glm::comp_min(&max).min(ray.tmax);
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
            let mut stack: [Option<&BVHNode>; 32] = [None; 32];
            let mut sp = 0;
            let mut p = Some(&self.nodes[0]);
            let mut ray = *original_ray;
            let invd: Vec3 = vec3(1.0, 1.0, 1.0).component_div(&ray.d);
            let mut isct = None;
            if self.nodes[0].is_leaf() {
                return self.intersect_leaf(&self.nodes[0], &mut ray);
            }
            macro_rules! intersect_leaf_and_pop {
                ($node:expr) => {
                    if let Some(hit) = self.intersect_leaf($node, &mut ray) {
                        isct = Some(hit);
                    }
                    if sp > 0 {
                        sp -= 1;
                        p = stack[sp];
                    } else {
                        p = None;
                    }
                };
            }
            macro_rules! intersect_leaf {
                ($node:expr) => {
                    if let Some(hit) = self.intersect_leaf($node, &mut ray) {
                        isct = Some(hit);
                    }
                };
            }
            while p.is_some() {
                let node = p.unwrap();
                if node.is_leaf() {
                    if let Some(hit) = self.intersect_leaf(node, &mut ray) {
                        isct = Some(hit);
                    }
                    if sp > 0 {
                        sp -= 1;
                        p = stack[sp];
                    } else {
                        p = None;
                    }
                } else {
                    let left = &self.nodes[node.left() as usize];
                    let right = &self.nodes[node.right() as usize];
                    let t_left = Self::intersect_aabb(&left.aabb, &ray, &invd);
                    let t_right = Self::intersect_aabb(&right.aabb, &ray, &invd);
                    if t_left < 0.0 && t_right < 0.0 {
                        if sp > 0 {
                            sp -= 1;
                            p = stack[sp];
                        } else {
                            p = None;
                        }
                        continue;
                    }
                    if t_left < 0.0 {
                        if right.is_leaf() {
                            intersect_leaf_and_pop!(right);
                        } else {
                            p = Some(right);
                        }
                    } else if t_right < 0.0 {
                        if left.is_leaf() {
                            intersect_leaf_and_pop!(left);
                        } else {
                            p = Some(left);
                        }
                    } else {
                        if left.is_leaf() && right.is_leaf() {
                            if t_left < t_right {
                                intersect_leaf!(left);
                                intersect_leaf_and_pop!(right);
                            } else {
                                intersect_leaf!(right);
                                intersect_leaf_and_pop!(left);
                            }
                        } else if left.is_leaf() {
                            intersect_leaf!(left);
                            p = Some(right);
                        } else if right.is_leaf() {
                            intersect_leaf!(right);
                            p = Some(left);
                        } else {
                            if t_left < t_right {
                                stack[sp] = Some(right);
                                sp += 1;
                                p = Some(left);
                            } else {
                                stack[sp] = Some(left);
                                sp += 1;
                                p = Some(right);
                            }
                        }
                    }
                }
            }
            isct
        }
        pub fn occlude(&self, original_ray: &Ray) -> bool {
            let mut stack: [Option<&BVHNode>; 64] = [None; 64];
            let mut sp = 0;
            let mut p = Some(&self.nodes[0]);
            let mut ray = *original_ray;
            let invd: Vec3 = vec3(1.0, 1.0, 1.0).component_div(&ray.d);
            while p.is_some() {
                let node = p.unwrap();
                let t = Self::intersect_aabb(&node.aabb, &ray, &invd);
                if t < 0.0 {
                    if sp > 0 {
                        sp -= 1;
                        p = stack[sp];
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
                        p = stack[sp];
                    } else {
                        p = None;
                    }
                } else {
                    stack[sp] = Some(&self.nodes[node.right() as usize]);
                    sp += 1;
                    p = Some(&self.nodes[node.left() as usize]);
                }
            }
            false
        }
    }
}
impl_as_any!(Arc<dyn Shape>);
impl Shape for Arc<dyn Shape> {
    fn intersect<'a>(&'a self, ray: &Ray) -> Option<Intersection<'a>> {
        self.as_ref().intersect(ray)
    }
    fn occlude(&self, ray: &Ray) -> bool {
        self.as_ref().occlude(ray)
    }
    fn bsdf<'a>(&'a self) -> Option<&'a dyn Bsdf> {
        self.as_ref().bsdf()
    }
    fn aabb(&self) -> Bounds3f {
        self.as_ref().aabb()
    }
    fn sample_surface(&self, u: &Vec3) -> SurfaceSample {
        self.as_ref().sample_surface(u)
    }
    fn area(&self) -> Float {
        self.as_ref().area()
    }
}
struct GenericBVHData {
    shapes: Vec<Arc<dyn Shape>>,
}

impl bvh::BVHData for GenericBVHData {
    fn intersect<'a>(&'a self, idx: u32, ray: &Ray) -> Option<Intersection<'a>> {
        self.shapes[idx as usize].intersect(ray)
    }
    fn occlude(&self, idx: u32, ray: &Ray) -> bool {
        self.shapes[idx as usize].occlude(ray)
    }
    fn bsdf<'a>(&'a self, idx: u32) -> Option<&'a dyn Bsdf> {
        self.shapes[idx as usize].bsdf()
    }
    fn aabb(&self, idx: u32) -> Bounds3f {
        self.shapes[idx as usize].aabb()
    }
}

pub struct Aggregate {
    bvh: bvh::BVHAccelerator<GenericBVHData>,
    area: Float,
}
impl Aggregate {
    pub fn shapes(&self) -> impl Iterator<Item = &Arc<dyn Shape>> {
        self.bvh.data.shapes.iter()
    }
    pub fn new(shapes: Vec<Arc<dyn Shape>>) -> Self {
        let v: Vec<u32> = (0..shapes.len() as u32).collect();
        let area: Float = shapes.iter().map(|s| s.area()).sum();
        let data = GenericBVHData { shapes };
        Self {
            bvh: bvh::SweepSAHBuilder::build(data, v),
            area,
        }
    }
    pub async fn new_async(shapes: Vec<Pin<Box<dyn Future<Output = Arc<dyn Shape>>>>>) -> Self {
        let v: Vec<u32> = (0..shapes.len() as u32).collect();
        let shapes = future::join_all(shapes.into_iter()).await;
        let area: Float = shapes.iter().map(|s| s.area()).sum();
        let data = GenericBVHData { shapes };
        Self {
            bvh: bvh::SweepSAHBuilder::build(data, v),
            area,
        }
    }
}
impl_as_any!(Aggregate);
impl Shape for Aggregate {
    fn intersect<'a>(&'a self, original_ray: &Ray) -> Option<Intersection<'a>> {
        self.bvh.intersect(original_ray)
    }
    fn occlude(&self, ray: &Ray) -> bool {
        self.bvh.occlude(ray)
    }
    fn bsdf<'a>(&'a self) -> Option<&'a dyn Bsdf> {
        None
    }
    fn aabb(&self) -> Bounds3f {
        Bounds3f::default()
    }
    fn area(&self) -> Float {
        self.area
    }
    fn sample_surface(&self, u: &Vec3) -> SurfaceSample {
        let len = self.bvh.data.shapes.len() as Float;
        let i = u[2] as Float * len;
        let i = i as usize;
        let u = vec3(u[0], u[1], (u[2] - i as Float / len) * (len - i as Float));
        self.bvh.data.shapes[i].sample_surface(&u)
    }
}
