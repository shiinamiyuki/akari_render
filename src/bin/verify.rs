use akari::bsdf::{DiffuseBsdfClosure, LocalBsdfClosure};
use akari::sampler::{PCGSampler, SobolSampler};
use akari::util::PerThread;
use akari::*;
use bsdf::DiffuseBsdf;
use ltc::GgxLtcBsdfClosure;
use rand::thread_rng;
use rand::Rng;

pub struct BsdfTester<B: LocalBsdfClosure> {
    bsdf: B,
}
impl<B: LocalBsdfClosure> BsdfTester<B> {
    #[allow(dead_code)]
    pub fn new(bsdf: B) -> Self {
        Self { bsdf }
    }
    #[allow(dead_code)]
    pub fn test_2pi(&self, wo: Vec3) -> f32 {
        use akari::sampler::Sampler;
        let n = 1u64 << 20;
        let mut local_sum = PerThread::new(|| 0.0f64);
        rayon::scope(|s| {
            for i in 0..rayon::current_num_threads() {
                let local_sum = &local_sum;
                s.spawn(move |_| {
                    let mut sampler = PCGSampler::new(i as u64);
                    let integral = (0..n)
                        .fold(RobustSum::new(0.0f64), |mut sum, _| {
                            sampler.start_next_sample();
                            let estimate = if let Some(s) = self.bsdf.sample(sampler.next2d(), wo) {
                                1.0 / s.pdf as f64
                            } else {
                                0.0
                            };
                            if estimate.is_finite() {
                                sum.add(estimate);
                                // sum += estimate
                            }
                            sum
                        })
                        .sum()
                        / n as f64;
                    *local_sum.get_mut() = integral;
                });
            }
        });
        let integral =
            local_sum.inner().iter().map(|x| *x).sum::<f64>() / rayon::current_num_threads() as f64;
        integral as f32
    }
}

fn verify<B: LocalBsdfClosure>(bsdf: B) {
    let mut rng = thread_rng();
    let tester = BsdfTester::new(bsdf);
    let w = consine_hemisphere_sampling(vec2(rng.gen(), rng.gen()));
    let pi2 = tester.test_2pi(w);
    println!("2pi = {} avg = {}", PI * 2.0, pi2);
}

fn test_diffuse() {
    let diffuse = DiffuseBsdfClosure {
        color: Spectrum::one(),
    };
    verify(diffuse);
}
fn test_ggx() {
    let mut roughness = 0.01;
    while roughness < 1.0 {
        let bsdf = GgxLtcBsdfClosure {
            color: Spectrum::one(),
            roughness,
        };
        println!("roughness = {}", roughness);
        verify(bsdf);
        roughness += 0.05;
    }
}
fn main() {
    test_diffuse();
    test_ggx();
}
