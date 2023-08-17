use std::env::{args, args_os, current_exe};
use std::f32::consts::PI;
use std::process::exit;
use std::sync::Arc;

use akari_render::color::Color;
use akari_render::geometry::{face_forward, xyz_to_spherical};
use akari_render::microfacet::TrowbridgeReitzDistribution;
use akari_render::sampler::{init_pcg32_buffer, IndependentSampler, Pcg32, Sampler};
use akari_render::scene::Scene;
use akari_render::surface::*;

use akari_render::surface::diffuse::DiffuseBsdf;
use akari_render::util::LocalFileResolver;
use akari_render::*;
use luisa_compute as luisa;

#[derive(Clone, Copy, Value)]
#[repr(C)]
struct PdfSample {
    pdf: f32,
    w: Float3,
    valid: bool,
}

fn test_bsdf_pdf(device: Device, sample_fn: impl Fn(Expr<Float3>) -> Expr<PdfSample>) {
    let n_threads: u32 = 32768 * 4;
    let seeds = init_pcg32_buffer(device.clone(), n_threads as usize);
    let n_iters: i32 = 4096 * 4;

    let final_bins = device.create_buffer::<f32>(100);
    final_bins.fill(0.0);
    let kernel = device.create_kernel::<()>(&|| {
        let i = dispatch_id().x();
        let bins = var!([f32; 100]);
        let sampler = IndependentSampler {
            state: var!(Pcg32, seeds.var().read(i)),
        };
        for_range(const_(0)..const_(n_iters), |_| {
            let sample = sample_fn(sampler.next_3d());
            lc_assert!(sample.pdf().is_finite());
            if_!(sample.valid() & sample.pdf().cmpgt(0.0), {
                lc_assert!(sample.pdf().cmpgt(0.0));
                let w = sample.w();
                let w = face_forward(w, make_float3(0.0, 1.0, 0.0));
                let (theta, phi) = xyz_to_spherical(w);
                let phi = (phi + PI) / (2.0 * PI);
                lc_assert!(phi.cmpge(0.0) & phi.cmple(1.000001));
                let cos_theta = theta.cos();
                lc_assert!(cos_theta.cmpge(0.0));
                let bin_i = (cos_theta * 10.0).uint().min(9);
                let bin_j = (phi * 10.0).uint().min(9);
                let i = bin_i * 10 + bin_j;
                bins.write(i, bins.read(i) + 1.0 / sample.pdf());
            });
        });
        for_range(0..100u32, |i| {
            final_bins.var().atomic_fetch_add(i, bins.read(i));
        });
    });
    kernel.dispatch([n_threads, 1, 1]);
    let total_samples = n_threads as usize * n_iters as usize;
    let final_bins = final_bins.copy_to_vec();
    let sum = final_bins.iter().copied().sum::<f32>();
    let expected = 2.0 * PI;
    println!("sum: {:6.3}", sum / total_samples as f32);
    let mut num_bad = 0;
    for i in 0..10 {
        for j in 0..10 {
            let i = i as usize;
            let j = j as usize;
            let bin = final_bins[i * 10 + j];
            let area = bin / total_samples as f32 * 100.0;
            if (area - expected).abs() > 0.1 {
                num_bad += 1;
            }
            print!("{:6.3} ", area);
        }
        println!("");
    }
    println!("bad: {}", num_bad);
    if num_bad > 0 {
        eprintln!("test failed");
        exit(-1);
    }
}
fn main() {
    let ctx = luisa::Context::new(current_exe().unwrap());
    let args = args().collect::<Vec<_>>();
    let device = if args.len() == 1 {
        ctx.create_device("cpu")
    } else {
        ctx.create_device(&args[1])
    };
    let mut current_dir = std::fs::canonicalize(std::env::current_exe().unwrap())
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();
    current_dir.push("scenes");
    current_dir.push("cbox");
    current_dir.push("scene.json");
    dbg!(&current_dir);
    let dummy_scene = Scene::load_from_path(device.clone(), current_dir);
    let color_repr = color::ColorRepr::Rgb(color::RgbColorSpace::SRgb);
    let eval = dummy_scene.texture_evaluator(color_repr);
    let test_bsdf = |name: &str, bsdf: &dyn Fn() -> Box<dyn Bsdf>| {
        println!("testing {}", name);
        test_bsdf_pdf(device.clone(), |u| {
            let bsdf = bsdf();
            let wo: Float3Expr = make_float3(0.0, 1.0, 0.0);

            let sample = bsdf.sample(
                wo,
                u.x(),
                u.yz(),
                &BsdfEvalContext {
                    texture: &eval,
                    color_repr,
                },
            );
            PdfSampleExpr::new(sample.pdf, sample.wi, sample.valid)
        });
    };
    test_bsdf("diffuse", &|| {
        Box::new(DiffuseBsdf {
            reflectance: Color::one(color_repr),
        })
    });

    for roughness in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] {
        test_bsdf(
            &format!("microfacet reflection, roughess={}", roughness),
            &|| {
                Box::new(MicrofacetReflection {
                    color: Color::one(color_repr),
                    fresnel: Box::new(ConstFresnel {}),
                    dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                        make_float2(roughness, roughness),
                        false,
                    )),
                })
            },
        );
    }

    for roughness in [0.8] {
        test_bsdf(
            &format!("microfacet transmission, roughess={}", roughness),
            &|| {
                Box::new(MicrofacetTransmission {
                    color: Color::one(color_repr),
                    fresnel: Box::new(ConstFresnel {}),
                    dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                        make_float2(roughness, roughness),
                        false,
                    )),
                    eta: const_(1.1f32),
                })
            },
        );
    }
}
