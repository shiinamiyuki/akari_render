// use std::env::{args, args_os, current_exe};
// use std::f32::consts::PI;
// use std::process::exit;

// use akari_render::color::Color;
// use akari_render::geometry::{face_forward, xyz_to_spherical};
// use akari_render::microfacet::{TrowbridgeReitzDistribution, MicrofacetDistribution};
// use akari_render::sampler::{init_pcg32_buffer, IndependentSampler, Pcg32, Sampler, init_pcg32_buffer_with_seed};
// use akari_render::sampling::{cos_sample_hemisphere, invert_cos_sample_hemisphere};
// use akari_render::scene::Scene;
// use akari_render::surface::*;

// use akari_render::surface::diffuse::DiffuseBsdf;
// use akari_render::*;
// use luisa::init_logger;
// use luisa_compute as luisa;

// #[derive(Clone, Copy, Value)]
// #[repr(C)]
// struct PdfSample {
//     pdf: f32,
//     w: Float3,
//     valid: bool,
// }

// fn test_bsdf_pdf(device: Device, sample_fn: impl Fn(Expr<Float3>) -> Expr<PdfSample>) {
//     let n_threads: u32 = 32768 * 4;
//     let seeds = init_pcg32_buffer(device.clone(), n_threads as usize);
//     let n_iters: i32 = 4096 * 4;

//     let final_bins = device.create_buffer::<f32>(100);
//     final_bins.fill(0.0);
//     let kernel = device.create_kernel::<()>(&|| {
//         let i = dispatch_id().x();
//         let bins = var!([f32; 100]);
//         let sampler = IndependentSampler::from_pcg32(var!(Pcg32, seeds.var().read(i)));
//         for_range(const_(0)..const_(n_iters), |_| {
//             let sample = sample_fn(sampler.next_3d());
//             lc_assert!(sample.pdf().is_finite());
//             if_!(sample.valid() & sample.pdf().cmpgt(0.0), {
//                 lc_assert!(sample.pdf().cmpgt(0.0));
//                 let w = sample.w();
//                 let w = face_forward(w, make_float3(0.0, 1.0, 0.0));
//                 let (theta, phi) = xyz_to_spherical(w);
//                 let phi = (phi + PI) / (2.0 * PI);
//                 lc_assert!(phi.cmpge(0.0) & phi.cmple(1.000001));
//                 let cos_theta = theta.cos();
//                 lc_assert!(cos_theta.cmpge(0.0));
//                 let bin_i = (cos_theta * 10.0).uint().min(9);
//                 let bin_j = (phi * 10.0).uint().min(9);
//                 let i = bin_i * 10 + bin_j;
//                 bins.write(i, bins.read(i) + 1.0 / sample.pdf());
//             });
//         });
//         for_range(0..100u32, |i| {
//             final_bins.var().atomic_fetch_add(i, bins.read(i));
//         });
//     });
//     kernel.dispatch([n_threads, 1, 1]);
//     let total_samples = n_threads as usize * n_iters as usize;
//     let final_bins = final_bins.copy_to_vec();
//     let sum = final_bins.iter().copied().sum::<f32>();
//     let expected = 2.0 * PI;
//     println!("sum: {:6.3}", sum / total_samples as f32);
//     let mut num_bad = 0;
//     for i in 0..10 {
//         for j in 0..10 {
//             let i = i as usize;
//             let j = j as usize;
//             let bin = final_bins[i * 10 + j];
//             let area = bin / total_samples as f32 * 100.0;
//             if (area - expected).abs() > 0.1 {
//                 num_bad += 1;
//             }
//             print!("{:6.3} ", area);
//         }
//         println!("");
//     }
//     println!("bad: {}", num_bad);
//     if num_bad > 0 {
//         eprintln!("test failed");
//         exit(-1);
//     }
// }

// fn test_bsdf(device: &Device) {
//     let mut current_dir = std::fs::canonicalize(std::env::current_exe().unwrap())
//         .unwrap()
//         .parent()
//         .unwrap()
//         .parent()
//         .unwrap()
//         .parent()
//         .unwrap()
//         .to_owned();
//     current_dir.push("scenes");
//     current_dir.push("cbox");
//     current_dir.push("scene.json");
//     dbg!(&current_dir);
//     let dummy_scene = Scene::load_from_path(device.clone(), current_dir);
//     let color_repr = color::ColorRepr::Rgb(color::RgbColorSpace::SRgb);
//     let eval = dummy_scene.texture_evaluator(color_repr);
//     let test_bsdf = |name: &str, bsdf: &dyn Fn() -> Box<dyn Bsdf>| {
//         println!("testing {}", name);
//         test_bsdf_pdf(device.clone(), |u| {
//             let bsdf = bsdf();
//             let wo: Float3Expr = make_float3(0.0, 1.0, 0.0);

//             let sample = bsdf.sample(
//                 wo,
//                 u.x(),
//                 u.yz(),
//                 &BsdfEvalContext {
//                     texture: &eval,
//                     color_repr,
//                 },
//             );
//             PdfSampleExpr::new(sample.pdf, sample.wi, sample.valid)
//         });
//     };
//     test_bsdf("diffuse", &|| {
//         Box::new(DiffuseBsdf {
//             reflectance: Color::one(color_repr),
//         })
//     });

//     for roughness in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] {
//         test_bsdf(
//             &format!("microfacet reflection, roughess={}", roughness),
//             &|| {
//                 Box::new(MicrofacetReflection {
//                     color: Color::one(color_repr),
//                     fresnel: Box::new(ConstFresnel {}),
//                     dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
//                         make_float2(roughness, roughness),
//                         false,
//                     )),
//                 })
//             },
//         );
//     }

//     for roughness in [0.8] {
//         test_bsdf(
//             &format!("microfacet transmission, roughess={}", roughness),
//             &|| {
//                 Box::new(MicrofacetTransmission {
//                     color: Color::one(color_repr),
//                     fresnel: Box::new(ConstFresnel {}),
//                     dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
//                         make_float2(roughness, roughness),
//                         false,
//                     )),
//                     eta: const_(1.1f32),
//                 })
//             },
//         );
//     }
// }
// fn test_invert_helper(
//     device: &Device,
//     name: &str,
//     sample: impl Fn(Expr<Float2>) -> Expr<Float3>,
//     invert: impl Fn(Expr<Float3>) -> Expr<Float2>,
// ) {
//     let count = 8192;
//     let samples = 256;
//     let rngs = init_pcg32_buffer_with_seed(device.clone(), count, 0);
//     let bads = device.create_buffer::<u32>(count);
//     bads.fill(0);
//     let printer = Printer::new(&device, 32768);
//     let kernel = device.create_kernel::<()>(&|| {
//         let i = dispatch_id().x();
//         let sampler = IndependentSampler::from_pcg32(var!(Pcg32, rngs.var().read(i)));
//         for_range(0..samples, |_| {
//             let u = sampler.next_2d();
//             let w = sample(u);
//             let u2 = invert(w);
//             let bad = (u2 - u).length().cmpgt(0.01);
//             if_!(bad, {
//                 bads.var().atomic_fetch_add(i, 1);
//                 lc_info!(printer, "bad: u: {:?} u2:{:?} w:{:?}", u, u2, w);
//             });
//         });
//     });
//     let stream = device.default_stream();
//     stream.with_scope(|s| {
//         s.reset_printer(&printer)
//             .submit([kernel.dispatch_async([count as u32, 1, 1])])
//             .print(&printer);
//     });

//     let bads = bads.copy_to_vec();
//     let bads = bads.iter().copied().sum::<u32>();
//     println!("Test invert: `{}`, bad: {}", name, bads);
// }
// fn test_invert(device: &Device) {
//     test_invert_helper(
//         device,
//         "cos_sample_hemisphere",
//         cos_sample_hemisphere,
//         invert_cos_sample_hemisphere,
//     );
//     let ax = 0.2;
//     let ay = 0.3;
//     test_invert_helper(
//         device,
//         "invert_ggx_iso",
//         |u|{
//             let dist = TrowbridgeReitzDistribution::from_alpha(make_float2(ax, ax), false);
//             dist.sample_wh(make_float3(0.0,1.0,0.0), u)
//         },
//         |w|{
//             let dist = TrowbridgeReitzDistribution::from_alpha(make_float2(ax, ax), false);
//             dist.invert_wh(make_float3(0.0,1.0,0.0), w)
//         },
//     );
//     test_invert_helper(
//         device,
//         "invert_ggx_aniso",
//         |u|{
//             let dist = TrowbridgeReitzDistribution::from_alpha(make_float2(ax, ay), false);
//             dist.sample_wh(make_float3(0.0,1.0,0.0), u)
//         },
//         |w|{
//             let dist = TrowbridgeReitzDistribution::from_alpha(make_float2(ax, ay), false);
//             dist.invert_wh(make_float3(0.0,1.0,0.0), w)
//         },
//     );
// }
// fn main() {
//     let ctx = luisa::Context::new(current_exe().unwrap());
//     let args = args().collect::<Vec<_>>();
//     if args.len() != 3 {
//         eprintln!("usage: {} <device> <test>", args[0]);
//         exit(-1);
//     }
//     let device = if args.len() == 1 {
//         ctx.create_device("cpu")
//     } else {
//         ctx.create_device(&args[1])
//     };
//     init_logger();
//     let test = &args[2];
//     match test.as_str() {
//         "bsdf" => test_bsdf(&device),
//         "invert" => test_invert(&device),
//         _ => {
//             eprintln!("unknown test {}", test);
//             exit(-1);
//         }
//     }
// }

fn main(){}