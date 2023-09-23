use std::env::{args, args_os, current_exe};
use std::f32::consts::PI;
use std::process::exit;

use akari_render::color::{Color, SampledWavelengthsExpr};
use akari_render::geometry::{face_forward, xyz_to_spherical};
use akari_render::microfacet::{MicrofacetDistribution, TrowbridgeReitzDistribution};
use akari_render::sampler::{
    init_pcg32_buffer, init_pcg32_buffer_with_seed, IndependentSampler, Pcg32, Sampler,
};
use akari_render::sampling::{cos_sample_hemisphere, invert_cos_sample_hemisphere};
use akari_render::svm::surface::*;

use akari_render::svm::surface::diffuse::DiffuseBsdf;
use akari_render::*;
use luisa::init_logger;
use luisa_compute as luisa;

mod bsdf_chi2_test {
    use std::{
        fs::{create_dir_all, File},
        marker::PhantomData,
        rc::Rc,
        sync::Arc,
    };

    use akari_render::{
        color::SampledWavelengths,
        geometry::spherical_to_xyz,
        util::{chi2cdf, integration::adaptive_simpson_2d},
    };
    use libm::expf;
    use rand::{thread_rng, Rng};

    use super::*;
    fn pdf_histogram(
        device: &Device,
        wo: Float3,
        sample: impl Fn(Expr<Float3>, Expr<Float3>) -> BsdfSample,
        sample_count: usize,
        threads: usize,
        theta_res: u32,
        phi_res: u32,
    ) -> Vec<u32> {
        let target = device.create_buffer((phi_res * theta_res) as usize);
        target.fill(0u32);
        assert!(sample_count % threads == 0);
        let samples_per_thread = sample_count / threads;
        let rngs = init_pcg32_buffer_with_seed(device.clone(), threads, 0);
        let kernel =
            device.create_kernel::<fn(Float3, u32)>(&|wo: Expr<Float3>, samples: Expr<u32>| {
                let i = dispatch_id().x;
                let pcg = rngs.var().read(i).var();
                let sampler = IndependentSampler::from_pcg32(pcg);
                for_range(0u32.expr()..samples, |_| {
                    let u = sampler.next_3d();
                    let s = sample(wo, u);
                    if_!(s.valid, {
                        // cpu_dbg!(s.wi);
                        let (theta, phi) = xyz_to_spherical(s.wi);
                        let phi = select(phi.lt(0.0), phi + 2.0 * PI, phi) / (2.0 * PI);
                        let theta = theta / PI;
                        // cpu_dbg!(Float2::expr(phi, theta));
                        // cpu_dbg!(Float4::expr(s.wi.x, s.wi.y, s.wi.z, s.wi.y.acos()));
                        // cpu_dbg!(Float2::expr(theta / PI, s.wi.y.acos() / PI));
                        // lc_assert!(phi.ge(0.0) & phi.le(1.0001));
                        // lc_assert!(theta.ge(0.0) & theta.le(1.0001));
                        let bin_theta = (theta * theta_res as f32).cast_u32().min(theta_res - 1);
                        let bin_phi = (phi * phi_res as f32).cast_u32().min(phi_res - 1);
                        let bin_idx = bin_theta * phi_res + bin_phi;
                        target.var().atomic_fetch_add(bin_idx, 1);
                    });
                });
            });
        kernel.dispatch([threads as u32, 1, 1], &wo, &(samples_per_thread as u32));
        target.copy_to_vec()
    }
    fn integrate_pdf(
        device: &Device,
        wo: Float3,
        pdf: impl Fn(Expr<Float3>, Expr<Float3>) -> Expr<f32> + 'static,
        theta_res: u32,
        phi_res: u32,
    ) -> Vec<f32> {
        let pdf = Arc::new(pdf);
        let target = device.create_buffer((phi_res * theta_res) as usize);
        target.fill(0.0f32);
        let kernel = device.create_kernel::<fn(Float3)>(&|wo: Expr<Float3>| {
            set_block_size([8, 8, 1]);
            let ij = dispatch_id().xy();
            let theta_h = PI / theta_res as f32;
            let phi_h = 2.0 * PI / phi_res as f32;
            let bin_idx = ij.x * phi_res + ij.y;
            let i = ij.x;
            let j = ij.y;
            let theta_begin = theta_h * i.cast_f32();
            let theta_end = theta_h * (i + 1).cast_f32();
            let phi_begin = phi_h * j.cast_f32();
            let phi_end = phi_h * (j + 1).cast_f32();
            let pdf = pdf.clone();
            let integral = adaptive_simpson_2d::<Float3>(
                device,
                wo,
                move |wo, sph| {
                    let phi = sph.x;
                    let theta = sph.y;
                    let sin_theta = theta.sin();
                    let wi = spherical_to_xyz(theta, phi);
                    pdf(wo, wi) * sin_theta
                },
                Float2::expr(phi_begin, theta_begin),
                Float2::expr(phi_end, theta_end),
                1e-6,
                6,
            );
            target.write(bin_idx, integral);
        });
        kernel.dispatch([theta_res, phi_res, 1], &wo);
        target.copy_to_vec()
    }
    fn compute_error(
        observed_freq: &[f64],
        exp_freq: &[f64],
        theta_res: u32,
        phi_res: u32,
        sample_count: usize,
    ) -> (f64, f64) {
        let mse = (0..theta_res * phi_res)
            .map(|i| {
                let i = i as usize;
                let diff = (observed_freq[i] - exp_freq[i]) / sample_count as f64;
                diff * diff
            })
            .sum::<f64>()
            / (theta_res * phi_res) as f64;
        let rel_mse = (0..theta_res * phi_res)
            .map(|i| {
                let i = i as usize;
                let diff = (observed_freq[i] - exp_freq[i]) / (exp_freq[i] + 1.0);
                diff * diff
            })
            .sum::<f64>()
            / (theta_res * phi_res) as f64;
        (mse, rel_mse)
    }
    // based on pbrt-v4
    fn chi2test(
        observed_freq: &[f64],
        exp_freq: &[f64],
        theta_res: u32,
        phi_res: u32,
        sample_count: usize,
        min_exp_freq: f64,
        significant_level: f64,
        num_tests: usize,
    ) -> (bool, String) {
        #[derive(Clone, Copy, Debug)]
        struct Cell {
            exp_freq: f64,
            index: usize,
        }

        let mut cells = (0..theta_res * phi_res)
            .map(|i| {
                let i = i as usize;
                Cell {
                    exp_freq: exp_freq[i],
                    index: i,
                }
            })
            .collect::<Vec<_>>();
        cells.sort_by(|a, b| a.exp_freq.partial_cmp(&b.exp_freq).unwrap());
        let mut pooled_freq = 0.0f64;
        let mut pooled_exp_freq = 0.0f64;
        let mut chsq = 0.0f64;
        let mut pooled_cells = 0;
        let mut dof = 0;
        for c in &cells {
            let exp_freq = exp_freq[c.index];
            if exp_freq == 0.0 {
                if observed_freq[c.index] > sample_count as f64 * 1e-5 {
                    return (
                        false,
                        format!("exp_freq == 0.0, observed_freq: {}", observed_freq[c.index]),
                    );
                }
            } else if exp_freq < min_exp_freq {
                pooled_freq += observed_freq[c.index];
                pooled_exp_freq += exp_freq;
                pooled_cells += 1;
            } else if pooled_exp_freq > 0.0 && pooled_exp_freq < min_exp_freq {
                pooled_freq += observed_freq[c.index];
                pooled_exp_freq += exp_freq;
                pooled_cells += 1;
            } else {
                let diff = observed_freq[c.index] - exp_freq;
                chsq += diff * diff / exp_freq;
                dof += 1;
            }
        }
        if pooled_exp_freq > 0.0 || pooled_freq > 0.0 {
            let diff = pooled_freq - pooled_exp_freq;
            chsq += diff * diff / pooled_exp_freq;
            dof += 1;
        }
        dof -= 1;
        if dof <= 0 {
            return (false, format!("dof <= 0: {}", dof));
        }

        let pval = 1.0 - chi2cdf(chsq, dof);

        // Sidak correction
        let alpha = 1.0 - (1.0 - significant_level).powf(1.0 / num_tests as f64);
        // dbg!(chsq, dof);
        if pval < alpha || !pval.is_finite() {
            (
                false,
                format!(
                    "Reject null hypothesis for pval: {}, alpha: {}, significant_level: {}",
                    pval, alpha, significant_level
                ),
            )
        } else {
            (true, "".to_string())
        }
    }
    fn dump_tables(
        mut file: File,
        observed_freq: &[f64],
        exp_freq: &[f64],
        theta_res: u32,
        phi_res: u32,
    ) {
        use std::io::Write;
        write!(file, "import numpy as np\n").unwrap();
        write!(file, "import matplotlib.pyplot as plt\n").unwrap();
        write!(file, "observed_freq = [").unwrap();
        for i in 0..observed_freq.len() {
            write!(file, "{}, ", observed_freq[i]).unwrap();
        }
        write!(file, "]\n").unwrap();
        write!(file, "exp_freq = [").unwrap();
        for i in 0..exp_freq.len() {
            write!(file, "{}, ", exp_freq[i]).unwrap();
        }
        write!(file, "]\n").unwrap();
        write!(
            file,
            "observed_freq = np.array(observed_freq).reshape({}, {})\n",
            theta_res, phi_res
        )
        .unwrap();
        write!(
            file,
            "exp_freq = np.array(exp_freq).reshape({}, {})\n",
            theta_res, phi_res
        )
        .unwrap();
        writeln!(
            file,
            r#"fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(211)
ax.set_title('Observed Frequencies')
ax.set_xlabel('phi')
ax.set_ylabel('theta')
ax.imshow(observed_freq)

ax = fig.add_subplot(212)
ax.set_title('Expected Frequencies')
ax.set_xlabel('phi')
ax.set_ylabel('theta')
ax.imshow(exp_freq)

plt.show()"#
        )
        .unwrap();
    }
    fn test_bsdf(
        device: &Device,
        desc: &str,
        positive: bool,
        bsdf: impl Fn() -> Box<dyn Surface> + 'static,
    ) {
        let theta_res = 80;
        let phi_res = 2 * theta_res;
        let sample_count = 1000000;
        let threads = 10000;
        let min_exp_freq = 5f64;
        let runs = 5;
        let significant_level = 0.01;
        let mut rng = thread_rng();
        let bsdf = Rc::new(bsdf);
        let mut has_error = false;
        let mut max_mse = 0.0f64;
        let mut max_rel_mse = 0.0f64;
        for run in 0..runs {
            let wo = {
                let r = rng.gen_range(0.0..1.0f32).sqrt();
                let phi = rng.gen_range(0.0f32..2.0 * PI);
                let sign = if positive {
                    1.0
                } else {
                    if rng.gen_range(0.0..1.0) < 0.5 {
                        -1.0
                    } else {
                        1.0
                    }
                };
                Float3::new(
                    r * phi.cos(),
                    (1.0 - r * r).sqrt() * sign as f32,
                    r * phi.sin(),
                )
            };
            let histogram = pdf_histogram(
                device,
                wo,
                |wo, u| {
                    let bsdf = bsdf();
                    bsdf.sample(
                        wo,
                        u.x,
                        u.yz(),
                        SampledWavelengthsExpr::rgb_wavelengths().var(),
                        &BsdfEvalContext {
                            color_repr: color::ColorRepr::Rgb(color::RgbColorSpace::SRgb),
                            _marker: PhantomData {},
                            ad_mode: ADMode::None,
                        },
                    )
                },
                sample_count,
                threads,
                theta_res,
                phi_res,
            );
            let observed_freq = histogram.iter().map(|&x| x as f64).collect::<Vec<_>>();
            let bsdf = bsdf.clone();
            let int_pdf = integrate_pdf(
                device,
                wo,
                move |wo, wi| {
                    let bsdf = bsdf();
                    bsdf.pdf(
                        wo,
                        wi,
                        SampledWavelengthsExpr::rgb_wavelengths(),
                        &BsdfEvalContext {
                            color_repr: color::ColorRepr::Rgb(color::RgbColorSpace::SRgb),
                            _marker: PhantomData {},
                            ad_mode: ADMode::None,
                        },
                    )
                },
                theta_res,
                phi_res,
            );
            let exp_freq = int_pdf
                .iter()
                .map(|&x| x as f64 * sample_count as f64)
                .collect::<Vec<_>>();
            let result = chi2test(
                &observed_freq,
                &exp_freq,
                theta_res,
                phi_res,
                sample_count,
                min_exp_freq,
                significant_level,
                runs,
            );
            let (mse, rel_mse) =
                compute_error(&observed_freq, &exp_freq, theta_res, phi_res, sample_count);
            max_mse = max_mse.max(mse);
            max_rel_mse = max_rel_mse.max(rel_mse);
            {
                create_dir_all("test_output").unwrap();
                let desc = desc.replace(".", "_").replace(".", "_");
                let path = format!("test_output/{}_{}.py", desc, run + 1);
                let file = File::create(path).unwrap();
                dump_tables(file, &observed_freq, &exp_freq, theta_res, phi_res)
            }
            if !result.0 {
                println!(
                    "Test `{}` failed at run {}/{}, wo: {:?}\nReason: {}\nMSE: {:e}, rel MSE: {:e}",
                    desc,
                    run + 1,
                    runs,
                    [wo.x, wo.y, wo.z],
                    result.1,
                    mse,
                    rel_mse
                );
                has_error = true;
            }
        }
        if !has_error {
            println!(
                "Test `{}` passed. Max error: MSE: {:e}, relMSE {:e}",
                desc, max_mse, max_rel_mse
            );
        }
    }
    pub fn test(device: &Device) {
        let color_repr = color::ColorRepr::Rgb(color::RgbColorSpace::SRgb);
        test_bsdf(device, "Diffuse", false, move || {
            Box::new(DiffuseBsdf {
                reflectance: Color::one(color_repr),
            })
        });
        for roughness in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] {
            test_bsdf(
                device,
                &format!("TRRefl Roughness {}", roughness),
                false,
                move || {
                    Box::new(MicrofacetReflection {
                        color: Color::one(color_repr),
                        fresnel: Box::new(ConstFresnel {}),
                        dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                            Float2::expr(roughness, roughness),
                            false,
                        )),
                    })
                },
            );
        }
        for roughness in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] {
            test_bsdf(
                device,
                &format!("TRTrans Roughness {}", roughness),
                true,
                move || {
                    Box::new(MicrofacetTransmission {
                        color: Color::one(color_repr),
                        eta: 1.33f32.expr(),
                        fresnel: Box::new(ConstFresnel {}),
                        dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                            Float2::expr(roughness, roughness),
                            false,
                        )),
                    })
                },
            );
        }
    }
}
mod invert {
    use super::*;
    fn test_invert_helper(
        device: &Device,
        name: &str,
        sample: impl Fn(Expr<Float2>) -> Expr<Float3>,
        invert: impl Fn(Expr<Float3>) -> Expr<Float2>,
    ) {
        let count = 8192;
        let samples = 256;
        let rngs = init_pcg32_buffer_with_seed(device.clone(), count, 0);
        let bads = device.create_buffer::<u32>(count);
        bads.fill(0);
        let printer = Printer::new(&device, 32768);
        let kernel = device.create_kernel::<fn()>(&|| {
            let i = dispatch_id().x;
            let sampler = IndependentSampler::from_pcg32(rngs.var().read(i).var());
            for_range(0..samples, |_| {
                let u = sampler.next_2d();
                let w = sample(u);
                let u2 = invert(w);
                let bad = (u2 - u).length().gt(0.01);
                if_!(bad, {
                    bads.var().atomic_fetch_add(i, 1);
                    lc_info!(printer, "bad: u: {:?} u2:{:?} w:{:?}", u, u2, w);
                });
            });
        });
        let stream = device.default_stream();
        stream.with_scope(|s| {
            s.reset_printer(&printer)
                .submit([kernel.dispatch_async([count as u32, 1, 1])])
                .print(&printer);
        });

        let bads = bads.copy_to_vec();
        let bads = bads.iter().copied().sum::<u32>();
        println!("Test invert: `{}`, bad: {}", name, bads);
    }
    pub fn test_invert(device: &Device) {
        test_invert_helper(
            device,
            "cos_sample_hemisphere",
            cos_sample_hemisphere,
            invert_cos_sample_hemisphere,
        );
        let ax = 0.2;
        let ay = 0.3;
        test_invert_helper(
            device,
            "invert_ggx_iso",
            |u| {
                let dist = TrowbridgeReitzDistribution::from_alpha(Float2::expr(ax, ax), false);
                dist.sample_wh(Float3::expr(0.0, 1.0, 0.0), u, ADMode::None)
            },
            |w| {
                let dist = TrowbridgeReitzDistribution::from_alpha(Float2::expr(ax, ax), false);
                dist.invert_wh(Float3::expr(0.0, 1.0, 0.0), w, ADMode::None)
            },
        );
        test_invert_helper(
            device,
            "invert_ggx_aniso",
            |u| {
                let dist = TrowbridgeReitzDistribution::from_alpha(Float2::expr(ax, ay), false);
                dist.sample_wh(Float3::expr(0.0, 1.0, 0.0), u, ADMode::None)
            },
            |w| {
                let dist = TrowbridgeReitzDistribution::from_alpha(Float2::expr(ax, ay), false);
                dist.invert_wh(Float3::expr(0.0, 1.0, 0.0), w, ADMode::None)
            },
        );
    }
}
fn main() {
    let ctx = luisa::Context::new(current_exe().unwrap());
    let args = args().collect::<Vec<_>>();
    if args.len() != 3 {
        eprintln!("usage: {} <device> <test>", args[0]);
        exit(-1);
    }
    let device = if args.len() == 1 {
        ctx.create_device("cpu")
    } else {
        ctx.create_device(&args[1])
    };
    init_logger();
    let test = &args[2];
    match test.as_str() {
        "bsdf" => bsdf_chi2_test::test(&device),
        "invert" => invert::test_invert(&device),
        _ => {
            eprintln!("unknown test {}", test);
            exit(-1);
        }
    }
}
