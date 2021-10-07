use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use akari::accel::Aggregate;
use akari::bsdf::NullBsdf;
use akari::camera::PerspectiveCamera;
// use akari::film::Film;
use akari::integrator::ao::RTAO;
use akari::integrator::Integrator;
use akari::light::Light;
use akari::light::PointLight;
use akari::light::UniformLightDistribution;
use akari::scene::Scene;
use akari::shape::Shape;
use akari::shape::Sphere;
use akari::texture::ConstantTexture;
use akari::*;
use rand::Rng;
fn main() {
    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(1)
    //     .build_global()
    //     .unwrap();
    let camera = {
        let m = glm::translate(&glm::identity(), &vec3(0.5, 0.5, 2.0));
        Arc::new(PerspectiveCamera::new(
            &uvec2(2048, 2048),
            &Transform::from_matrix(&m),
            (40.0 as Float).to_radians(),
        ))
    };
    let mut rng = rand::thread_rng();
    let shapes: Vec<_> = (0..1000000)
        .map(|_| {
            let x: Float = rng.gen();
            let y: Float = rng.gen();
            let z: Float = rng.gen();
            Arc::new(Sphere {
                center: vec3(x, y, z),
                radius: 0.005,
                bsdf: Arc::new(NullBsdf {}),
            }) as Arc<dyn Shape>
        })
        .collect();
    let shape = Arc::new(Aggregate::new(shapes));
    let lights = vec![Arc::new(PointLight {
        position: vec3(0.0, 2.0, 0.0),
        emission: Arc::new(ConstantTexture::<Spectrum> {
            value: Spectrum::from_srgb(&vec3(2.0, 2.0, 2.0)),
        }),
    }) as Arc<dyn Light>];
    let scene = Scene {
        shape_to_light: HashMap::new(),
        ray_counter: AtomicU64::new(0),
        lights: lights.clone(),
        light_distr: Arc::new(UniformLightDistribution::new(lights.clone())),
        shape,
        camera,
        meshes:vec![],
    };
    let mut integrator = RTAO { spp: 1 };
    let (film, time) = profile(|| integrator.render(&scene));
    println!(
        "took {}s, {}M Rays/s",
        time,
        scene.ray_counter.load(Ordering::Relaxed) as f64 / 1e6 / time
    );
    let img = film.to_rgb_image();
    img.save("bench.png").unwrap();
}
