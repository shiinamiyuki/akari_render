use akari::color::glam_linear_to_srgb;
use akari::scenegraph::node;
use akari::scenegraph::node::LookAt;
use akari::util::binserde;
use akari::*;
use akari_render as akari;
use clap::Parser;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::path::Path;
use std::process::exit;
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    scene: String,

    #[arg(short, long)]
    model: String,

    #[arg(short, long, default_value_t = false)]
    force: bool,
}

fn import(path: &str, scene: &mut node::Scene, forced: bool, generate_normal: Option<f32>) {
    let (imported_models, models, materials) = akari::mesh::load_model(path, generate_normal);
    let mut cvt_mat: HashMap<String, node::Bsdf> = HashMap::new();
    let mut cvt_names = vec![];
    println!("# of models: {}", models.len());
    println!("# of materials: {}", materials.len());
    let default_bsdf = node::Bsdf::Principled {
        color: node::Texture::SRgb {
            values: [0.8, 0.8, 0.8],
        },
        subsurface_radius: node::Texture::SRgbLinear {
            values: [1.0, 0.2, 0.1],
        },
        subsurface: node::Texture::Float { value: 0.0 },
        subsurface_ior: node::Texture::Float { value: 1.45 },
        subsurface_color: node::Texture::SRgb {
            values: [0.0, 0.0, 0.0],
        },
        metallic: node::Texture::Float { value: 0.0 },
        specular_tint: node::Texture::Float { value: 0.0 },
        roughness: node::Texture::Float { value: 0.4 },
        anisotropic: node::Texture::Float { value: 0.0 },
        anisotropic_rotation: node::Texture::Float { value: 0.0 },
        sheen: node::Texture::Float { value: 0.0 },
        sheen_tint: node::Texture::Float { value: 0.5 },
        clearcoat: node::Texture::Float { value: 0.0 },
        clearcoat_roughness: node::Texture::Float { value: 0.03 },
        ior: 1.45,
        transmission: node::Texture::Float { value: 0.0 },
        emission: node::Texture::SRgbLinear {
            values: [0.0, 0.0, 0.0],
        },
    };
    for (i, m) in materials.iter().enumerate() {
        let name = if m.name.is_empty() {
            let path = Path::new(path);
            println!("warning: material with empty name");
            String::from(format!(
                "{}_{}",
                path.file_stem().unwrap().to_str().unwrap(),
                i
            ))
        } else {
            m.name.clone()
        };
        cvt_names.push(name.clone());
        if cvt_mat.contains_key(&name) {
            println!("duplicated material name!");
        }
        let mut bsdf = default_bsdf.clone();
        match &mut bsdf {
            node::Bsdf::Principled {
                ref mut color,
                ref mut roughness,
                ref mut metallic,
                ..
            } => {
                let has_diffuse = {
                    if m.diffuse_texture.is_empty() {
                        m.diffuse.iter().any(|x| *x > 0.0)
                    } else {
                        true
                    }
                };
                let has_specular = {
                    if m.specular_texture.is_empty() {
                        m.specular.iter().any(|x| *x > 0.0)
                    } else {
                        true
                    }
                };
                *roughness = node::Texture::Float {
                    value: (2.0 / (m.shininess + 2.0)).sqrt(),
                };
                let mut max_specular = 1.0;
                let mut max_diffuse = 1.0;
                if has_diffuse {
                    if m.diffuse_texture.is_empty() {
                        *color = node::Texture::SRgb {
                            values: glam_linear_to_srgb(glam::vec3(
                                m.diffuse[0],
                                m.diffuse[1],
                                m.diffuse[2],
                            ))
                            .into(),
                        };
                        max_diffuse = m
                            .diffuse
                            .iter()
                            .map(|x| *x)
                            .reduce(|a, b| a.max(b))
                            .unwrap();
                    } else {
                        *color = node::Texture::Image {
                            path: m.diffuse_texture.clone().replace("\\", "/"),
                            colorspace: "srgb".into(),
                        };
                    }
                } else if has_specular {
                    if m.specular_texture.is_empty() {
                        *color = node::Texture::SRgb {
                            values: glam_linear_to_srgb(glam::vec3(
                                m.specular[0],
                                m.specular[1],
                                m.specular[2],
                            ))
                            .into(),
                        };
                        max_specular = m
                            .specular
                            .iter()
                            .map(|x| *x)
                            .reduce(|a, b| a.max(b))
                            .unwrap();
                    } else {
                        *color = node::Texture::Image {
                            path: m.specular_texture.clone().replace("\\", "/"),
                            colorspace: "srgb".into(),
                        };
                    }
                    *metallic = node::Texture::Float { value: 1.0 };
                } else {
                    *color = node::Texture::SRgb {
                        values: [0.0, 0.0, 0.0],
                    };
                }
                if has_diffuse && has_specular && (max_specular + max_diffuse > 0.0) {
                    *metallic = node::Texture::Float {
                        value: max_specular / (max_specular + max_diffuse),
                    };
                }
            }
            _ => unreachable!(),
        };
        cvt_mat.insert(name.clone(), bsdf);
    }
    let mut cvt_models = vec![];
    for (i, m) in models.iter().enumerate() {
        let mesh = &m.mesh;
        let path = Path::new(path);
        let model_name = format!(
            "{}_{}_{}.mesh",
            path.file_stem().unwrap().to_str().unwrap(),
            m.name,
            i
        );
        let model_path = path.parent().unwrap().join(model_name.clone());
        // {
        //     let bson_data = bson::to_document(&imported_models[i]).unwrap();
        //     let mut file = File::create(model_path).unwrap();
        //     bson_data.to_writer(&mut file).unwrap();
        // }
        {
            let mut file = File::create(model_path).unwrap();
            binserde::Encode::encode(&imported_models[i], &mut file).unwrap();
        }
        let j: node::Shape = if let Some(id) = mesh.material_id {
            node::Shape::Mesh {
                path: model_name,
                bsdf: cvt_names[id].clone(),
                transform: None,
            }
        } else {
            node::Shape::Mesh {
                path: model_name,
                bsdf: "".into(),
                transform: None,
            }
        };
        cvt_models.push(j)
    }
    {
        let named = &mut scene.bsdfs;
        for (k, v) in cvt_mat.iter() {
            if let Some(_) = named.insert(k.clone(), v.clone()) {
                println!("warning! overrided previous material");
                if forced {
                    println!("force mode, overriding");
                } else {
                    println!("exiting, pass -f to force override");
                    exit(1);
                }
            }
        }
    }
    {
        for shape in cvt_models.into_iter() {
            scene.shapes.push(shape);
        }
    }
}
fn main() {
    let args = Args::parse();
    let force = args.force;
    let path = &args.scene;
    let model = &args.model;
    let mut scene: node::Scene = {
        match File::open(path) {
            Ok(file) => {
                let reader = BufReader::new(file);
                let json = serde_json::from_reader(reader).unwrap();
                serde_json::from_value(json).unwrap()
            }
            Err(_) => node::Scene {
                bsdfs: HashMap::new(),
                lights: vec![],
                shapes: vec![],
                camera: node::Camera::Perspective {
                    res: (512, 512),
                    fov: 80.0,
                    focal: 1.0,
                    lens_radius: 0.0,
                    transform: node::Transform::LookAt(LookAt {
                        eye: [0.0, 0.0, 0.0],
                        center: [0.0, 0.0, -1.0],
                        up: [0.0, 1.0, 0.0],
                    }),
                },
            },
        }
    };
    import(&model, &mut scene, force, None);
    {
        let file = File::create(path).unwrap();
        let json = serde_json::to_value(&scene).unwrap();
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &json).unwrap();
    }
}
