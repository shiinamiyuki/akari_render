use akari_nodegraph::*;

use std::process::Command;
fn gen_nodegraph_defs() {
    let coordinate_system = Enum {
        name: "CoordinateSystem".to_string(),
        variants: vec!["Blender".to_string(), "Akari".to_string()],
    };
    let colorspace = Enum {
        name: "ColorSpace".to_string(),
        variants: vec!["ACEScg".to_string(), "SRGB".to_string()],
    };
    let color_pipeline = Enum {
        name: "ColorPipeline".to_string(),
        variants: vec![
            "SRGB".to_string(),
            "ACEScg".to_string(),
            "Spectral".to_string(),
        ],
    };
    let texture_nodes = vec![
        NodeDesc {
            name: "RGBImageTexture".to_string(),
            category: "RGBTexture".to_string(),
            inputs: vec![
                InputSocketDesc {
                    name: "path".to_string(),
                    kind: SocketKind::String,
                    default: SocketValue::String("".to_string()),
                },
                InputSocketDesc {
                    name: "colorspace".to_string(),
                    kind: SocketKind::Enum(colorspace.name.clone()),
                    default: SocketValue::Enum("SRGB".to_string()),
                },
            ],
            outputs: vec![OutputSocketDesc {
                name: "rgb".to_string(),
                kind: SocketKind::Node("RGBTexture".to_string()),
            }],
        },
        NodeDesc {
            name: "RGB".to_string(),
            category: "RGBTexture".to_string(),
            inputs: vec![
                InputSocketDesc {
                    name: "r".to_string(),
                    kind: SocketKind::Float,
                    default: SocketValue::Float(0.0),
                },
                InputSocketDesc {
                    name: "g".to_string(),
                    kind: SocketKind::Float,
                    default: SocketValue::Float(0.0),
                },
                InputSocketDesc {
                    name: "b".to_string(),
                    kind: SocketKind::Float,
                    default: SocketValue::Float(0.0),
                },
                InputSocketDesc {
                    name: "colorspace".to_string(),
                    kind: SocketKind::Enum(colorspace.name.clone()),
                    default: SocketValue::Enum("SRGB".to_string()),
                },
            ],
            outputs: vec![OutputSocketDesc {
                name: "rgb".to_string(),
                kind: SocketKind::Node("RGBTexture".to_string()),
            }],
        },
        NodeDesc {
            name: "MixRGB".to_string(),
            category: "RGBTexture".to_string(),
            inputs: vec![
                InputSocketDesc {
                    name: "rgb1".to_string(),
                    kind: SocketKind::Node("RGBTexture".to_string()),
                    default: SocketValue::Node(None),
                },
                InputSocketDesc {
                    name: "rgb2".to_string(),
                    kind: SocketKind::Node("RGBTexture".to_string()),
                    default: SocketValue::Node(None),
                },
                InputSocketDesc {
                    name: "fac".to_string(),
                    kind: SocketKind::Float,
                    default: SocketValue::Float(0.0),
                },
            ],
            outputs: vec![OutputSocketDesc {
                name: "rgb".to_string(),
                kind: SocketKind::Node("RGBTexture".to_string()),
            }],
        },
        NodeDesc {
            name: "SpectralUplift".to_string(),
            category: "SpectrumTexture".to_string(),
            inputs: vec![InputSocketDesc {
                name: "rgb".to_string(),
                kind: SocketKind::Node("RGBTexture".to_string()),
                default: SocketValue::Node(None),
            }],
            outputs: vec![OutputSocketDesc {
                name: "texture".to_string(),
                kind: SocketKind::Node("SpectrumTexture".to_string()),
            }],
        },
    ];
    let create_float_input = |name: &str, default: f64| InputSocketDesc {
        name: name.to_string(),
        kind: SocketKind::Float,
        default: SocketValue::Float(default),
    };
    let create_int_input = |name: &str, default: i64| InputSocketDesc {
        name: name.to_string(),
        kind: SocketKind::Int,
        default: SocketValue::Int(default),
    };
    let create_float3_input = |name: &str, default: [f32; 3]| InputSocketDesc {
        name: name.to_string(),
        kind: SocketKind::List(Box::new(SocketKind::Float), Some(3)),
        default: SocketValue::List(vec![
            SocketValue::Float(default[0] as f64),
            SocketValue::Float(default[1] as f64),
            SocketValue::Float(default[2] as f64),
        ]),
    };
    let create_spectrum_input = |name: &str| InputSocketDesc {
        name: name.to_string(),
        kind: SocketKind::Node("SpectrumTexture".to_string()),
        default: SocketValue::Node(None),
    };
    let create_string_input = |name: &str, default: &str| InputSocketDesc {
        name: name.to_string(),
        kind: SocketKind::String,
        default: SocketValue::String(default.to_string()),
    };
    let vec_math_nodes = vec![NodeDesc {
        name: "Vec3".to_string(),
        category: "Bsdf".to_string(),
        inputs: vec![
            create_float_input("x", 0.2),
            create_float_input("y", 0.2),
            create_float_input("z", 0.2),
        ],
        outputs: vec![OutputSocketDesc {
            name: "bsdf".to_string(),
            kind: SocketKind::Node("Float3".to_string()),
        }],
    }];
    let transform_nodes = vec![
        NodeDesc {
            name: "MatrixTransform".to_string(),
            category: "Transform".to_string(),
            inputs: vec![InputSocketDesc {
                name: "matrix".to_string(),
                kind: SocketKind::List(Box::new(SocketKind::Float), Some(16)),
                default: SocketValue::List(
                    [
                        1.0, 0.0, 0.0, 0.0, // row 0
                        0.0, 1.0, 0.0, 0.0, //
                        0.0, 0.0, 1.0, 0.0, //
                        0.0, 0.0, 0.0, 1.0, //
                    ]
                    .into_iter()
                    .map(|x| SocketValue::Float(x))
                    .collect(),
                ),
            }],
            outputs: vec![OutputSocketDesc {
                name: "transform".to_string(),
                kind: SocketKind::Node("Transform".to_string()),
            }],
        },
        NodeDesc {
            name: "TRS".to_string(),
            category: "Transform".to_string(),
            inputs: vec![
                create_float3_input("translation", [0.0, 0.0, 0.0]),
                create_float3_input("rotation", [0.0, 0.0, 0.0]),
                create_float3_input("scale", [1.0, 1.0, 1.0]),
                InputSocketDesc {
                    name: "coordinate_system".to_string(),
                    kind: SocketKind::Enum(coordinate_system.name.clone()),
                    default: SocketValue::Enum("Blender".to_string()),
                },
            ],
            outputs: vec![OutputSocketDesc {
                name: "transform".to_string(),
                kind: SocketKind::Node("Transform".to_string()),
            }],
        },
    ];
    let camera_nodes = vec![NodeDesc {
        name: "PerspectiveCamera".into(),
        category: "Camera".into(),
        inputs: vec![
            InputSocketDesc {
                name: "transform".to_string(),
                kind: SocketKind::Node("Transform".into()),
                default: SocketValue::Node(None),
            },
            create_float_input("fov", 45.0),
            create_float_input("focal_distance", 0.0),
            create_float_input("fstop", 2.8),
            create_int_input("width", 512),
            create_int_input("height", 512),
        ],
        outputs: vec![OutputSocketDesc {
            name: "camera".to_string(),
            kind: SocketKind::Node("Camera".into()),
        }],
    }];
    let bsdf_nodes = vec![
        NodeDesc {
            name: "PrincipledBsdf".to_string(),
            category: "Bsdf".to_string(),
            inputs: vec![
                create_spectrum_input("color"),
                create_float_input("roughness", 0.2),
                create_float_input("metallic", 0.0),
                create_float_input("specular", 0.5),
                create_float_input("clearcoat", 0.0),
                create_float_input("clearcoat_roughness", 0.0),
                create_float_input("transmission", 0.0),
                create_float_input("ior", 1.333),
                create_spectrum_input("emission"),
            ],
            outputs: vec![OutputSocketDesc {
                name: "bsdf".to_string(),
                kind: SocketKind::Node("Bsdf".to_string()),
            }],
        },
        NodeDesc {
            name: "DiffuseBsdf".to_string(),
            category: "Bsdf".to_string(),
            inputs: vec![create_spectrum_input("color")],
            outputs: vec![OutputSocketDesc {
                name: "bsdf".to_string(),
                kind: SocketKind::Node("Bsdf".to_string()),
            }],
        },
        NodeDesc {
            name: "MixBsdf".to_string(),
            category: "Bsdf".to_string(),
            inputs: vec![
                InputSocketDesc {
                    name: "bsdf1".to_string(),
                    kind: SocketKind::Node("Bsdf".to_string()),
                    default: SocketValue::Node(None),
                },
                InputSocketDesc {
                    name: "bsdf2".to_string(),
                    kind: SocketKind::Node("Bsdf".to_string()),
                    default: SocketValue::Node(None),
                },
                InputSocketDesc {
                    name: "fac".to_string(),
                    kind: SocketKind::Float,
                    default: SocketValue::Float(0.0),
                },
            ],
            outputs: vec![OutputSocketDesc {
                name: "bsdf".to_string(),
                kind: SocketKind::Node("Bsdf".to_string()),
            }],
        },
        NodeDesc {
            name: "MaterialOutput".to_string(),
            category: "Material".to_string(),
            inputs: vec![InputSocketDesc {
                name: "surface".to_string(),
                kind: SocketKind::Node("Bsdf".to_string()),
                default: SocketValue::Node(None),
            }],
            outputs: vec![OutputSocketDesc {
                name: "material".to_string(),
                kind: SocketKind::Node("Material".to_string()),
            }],
        },
    ];
    let light_nodes = vec![
        NodeDesc {
            name: "LightOutput".to_string(),
            category: "Light".to_string(),
            inputs: vec![InputSocketDesc {
                name: "surface".to_string(),
                kind: SocketKind::Node("Bsdf".to_string()),
                default: SocketValue::Node(None),
            }],
            outputs: vec![OutputSocketDesc {
                name: "light".to_string(),
                kind: SocketKind::Node("LightOutput".to_string()),
            }],
        },
        NodeDesc {
            name: "AreaLight".to_string(),
            category: "Light".to_string(),
            inputs: vec![InputSocketDesc {
                name: "emission".to_string(),
                kind: SocketKind::Node("LightOutput".to_string()),
                default: SocketValue::Node(None),
            }],
            outputs: vec![OutputSocketDesc {
                name: "light".to_string(),
                kind: SocketKind::Node("Light".to_string()),
            }],
        },
    ];
    let mesh_node = NodeDesc {
        name: "Mesh".to_string(),
        category: "Geometry".to_string(),
        inputs: vec![
            create_string_input("name", ""),
            InputSocketDesc {
                name: "buffers".to_string(),
                kind: SocketKind::List(Box::new(SocketKind::Node("Buffer".to_string())), None),
                default: SocketValue::Node(None),
            },
        ],
        outputs: vec![OutputSocketDesc {
            name: "mesh".to_string(),
            kind: SocketKind::Node("Geometry".to_string()),
        }],
    };
    let instance_node = NodeDesc {
        name: "Instance".to_string(),
        category: "Instance".to_string(),
        inputs: vec![
            create_string_input("name", ""),
            InputSocketDesc {
                name: "geometry".to_string(),
                kind: SocketKind::Node("Geometry".to_string()),
                default: SocketValue::Node(None),
            },
            InputSocketDesc {
                name: "material".to_string(),
                kind: SocketKind::Node("Material".to_string()),
                default: SocketValue::Node(None),
            },
            InputSocketDesc {
                name: "transform".to_string(),
                kind: SocketKind::Node("Transform".to_string()),
                default: SocketValue::Node(None),
            },
        ],
        outputs: vec![OutputSocketDesc {
            name: "instance".to_string(),
            kind: SocketKind::Node("Instance".to_string()),
        }],
    };
    let scene_nodes = vec![NodeDesc {
        name: "Scene".to_string(),
        category: "Scene".to_string(),
        inputs: vec![
            InputSocketDesc {
                name: "instances".to_string(),
                kind: SocketKind::List(Box::new(SocketKind::Node("Instance".to_string())), None),
                default: SocketValue::List(vec![]),
            },
            InputSocketDesc {
                name: "lights".to_string(),
                kind: SocketKind::List(Box::new(SocketKind::Node("Light".to_string())), None),
                default: SocketValue::List(vec![]),
            },
            InputSocketDesc {
                name: "camera".to_string(),
                kind: SocketKind::Node("Camera".to_string()),
                default: SocketValue::Node(None),
            },
        ],
        outputs: vec![OutputSocketDesc {
            name: "scene".to_string(),
            kind: SocketKind::Node("Scene".to_string()),
        }],
    }];
    let misc_nodes = vec![
        NodeDesc {
            name: "Float".to_string(),
            category: "Math".to_string(),
            inputs: vec![InputSocketDesc {
                name: "value".to_string(),
                kind: SocketKind::Float,
                default: SocketValue::Float(0.0),
            }],
            outputs: vec![OutputSocketDesc {
                name: "float".to_string(),
                kind: SocketKind::Float,
            }],
        },
        NodeDesc {
            name: "Buffer".to_string(),
            category: "Misc".to_string(),
            inputs: vec![
                create_string_input("name", ""),
                create_string_input("path", ""),
            ],
            outputs: vec![OutputSocketDesc {
                name: "buffer".to_string(),
                kind: SocketKind::Node("Buffer".to_string()),
            }],
        },
    ];
    let enums = vec![coordinate_system, colorspace, color_pipeline];
    let mut nodes = vec![];
    nodes.extend(texture_nodes);
    nodes.extend(bsdf_nodes);
    nodes.extend(light_nodes);
    nodes.extend(scene_nodes);
    nodes.extend(misc_nodes);
    nodes.extend(camera_nodes);
    nodes.extend(transform_nodes);
    nodes.extend_from_slice(&[mesh_node, instance_node]);
    let graph = NodeGraphDesc { nodes, enums };
    std::fs::write("src/nodes.rs", gen::gen_rust_for_nodegraph(&graph)).unwrap();
    // format the generated code
    let output = Command::new("rustfmt")
        .arg("src/nodes.rs")
        .output()
        .expect("failed to execute rustfmt");
    if !output.status.success() {
        panic!(
            "rustfmt failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let graph_json = serde_json::to_string_pretty(&graph).unwrap();
    std::fs::write("src/nodes.json", graph_json).unwrap();
}
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cpp_ext");
    // gen_rgb2spec();
    let out = cmake::Config::new("cpp_ext")
        .generator("Ninja")
        .define("CMAKE_BUILD_TYPE", "Release")
        .no_build_target(true)
        .build();
    dbg!(out.display());
    println!("cargo:rustc-link-search=native={}/build", out.display());
    println!("cargo:rustc-link-lib=static=akari_cpp_ext");
    gen_nodegraph_defs();
}
