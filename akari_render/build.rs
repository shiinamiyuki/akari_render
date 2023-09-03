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
    let integrator_nodes = vec![NodeDesc {
        name: "PathTracer".to_string(),
        category: "Integrator".to_string(),
        inputs: vec![
            InputSocketDesc {
                name: "min_depth".to_string(),
                kind: SocketKind::Int,
                default: SocketValue::Int(5),
            },
            InputSocketDesc {
                name: "max_depth".to_string(),
                kind: SocketKind::Int,
                default: SocketValue::Int(7),
            },
        ],
        outputs: vec![OutputSocketDesc {
            name: "integrator".to_string(),
            kind: SocketKind::Node("Integrator".to_string()),
        }],
    }];
    let render_node = NodeDesc {
        name: "Render".to_string(),
        category: "RenderOutput".to_string(),
        inputs: vec![
            InputSocketDesc {
                name: "scene".to_string(),
                kind: SocketKind::Node("Scene".to_string()),
                default: SocketValue::Node(None),
            },
            InputSocketDesc {
                name: "integrator".to_string(),
                kind: SocketKind::Node("Integrator".to_string()),
                default: SocketValue::Node(None),
            },
            InputSocketDesc {
                name: "width".to_string(),
                kind: SocketKind::Int,
                default: SocketValue::Int(800),
            },
            InputSocketDesc {
                name: "height".to_string(),
                kind: SocketKind::Int,
                default: SocketValue::Int(600),
            },
            InputSocketDesc {
                name: "color".to_string(),
                kind: SocketKind::Enum(color_pipeline.name.clone()),
                default: SocketValue::Enum("SRGB".to_string()),
            },
            create_float_input("max_time", 0.0),
        ],
        outputs: vec![OutputSocketDesc {
            name: "image".to_string(),
            kind: SocketKind::Node("Image".to_string()),
        }],
    };
    let light_nodes = vec![NodeDesc {
        name: "AreaLight".to_string(),
        category: "Light".to_string(),
        inputs: vec![
            create_spectrum_input("color"),
            create_float_input("intensity", 1.0),
        ],
        outputs: vec![OutputSocketDesc {
            name: "light".to_string(),
            kind: SocketKind::Node("Light".to_string()),
        }],
    }];
    let mesh_node = NodeDesc {
        name: "Mesh".to_string(),
        category: "Geometry".to_string(),
        inputs: vec![
            create_string_input("name", ""),
            InputSocketDesc {
                name: "buffers".to_string(),
                kind: SocketKind::List(Box::new(SocketKind::Node("Buffer".to_string()))),
                default: SocketValue::Node(None),
            },
            InputSocketDesc {
                name: "material".to_string(),
                kind: SocketKind::Node("Material".to_string()),
                default: SocketValue::Node(None),
            },
        ],
        outputs: vec![OutputSocketDesc {
            name: "mesh".to_string(),
            kind: SocketKind::Node("Geometry".to_string()),
        }],
    };
    let scene_nodes = vec![NodeDesc {
        name: "Scene".to_string(),
        category: "Scene".to_string(),
        inputs: vec![
            InputSocketDesc {
                name: "geometries".to_string(),
                kind: SocketKind::List(Box::new(SocketKind::Node("Geometry".to_string()))),
                default: SocketValue::List(vec![]),
            },
            InputSocketDesc {
                name: "lights".to_string(),
                kind: SocketKind::List(Box::new(SocketKind::Node("Light".to_string()))),
                default: SocketValue::List(vec![]),
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
    nodes.extend(integrator_nodes);
    nodes.extend(light_nodes);
    nodes.extend(scene_nodes);
    nodes.extend(misc_nodes);
    nodes.extend_from_slice(&[mesh_node, render_node]);
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
