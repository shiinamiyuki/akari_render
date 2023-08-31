use akari_nodegraph::*;
fn main() {
    let coordinate_system = Enum {
        name: "CoordinateSystem".to_string(),
        variants: vec!["Blender".to_string(), "Akari".to_string()],
    };
    let colorspace = Enum {
        name: "ColorSpace".to_string(),
        variants: vec!["Linear".to_string(), "SRGB".to_string()],
    };
    let color_pipeline = Enum {
        name: "ColorPipeline".to_string(),
        variants: vec![
            "SRGB".to_string(),
            "ACEScg".to_string(),
            "Spectral".to_string(),
        ],
    };
    let enums = vec![coordinate_system, colorspace, color_pipeline];
    let nodes = vec![
        NodeDesc {
            name: "RGB".to_string(),
            category: "Texture".to_string(),
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
            ],
            outputs: vec![OutputSocketDesc {
                name: "rgb".to_string(),
                kind: SocketKind::Node("Color".to_string()),
            }],
        },
        NodeDesc{
            name:"DiffuseBsdf".to_string(),
            category:"Bsdf".to_string(),
            inputs:vec![
                InputSocketDesc{
                    name:"color".to_string(),
                    kind:SocketKind::Node("Color".to_string()),
                    default:SocketValue::Node(None),
                },
                InputSocketDesc{
                    name:"roughness".to_string(),
                    kind:SocketKind::Float,
                    default:SocketValue::Float(0.0),
                },
            ],
            outputs:vec![
                OutputSocketDesc{
                    name:"bsdf".to_string(),
                    kind:SocketKind::Node("Bsdf".to_string()),
                }
            ]
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
    ];
    let graph = NodeGraphDesc { nodes, enums };
    println!("{}", gen::gen_rust_for_nodegraph(&graph));
}
