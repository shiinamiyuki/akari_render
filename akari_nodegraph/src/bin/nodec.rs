use akari_nodegraph::*;
fn main() {
    let mut enums = vec![
        Enum {
            name: "CoordinateSystem".to_string(),
            variants: vec!["Blender".to_string(), "Akari".to_string()],
        },
        Enum {
            name: "ColorSpace".to_string(),
            variants: vec!["Linear".to_string(), "sRGB".to_string()],
        },
    ];
    let mut integrator_nodes = vec![NodeDesc {
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
            name: "color".to_string(),
            kind: SocketKind::Node("Image".to_string()),
        }],
    }];
    let graph = NodeGraphDesc {
        nodes: integrator_nodes,
        enums,
    };
    println!("{}", gen::gen_rust_for_nodegraph(&graph));
}
