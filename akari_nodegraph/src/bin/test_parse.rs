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
        NodeDesc {
            name: "DiffuseBsdf".to_string(),
            category: "Bsdf".to_string(),
            inputs: vec![
                InputSocketDesc {
                    name: "color".to_string(),
                    kind: SocketKind::Node("Color".to_string()),
                    default: SocketValue::Node(None),
                },
                InputSocketDesc {
                    name: "roughness".to_string(),
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
    let src = r#"color = RGB[r=0.0, g=0.0, b=0.0]
bsdf = DiffuseBsdf[color=color, roughness=0.1]
bsdf2 = DiffuseBsdf[color=color, roughness=0.2]
mix = MixBsdf[bsdf1=bsdf, bsdf2=bsdf2.bsdf, fac=0.5]
    "#;
    let graph = parse::parse(src, &graph).unwrap();
    let json = serde_json::to_string_pretty(&graph).unwrap();
    println!("{}", json);
    let mix = graph.get(&"mix".to_string()).unwrap();
    let mix = MixBsdf::from_node(&graph, mix).unwrap();
    dbg!(mix.in_fac);
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum CoordinateSystem {
    Blender,
    Akari,
}
impl akari_nodegraph::EnumProxy for CoordinateSystem {
    fn from_str(__s: &str) -> Self {
        match __s {
            stringify!(Blender) => Self::Blender,
            stringify!(Akari) => Self::Akari,
            _ => panic!("Invalid variant for enum #name: {}", __s),
        }
    }
    fn to_str(&self) -> &str {
        match self {
            CoordinateSystem::Blender => stringify!(Blender),
            CoordinateSystem::Akari => stringify!(Akari),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum ColorSpace {
    Linear,
    SRGB,
}
impl akari_nodegraph::EnumProxy for ColorSpace {
    fn from_str(__s: &str) -> Self {
        match __s {
            stringify!(Linear) => Self::Linear,
            stringify!(SRGB) => Self::SRGB,
            _ => panic!("Invalid variant for enum #name: {}", __s),
        }
    }
    fn to_str(&self) -> &str {
        match self {
            ColorSpace::Linear => stringify!(Linear),
            ColorSpace::SRGB => stringify!(SRGB),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum ColorPipeline {
    SRGB,
    ACEScg,
    Spectral,
}
impl akari_nodegraph::EnumProxy for ColorPipeline {
    fn from_str(__s: &str) -> Self {
        match __s {
            stringify!(SRGB) => Self::SRGB,
            stringify!(ACEScg) => Self::ACEScg,
            stringify!(Spectral) => Self::Spectral,
            _ => panic!("Invalid variant for enum #name: {}", __s),
        }
    }
    fn to_str(&self) -> &str {
        match self {
            ColorPipeline::SRGB => stringify!(SRGB),
            ColorPipeline::ACEScg => stringify!(ACEScg),
            ColorPipeline::Spectral => stringify!(Spectral),
        }
    }
}
#[derive(Clone, Debug)]
pub struct RGB {
    pub in_r: akari_nodegraph::NodeProxyInput<f64>,
    pub in_g: akari_nodegraph::NodeProxyInput<f64>,
    pub in_b: akari_nodegraph::NodeProxyInput<f64>,
    pub out_rgb: akari_nodegraph::NodeProxyOutput<Color>,
}
impl akari_nodegraph::NodeProxy for RGB {
    fn from_node(
        graph: &akari_nodegraph::NodeGraph,
        node: &akari_nodegraph::Node,
    ) -> Result<Self, akari_nodegraph::NodeProxyError> {
        if !node.isa(Self::ty()) {
            return Err(akari_nodegraph::NodeProxyError {
                msg: format!("Node is not of type {}", Self::ty()),
            });
        }
        let in_r = node.input("r")?.as_proxy_input::<f64>()?;
        let in_g = node.input("g")?.as_proxy_input::<f64>()?;
        let in_b = node.input("b")?.as_proxy_input::<f64>()?;
        let out_rgb = node.output("rgb")?.as_proxy_output::<Color>()?;
        Ok(Self {
            in_r,
            in_g,
            in_b,
            out_rgb,
        })
    }
    fn category() -> &'static str {
        stringify!("Texture")
    }
    fn ty() -> &'static str {
        stringify!(RGB)
    }
}
#[derive(Clone, Debug)]
pub struct DiffuseBsdf {
    pub in_color: akari_nodegraph::NodeProxyInput<Color>,
    pub in_roughness: akari_nodegraph::NodeProxyInput<f64>,
    pub out_bsdf: akari_nodegraph::NodeProxyOutput<Bsdf>,
}
impl akari_nodegraph::NodeProxy for DiffuseBsdf {
    fn from_node(
        graph: &akari_nodegraph::NodeGraph,
        node: &akari_nodegraph::Node,
    ) -> Result<Self, akari_nodegraph::NodeProxyError> {
        if !node.isa(Self::ty()) {
            return Err(akari_nodegraph::NodeProxyError {
                msg: format!("Node is not of type {}", Self::ty()),
            });
        }
        let in_color = node.input("color")?.as_proxy_input::<Color>()?;
        let in_roughness = node.input("roughness")?.as_proxy_input::<f64>()?;
        let out_bsdf = node.output("bsdf")?.as_proxy_output::<Bsdf>()?;
        Ok(Self {
            in_color,
            in_roughness,
            out_bsdf,
        })
    }
    fn category() -> &'static str {
        stringify!("Bsdf")
    }
    fn ty() -> &'static str {
        stringify!(DiffuseBsdf)
    }
}
#[derive(Clone, Debug)]
pub struct MixBsdf {
    pub in_bsdf1: akari_nodegraph::NodeProxyInput<Bsdf>,
    pub in_bsdf2: akari_nodegraph::NodeProxyInput<Bsdf>,
    pub in_fac: akari_nodegraph::NodeProxyInput<f64>,
    pub out_bsdf: akari_nodegraph::NodeProxyOutput<Bsdf>,
}
impl akari_nodegraph::NodeProxy for MixBsdf {
    fn from_node(
        graph: &akari_nodegraph::NodeGraph,
        node: &akari_nodegraph::Node,
    ) -> Result<Self, akari_nodegraph::NodeProxyError> {
        if !node.isa(Self::ty()) {
            return Err(akari_nodegraph::NodeProxyError {
                msg: format!("Node is not of type {}", Self::ty()),
            });
        }
        let in_bsdf1 = node.input("bsdf1")?.as_proxy_input::<Bsdf>()?;
        let in_bsdf2 = node.input("bsdf2")?.as_proxy_input::<Bsdf>()?;
        let in_fac = node.input("fac")?.as_proxy_input::<f64>()?;
        let out_bsdf = node.output("bsdf")?.as_proxy_output::<Bsdf>()?;
        Ok(Self {
            in_bsdf1,
            in_bsdf2,
            in_fac,
            out_bsdf,
        })
    }
    fn category() -> &'static str {
        stringify!("Bsdf")
    }
    fn ty() -> &'static str {
        stringify!(MixBsdf)
    }
}
#[derive(Clone, Debug)]
pub struct Color;
impl akari_nodegraph::SocketType for Color {
    fn is_primitive() -> bool {
        false
    }
    fn ty() -> &'static str {
        stringify!(Color)
    }
}
#[derive(Clone, Debug)]
pub struct Bsdf;
impl akari_nodegraph::SocketType for Bsdf {
    fn is_primitive() -> bool {
        false
    }
    fn ty() -> &'static str {
        stringify!(Bsdf)
    }
}
