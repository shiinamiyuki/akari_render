use crate::*;
use indexmap::IndexMap;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
fn gen_rust_for_desc(
    desc: &NodeDesc,
    categories: &mut IndexMap<String, Vec<String>>,
) -> TokenStream {
    let name = &desc.name;
    let category = &desc.category;
    if category != "" {
        categories
            .entry(category.clone())
            .or_default()
            .push(name.clone());
    } else {
        categories.insert(name.clone(), vec![]);
    }
    let name = format_ident!("{}", name);
    fn gen_socket_kind(kind: &SocketKind) -> TokenStream {
        match kind {
            SocketKind::Float => quote!(akari_nodegraph::NodeProxyInput<f64>),
            SocketKind::Int => quote!(akari_nodegraph::NodeProxyInput<i64>),
            SocketKind::Bool => quote!(akari_nodegraph::NodeProxyInput<bool>),
            SocketKind::String => quote!(akari_nodegraph::NodeProxyInput<String>),
            SocketKind::Enum(Enum { name, .. }) => {
                let name = format_ident!("{}", name);
                quote!(#name)
            }
            SocketKind::Node(t) => {
                let t = format_ident!("{}", t);
                quote!(akari_nodegraph::NodeProxyRef<#t>)
            }
            SocketKind::List(inner) => {
                let inner = gen_socket_kind(inner);
                quote!(Vec<#inner>)
            }
        }
    }
    let inputs = desc
        .inputs
        .iter()
        .map(|i| {
            let name = format_ident!("in_{}", i.name);
            let kind = &i.kind;
            let ty = gen_socket_kind(kind);
            quote! {
                pub #name: #ty,
            }
        })
        .collect::<Vec<_>>();
    quote! {
        #[derive(Clone, Debug)]
        pub struct #name {
            #(#inputs)*
        }
        impl akari_nodegraph::NodeProxy for #name {
            fn from_node(graph: &akari_nodegraph::NodeGraph, node: &akari_nodegraph::Node) -> Self {
                todo!()
            }
            fn to_node(&self, graph: & akari_nodegraph::NodeGraph) -> akari_nodegraph::Node {
                todo!()
            }
            fn category() -> &'static str {
                stringify!(#category)
            }
        }
    }
}

fn gen_rust_for_enum(desc: &Enum) -> TokenStream {
    let name = format_ident!("{}", desc.name);
    let variants = desc
        .variants
        .iter()
        .map(|v| format_ident!("{}", v))
        .collect::<Vec<_>>();
    quote! {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
        pub enum #name {
            #(#variants,)*
        }
        impl akari_nodegraph::EnumProxy for #name {
            fn from_str(__s: &str) -> Self {
                match __s {
                    #(stringify!(#variants) => Self::#variants,)*
                    _ => panic!("Invalid variant for enum #name: {}", __s),
                }
            }
            fn to_str(&self) -> &str {
                match self {
                    #(#name::#variants => stringify!(#variants),)*
                }
            }
        }
    }
}
fn gen_rust_for_enums(descs: &[Enum]) -> TokenStream {
    let mut tokens = TokenStream::new();
    for desc in descs {
        tokens.extend(gen_rust_for_enum(desc));
    }
    tokens
}
fn gen_rust_for_descs(descs: &[NodeDesc]) -> TokenStream {
    let mut tokens = TokenStream::new();
    let mut categories = IndexMap::new();
    for desc in descs {
        tokens.extend(gen_rust_for_desc(desc, &mut categories));
    }
    let categories = categories
        .into_iter()
        .map(|(k, v)| {
            let k = format_ident!("{}", k);
            if v.is_empty() {
                return quote! {
                    impl akari_nodegraph::CategoryProxy for #k {}
                };
            }
            let vs = v.iter().map(|v| format_ident!("{}", v)).collect::<Vec<_>>();
            quote! {
                #[derive(Clone, Debug)]
                pub enum #k {
                    #(#vs(#vs),)*
                }
                impl akari_nodegraph::CategoryProxy for #k {}
            }
        })
        .collect::<Vec<_>>();
    quote! {
        #(#categories)*
        #tokens
    }
}

pub fn gen_rust_for_nodegraph(desc: &NodeGraphDesc) -> String {
    let enums = &desc.enums;
    let nodes = &desc.nodes;
    let enums = gen_rust_for_enums(enums);
    let nodes = gen_rust_for_descs(nodes);
    quote! {
        #enums
        #nodes
    }
    .to_string()
}
