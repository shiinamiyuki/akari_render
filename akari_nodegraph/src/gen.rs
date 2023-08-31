use std::collections::HashSet;

use crate::*;
use indexmap::IndexSet;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
fn gen_rust_for_desc(desc: &NodeDesc, socket_types: &mut IndexSet<String>) -> TokenStream {
    let name = &desc.name;
    let category = &desc.category;
    assert!(category.len() > 0, "Category cannot be empty");
    let name = format_ident!("{}", name);
    fn gen_socket_input(kind: &SocketKind, socket_types: &mut IndexSet<String>) -> TokenStream {
        match kind {
            SocketKind::Float => quote!(akari_nodegraph::NodeProxyInput<f64>),
            SocketKind::Int => quote!(akari_nodegraph::NodeProxyInput<i64>),
            SocketKind::Bool => quote!(akari_nodegraph::NodeProxyInput<bool>),
            SocketKind::String => quote!(akari_nodegraph::NodeProxyInput<String>),
            SocketKind::Enum(name) => {
                let name = format_ident!("{}", name);
                quote!(#name)
            }
            SocketKind::Node(t) => {
                socket_types.insert(t.clone());
                let t = format_ident!("{}", t);
                quote!(akari_nodegraph::NodeProxyInput<#t>)
            }
            SocketKind::List(inner) => {
                let inner = gen_socket_input(inner, socket_types);
                quote!(Vec<#inner>)
            }
        }
    }
    fn gen_socket_output(kind: &SocketKind, socket_types: &mut IndexSet<String>) -> TokenStream {
        match kind {
            SocketKind::Float => quote!(akari_nodegraph::NodeProxyOutput<f64>),
            SocketKind::Int => quote!(akari_nodegraph::NodeProxyOutput<i64>),
            SocketKind::Bool => quote!(akari_nodegraph::NodeProxyOutput<bool>),
            SocketKind::String => quote!(akari_nodegraph::NodeProxyOutput<String>),
            SocketKind::Enum(_) => {
                panic!("Enum cannot be output");
            }
            SocketKind::Node(t) => {
                socket_types.insert(t.clone());
                let t = format_ident!("{}", t);
                quote!(akari_nodegraph::NodeProxyOutput<#t>)
            }
            SocketKind::List(inner) => {
                let inner = gen_socket_input(inner, socket_types);
                quote!(Vec<#inner>)
            }
        }
    }
    let input_names = desc
        .inputs
        .iter()
        .map(|i| format_ident!("in_{}", i.name))
        .collect::<Vec<_>>();
    let output_names = desc
        .outputs
        .iter()
        .map(|o| format_ident!("out_{}", o.name))
        .collect::<Vec<_>>();
    let inputs = desc
        .inputs
        .iter()
        .map(|i| {
            let name = format_ident!("in_{}", i.name);
            let kind = &i.kind;
            let ty = gen_socket_input(kind, socket_types);
            quote! {
                pub #name: #ty,
            }
        })
        .collect::<Vec<_>>();
    let outputs = desc
        .outputs
        .iter()
        .map(|o| {
            let name = format_ident!("out_{}", o.name);
            let kind = &o.kind;
            let ty = gen_socket_output(kind, socket_types);
            quote! {
                pub #name: #ty,
            }
        })
        .collect::<Vec<_>>();
    let make_inputs = desc
        .inputs
        .iter()
        .map(|i| {
            let key = &i.name;
            let name = format_ident!("in_{}", i.name);
            let kind = &i.kind;
            let ty = format_ident!("{}", kind.ty());
            match kind {
                SocketKind::Float
                | SocketKind::Int
                | SocketKind::Bool
                | SocketKind::String
                | SocketKind::Node(_) => {
                    quote! {
                        let #name = node.input(#key)?.as_proxy_input::<#ty>()?;
                    }
                }
                SocketKind::Enum(e) => {
                    let e = format_ident!("{}", e);
                    quote! {
                        let #name = node.input(#key)?.as_enum::<#e>()?;
                    }
                }

                SocketKind::List(_) => {
                    quote! {
                        let #name = node.input(#key)?.as_proxy_input_list::<#ty>()?;
                    }
                }
            }
        })
        .collect::<Vec<_>>();
    let make_outputs = desc
        .outputs
        .iter()
        .map(|o| {
            let key = &o.name;
            let name = format_ident!("out_{}", o.name);
            let kind = &o.kind;
            let ty = format_ident!("{}", kind.ty());
            quote! {
                let #name = node.output(#key)?.as_proxy_output::<#ty>()?;
            }
        })
        .collect::<Vec<_>>();
    quote! {
        #[derive(Clone, Debug)]
        pub struct #name {
            #(#inputs)*
            #(#outputs)*
        }
        impl akari_nodegraph::NodeProxy for #name {
            fn from_node(graph: &akari_nodegraph::NodeGraph, node: &akari_nodegraph::Node) -> Result<Self, akari_nodegraph::NodeProxyError> {
                if !node.isa(Self::ty()) {
                    return Err(akari_nodegraph::NodeProxyError {
                        msg: format!("Node is not of type {}", Self::ty()),
                    });
                }
                #(#make_inputs)*
                #(#make_outputs)*
                Ok(Self {
                    #(#input_names,)*
                    #(#output_names,)*
                })
            }
            // fn to_node(&self, graph: & akari_nodegraph::NodeGraph) -> akari_nodegraph::Node {
            //     todo!()
            // }
            fn category() -> &'static str {
                stringify!(#category)
            }
            fn ty() -> &'static str {
                stringify!(#name)
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
    let node_types = descs.iter().map(|d| d.name.clone()).collect::<HashSet<_>>();
    let mut socket_types = IndexSet::new();
    for desc in descs {
        tokens.extend(gen_rust_for_desc(desc, &mut socket_types));
    }
    let socket_types = socket_types
        .iter()
        .map(|t| {
            let def = if node_types.contains(t) {
                quote! {}
            } else {
                let t = format_ident!("{}", t);
                quote! {
                    #[derive(Clone, Debug)]
                    pub struct #t;
                }
            };
            let t = format_ident!("{}", t);
            quote! {
                #def
                impl akari_nodegraph::SocketType for #t {
                    fn is_primitive() -> bool {
                        false
                    }
                    fn ty() -> &'static str {
                        stringify!(#t)
                    }
                }
            }
        })
        .collect::<Vec<_>>();
    quote! {
        #tokens
        #(#socket_types)*
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
