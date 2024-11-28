use std::{collections::HashMap, rc::Rc};
pub mod parse;
pub use parse::{parse, ParsingError, ParsingLog, ParserConfig};

pub type StringRef = Rc<String>;
#[derive(Debug, Clone)]
pub struct RcSlice<T: Clone> {
    pub data: Rc<Vec<T>>,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone)]
pub enum Primitive {
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Bytes(Vec<u8>),
    Float2([f32; 2]),
    Float3([f32; 3]),
    Float4([f32; 4]),
    Double2([f64; 2]),
    Double3([f64; 3]),
    Double4([f64; 4]),
    Int2([i32; 2]),
    Int3([i32; 3]),
    Int4([i32; 4]),
    UInt2([u32; 2]),
    UInt3([u32; 3]),
    UInt4([u32; 4]),
}

#[derive(Debug, Clone)]
pub enum PrimitiveArray {
    U32(RcSlice<u32>),
    I32(RcSlice<i32>),
    F32(RcSlice<f32>),
    F64(RcSlice<f64>),
    Bool(RcSlice<bool>),
    String(RcSlice<String>),
    Bytes(RcSlice<Vec<u8>>),
    Float2(RcSlice<[f32; 2]>),
    Float3(RcSlice<[f32; 3]>),
    Float4(RcSlice<[f32; 4]>),
    Double2(RcSlice<[f64; 2]>),
    Double3(RcSlice<[f64; 3]>),
    Double4(RcSlice<[f64; 4]>),
    Int2(RcSlice<[i32; 2]>),
    Int3(RcSlice<[i32; 3]>),
    Int4(RcSlice<[i32; 4]>),
    UInt2(RcSlice<[u32; 2]>),
    UInt3(RcSlice<[u32; 3]>),
    UInt4(RcSlice<[u32; 4]>),
}

impl Primitive {
    pub fn ty(&self) -> PrimitiveType {
        match self {
            Primitive::U32(_) => PrimitiveType::U32,
            Primitive::I32(_) => PrimitiveType::I32,
            Primitive::F32(_) => PrimitiveType::F32,
            Primitive::F64(_) => PrimitiveType::F64,
            Primitive::Bool(_) => PrimitiveType::Bool,
            Primitive::String(_) => PrimitiveType::String,
            Primitive::Bytes(_) => PrimitiveType::Bytes,
            Primitive::Float2(_) => PrimitiveType::Float2,
            Primitive::Float3(_) => PrimitiveType::Float3,
            Primitive::Float4(_) => PrimitiveType::Float4,
            Primitive::Double2(_) => PrimitiveType::Double2,
            Primitive::Double3(_) => PrimitiveType::Double3,
            Primitive::Double4(_) => PrimitiveType::Double4,
            Primitive::Int2(_) => PrimitiveType::Int2,
            Primitive::Int3(_) => PrimitiveType::Int3,
            Primitive::Int4(_) => PrimitiveType::Int4,
            Primitive::UInt2(_) => PrimitiveType::UInt2,
            Primitive::UInt3(_) => PrimitiveType::UInt3,
            Primitive::UInt4(_) => PrimitiveType::UInt4,
        }
    }
}
#[derive(Debug, Clone)]
pub enum Type {
    Primitive(PrimitiveType),
    Array(ArrayType),
    Reference(ReferenceType),
    Struct(StructType),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveType {
    U32,
    I32,
    F32,
    F64,
    Bool,
    String,
    Bytes,
    Float2,
    Float3,
    Float4,
    Double2,
    Double3,
    Double4,
    Int2,
    Int3,
    Int4,
    UInt2,
    UInt3,
    UInt4,
}

#[derive(Debug, Clone)]
pub struct ArrayType {
    pub element: Rc<Type>,
    pub length: usize,
}
#[derive(Debug, Clone)]
pub struct ReferenceType {
    pub ty: Rc<Type>,
}
#[derive(Debug, Clone)]
pub struct OptionalType {
    pub ty: Rc<Type>,
}
#[derive(Debug, Clone)]
pub struct Enum {
    pub ty: Rc<Type>,
    pub variants: Value,
}
#[derive(Debug, Clone)]
pub struct UnionType {
    pub variants: Vec<Rc<Type>>,
}
#[derive(Debug, Clone)]
pub struct StructType {
    pub fields: Vec<(StringRef, Rc<Type>)>,
}
#[derive(Debug, Clone)]
pub struct StructValue {
    pub ty: Rc<StructType>,
    pub fields: Vec<(StringRef, Value)>,
}
#[derive(Debug, Clone)]
pub struct OpaqueType {
    pub name: Rc<String>,
    pub fields: HashMap<StringRef, Rc<Type>>,
}
#[derive(Debug, Clone)]
pub struct FieldAcess {
    pub ty: Rc<Type>,
    pub field: StringRef,
}
#[derive(Debug, Clone)]
pub struct IndexAcess {
    pub ty: Rc<Type>,
    pub index: usize,
}
#[derive(Debug, Clone)]
pub struct Input {
    pub name: StringRef,
    pub ty: Rc<Type>,
}
#[derive(Debug, Clone)]
pub enum Value {
    Null,
    Input(Input),
    Primitive(Primitive),
    PrimitiveArray(PrimitiveArray),
    Struct(StructValue),
    Array(Vec<Value>),
    Ref(Rc<Value>),
    FieldAcess(FieldAcess),
    IndexAcess(IndexAcess),
}

#[derive(Debug, Clone)]
pub struct NodeDef {
    pub name: String,
    pub inputs: HashMap<String, (Rc<Type>, Option<Value>)>,
    /// name, type, default value
    pub outputs: HashMap<String, Rc<Type>>,
}

#[derive(Debug, Clone)]
pub struct NodeInstance {
    pub def: Rc<NodeDef>,
    pub inputs: HashMap<StringRef, Value>,
    pub outputs: HashMap<StringRef, Value>,
}

#[derive(Debug, Clone)]
pub struct GroupInstance {
    pub def: Rc<NodeDef>,
    pub inputs: HashMap<StringRef, Input>,
    pub outputs: HashMap<StringRef, Value>,
}

#[derive(Debug, Clone)]
pub struct StructInit {
    pub ty: Rc<Type>,
    pub fields: Vec<(StringRef, Value)>,
}

#[derive(Debug, Clone)]
pub enum NodeLikeInstance {
    Node(NodeInstance),
    Group(GroupInstance),
    Struct(StructInit),
}

#[derive(Debug, Clone)]
pub struct GroupDef {
    pub def: Rc<NodeDef>,
    pub inputs: HashMap<StringRef, Input>,
    pub internal_nodes: Vec<NodeLikeInstance>,
    /// internal nodes, sorted by dependency order
    pub outputs: HashMap<StringRef, Rc<Type>>,
}

#[derive(Debug)]
pub struct Graph {
    pub internal_nodes: Vec<NodeLikeInstance>,
    pub export: Value,
}
