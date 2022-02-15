use std::{
    cell::RefCell,
    collections::HashMap,
    hash::Hash,
    marker::PhantomData,
    rc::{Rc, Weak},
};

use smallvec::{smallvec, SmallVec};
pub(crate) struct Program {
    pub(crate) entry: Rc<BasicBlock>,
    pub(crate) exit: Rc<BasicBlock>,
    pub(crate) nodes: HashMap<usize, Node>,
    pub(crate) node2bb: HashMap<usize, Rc<BasicBlock>>,
    pub(crate) outputs: Vec<usize>,
}

pub(crate) struct Recorder {
    nodes: Vec<Node>,
    bbs: Vec<Rc<BasicBlock>>,
}
impl Recorder {
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            bbs: vec![Rc::new(BasicBlock::new())],
        }
    }
}
struct ProgramBuilder<'a> {
    recorder: &'a Recorder,
    nodes: HashMap<usize, Node>,
    bbs: HashMap<*const BasicBlock, Rc<BasicBlock>>,
    entry: Option<Rc<BasicBlock>>,
    node2bb: HashMap<usize, Rc<BasicBlock>>,
}
impl<'a> ProgramBuilder<'a> {
    fn collect(&mut self, node: usize) {
        if self.nodes.contains_key(&node) {
            return;
        }
        let mut new_node = self.recorder.nodes[node].clone();
        match &mut new_node {
            Node::Phi { bb0, bb1, .. } => {
                *bb0 = self.bbs.get(&Rc::as_ptr(bb0)).unwrap().clone();
                *bb1 = self.bbs.get(&Rc::as_ptr(bb1)).unwrap().clone();
            }
            Node::Cond { x, y, .. } => {
                *x = Rc::downgrade(self.bbs.get(&Rc::as_ptr(&x.upgrade().unwrap())).unwrap());
                *y = Rc::downgrade(self.bbs.get(&Rc::as_ptr(&y.upgrade().unwrap())).unwrap());
            }
            _ => {}
        }
        self.nodes.insert(node, new_node);
        let deps = self.recorder.nodes[node].depends();
        for dep in deps {
            self.collect(dep);
        }
    }
    fn collect_bbs(&mut self, bb: &Rc<BasicBlock>) -> Rc<BasicBlock> {
        if let Some(bb) = self.bbs.get(&Rc::as_ptr(bb)) {
            return bb.clone();
        }
        let mut new_bb = Rc::new(BasicBlock::new());
        {
            let new_bb = Rc::get_mut(&mut new_bb).unwrap();
            new_bb.outputs = bb.outputs.clone();
            if let Some(merge) = &bb.merge {
                new_bb.merge = Some(Rc::downgrade(self.bbs.get(&Weak::as_ptr(merge)).unwrap()));
            }
        }
        self.bbs.insert(Rc::as_ptr(bb), new_bb.clone());
        for pred in &bb.preds {
            let new_pred = self.collect_bbs(pred);
            unsafe {
                let new_bb = &mut *(Rc::as_ptr(&new_bb) as *mut BasicBlock);
                new_bb.preds.push(new_pred.clone());
            }
            unsafe {
                let new_pred = &mut *(Rc::as_ptr(&new_pred) as *mut BasicBlock);
                new_pred.succs.push(Rc::downgrade(&new_bb));
            }
        }
        if bb.preds.is_empty() {
            assert!(self.entry.is_none());
            self.entry = Some(new_bb.clone());
        }
        new_bb
    }
    fn add_node_to_bb(&mut self) {
        for (old_bb, bb) in &mut self.bbs {
            unsafe {
                let old_bb = &**old_bb;
                let bb_ = &mut *(Rc::as_ptr(bb) as *mut BasicBlock);
                for n in &old_bb.nodes {
                    if let Some(_) = self.nodes.get(n) {
                        bb_.nodes.push(*n);
                        self.node2bb.insert(*n, bb.clone());
                    }
                }
            }
        }
    }
}
impl Recorder {
    fn add_node(&mut self, node: Node) -> usize {
        let i = self.nodes.len();
        self.nodes.push(node);
        unsafe {
            let bb = self.bbs.last_mut().unwrap();
            let bb = &mut *(bb.as_ref() as *const BasicBlock as *mut BasicBlock);
            bb.nodes.push(i);
        }
        i
    }
    pub(crate) fn collect(&self, outputs: &[usize]) -> Rc<Program> {
        let mut prog = ProgramBuilder {
            recorder: self,
            nodes: HashMap::new(),
            bbs: HashMap::new(),
            entry: None,
            node2bb: HashMap::new(),
        };
        // for bb in &self.bbs{
        //     println!("{}",bb.nodes.len());
        // }
        let exit = prog.collect_bbs(self.bbs.last().unwrap());
        for o in outputs {
            prog.collect(*o);
        }
        // for (_, bb) in &prog.bbs {
        //     prog.collect(*bb.nodes.last().unwrap());
        // }
        {
            let terms: Vec<_> = prog
                .bbs
                .iter()
                .map(|(bb, _)| unsafe { &**bb })
                .filter(|bb| !bb.nodes.is_empty())
                .map(|bb| *bb.nodes.last().unwrap())
                .collect();
            for term in terms {
                prog.collect(term);
            }
        }
        prog.add_node_to_bb();
        Rc::new(Program {
            nodes: prog.nodes,
            outputs: outputs.to_vec(),
            exit,
            entry: prog.entry.unwrap(),
            node2bb: prog.node2bb,
        })
    }
}
thread_local! {
    pub(crate) static RECORDER: RefCell<Recorder> = RefCell::new(Recorder::new());
}

#[derive(Clone)]
pub(crate) struct BasicBlock {
    pub(crate) preds: Vec<Rc<BasicBlock>>,
    pub(crate) succs: Vec<Weak<BasicBlock>>,
    pub(crate) nodes: Vec<usize>,
    pub(crate) outputs: Vec<usize>,
    pub(crate) merge: Option<Weak<BasicBlock>>,
}
impl BasicBlock {
    pub(crate) fn new() -> Self {
        Self {
            preds: vec![],
            nodes: vec![],
            outputs: vec![],
            succs: vec![],
            merge: None,
        }
    }
}
#[derive(Clone)]
pub(crate) enum Node {
    Arg {
        ty: String,
        idx: usize,
    },
    Const {
        ty: String,
        val: String,
    },
    Cast {
        from: String,
        to: String,
        val: usize,
    },
    Binary {
        ty: String,
        op: &'static str,
        lhs: usize,
        rhs: usize,
    },
    Unary {
        ty: String,
        op: &'static str,
        val: usize,
    },
    Func {
        ty: String,
        f: &'static str,
        args: SmallVec<[usize; 3]>,
    },
    Select {
        ty: String,
        cond: usize,
        x: usize,
        y: usize,
    },
    Cond {
        // ty: Vec<String>,
        cond: usize,
        x: Weak<BasicBlock>,
        y: Weak<BasicBlock>,
    },
    Phi {
        ty: String,
        bb0: Rc<BasicBlock>,
        out0: usize, // n-th output!!!
        bb1: Rc<BasicBlock>,
        out1: usize,
    },
}
impl Node {
    pub(crate) fn ty(&self) -> &str {
        match self {
            Node::Cond { .. } => panic!(),
            Node::Arg { ty, .. } => ty,
            Node::Const { ty, .. } => ty,
            Node::Cast { to, .. } => to,
            Node::Binary { ty, .. } => ty,
            Node::Unary { ty, .. } => ty,
            Node::Select { ty, .. } => ty,
            Node::Func { ty, .. } => ty,
            Node::Phi { ty, .. } => ty,
        }
    }
    pub(crate) fn depends(&self) -> SmallVec<[usize; 3]> {
        match self {
            Node::Cond { cond, .. } => smallvec![*cond],
            Node::Arg { ty: _, idx: _ } => smallvec![],
            Node::Const { ty: _, val: _ } => smallvec![],
            Node::Cast {
                from: _,
                to: _,
                val,
            } => smallvec![*val],
            Node::Binary {
                ty: _,
                op: _,
                lhs,
                rhs,
            } => smallvec![*lhs, *rhs],
            Node::Unary { ty: _, op: _, val } => smallvec![*val],
            Node::Select { ty: _, cond, x, y } => smallvec![*cond, *x, *y],
            Node::Func { args, .. } => args.clone(),
            Node::Phi {
                bb0,
                bb1,
                out0,
                out1,
                ..
            } => {
                smallvec![bb0.outputs[*out0], bb1.outputs[*out1]]
            }
        }
    }
}
#[derive(Clone, Copy)]
pub struct Var {
    pub(crate) node: usize,
    pub(crate) ty: &'static str,
}
impl Var {
    pub fn arg(i: usize, ty: &'static str) -> Self {
        let node = Node::Arg {
            ty: ty.into(),
            idx: i,
        };
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            Self {
                node: r.add_node(node),
                ty,
            }
        })
    }
    pub fn is_float(&self) -> bool {
        match self.ty {
            "f32" | "f64" => true,
            _ => false,
        }
    }
    pub fn is_int(&self) -> bool {
        match self.ty {
            "u8" | "u16" | "u32" | "u64" | "usize" | "i8" | "i16" | "i32" | "i64" | "isize" => true,
            _ => false,
        }
    }
}

// pub trait Expand {
//     fn expand(&self) -> Vec<AnyVar>;
// }
// pub trait CondStmt<T> {
//     fn cond<F: FnOnce() -> T>(&self, then: F, else_: F);
// }

// impl<A:Expand> Expand for (A,){
//     fn expand(&self)->Vec<AnyVar>{
//         self.0.expand()
//     }
// }
// impl<B:Expand> Expand for (B,){
//     fn expand(&self)->Vec<AnyVar>{
//         self.0.expand().co
//     }
// }
// macro_rules! impl_expand {
//     ($($t:ident,)+ ) => {
//         impl<$($t:Expand),+> Expand for($($t,)+){
//             #[allow(non_snake_case)]
//             fn expand(&self)->Vec<AnyVar>{
//                 let ($($t,)+) = self;
//                 let mut v = vec![];
//                 $(
//                     v.extend($t.expand());
//                 )+
//                 v
//             }
//         }
//     };
// }

impl Var {
    pub fn cast(&self, ty: &'static str) -> Self {
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            let cast = Node::Cast {
                to: ty.into(),
                from: self.ty.into(),
                val: self.node,
            };
            Self {
                node: r.add_node(cast),
                ty,
            }
        })
    }
    pub fn cond<F1: Fn() -> Vec<Var>, F2: Fn() -> Vec<Var>>(
        &self,
        then: F1,
        else_: F2,
    ) -> Vec<Var> {
        assert_eq!(self.ty, "bool");
        let (pred, b0, b1) = RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            let mut b0 = BasicBlock::new();
            let mut b1 = BasicBlock::new();
            b0.preds.push(r.bbs.last().unwrap().clone());
            b1.preds.push(r.bbs.last().unwrap().clone());
            let b0 = Rc::new(b0);
            let b1 = Rc::new(b1);
            let cond = Node::Cond {
                cond: self.node,
                x: Rc::downgrade(&b0),
                y: Rc::downgrade(&b1),
            };
            r.add_node(cond.clone());
            let pred = r.bbs.last().unwrap().clone();
            r.bbs.push(b0.clone());
            (pred, b0, b1)
        });
        let a = then();
        for a in &a {
            unsafe {
                let b0 = &mut *(b0.as_ref() as *const BasicBlock as *mut BasicBlock);
                b0.outputs.push(a.node);
            }
        }
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            r.bbs.push(b1.clone());
        });
        let b = else_();
        for b in &b {
            unsafe {
                let b1 = &mut *(b1.as_ref() as *const BasicBlock as *mut BasicBlock);
                b1.outputs.push(b.node);
            }
        }
        assert_eq!(a.len(), b.len());
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            let mut merge = BasicBlock::new();
            merge.preds.push(b0.clone());
            merge.preds.push(b1.clone());
            let merge = Rc::new(merge);
            unsafe {
                let pred = &mut *(pred.as_ref() as *const BasicBlock as *mut BasicBlock);
                pred.merge = Some(Rc::downgrade(&merge));
            }
            r.bbs.push(merge);
            a.iter()
                .zip(b.iter())
                .enumerate()
                .map(|(i, (x, y))| {
                    assert_eq!(
                        x.ty, y.ty,
                        "if stmt returning {} and {} for output {}",
                        x.ty, y.ty, i
                    );
                    let nx = r.nodes[x.node].clone();
                    let ny = r.nodes[y.node].clone();
                    assert_eq!(nx.ty(), ny.ty());
                    let phi = Node::Phi {
                        bb0: b0.clone(),
                        bb1: b1.clone(),
                        out0: i,
                        out1: i,
                        ty: nx.ty().into(),
                    };
                    let phi = r.add_node(phi);
                    Var {
                        ty: x.ty,
                        node: phi,
                    }
                })
                .collect::<Vec<_>>()
        })
    }
}

// impl_cond!(T0,);
// impl_cond!(T0, T1,);
// impl_cond!(T0, T1, T2,);
// impl_cond!(T0, T1, T2, T3,);
// impl_cond!(T0, T1, T2, T3, T4,);
// impl_cond!(T0, T1, T2, T3, T4, T5,);
// impl_cond!(T0, T1, T2, T3, T4, T5, T6,);
// impl_cond!(T0, T1, T2, T3, T4, T5, T6, T7,);
// impl_cond!(T0, T1, T2, T3, T4, T5, T6, T7, T8,);

macro_rules! impl_from {
    ($t:ty) => {
        impl From<$t> for Var {
            fn from(x: $t) -> Self {
                let node = Node::Const {
                    ty: stringify!($t).into(),
                    val: x.to_string(),
                };
                RECORDER.with(|r| {
                    let mut r = r.borrow_mut();
                    Self {
                        node: r.add_node(node),
                        ty: stringify!($t),
                    }
                })
            }
        }
    };
}
macro_rules! impl_binary_ {
    ($op:ident, $func:ident, $tok:tt, $check:expr) => {
        impl std::ops::$op<Var> for Var {
            type Output = Self;
            fn $func(self, rhs: Self) -> Self::Output {
                assert_eq!(self.ty, rhs.ty);
                let node = Node::Binary {
                    ty: self.ty.into(),
                    op: stringify!($tok),
                    lhs: self.node,
                    rhs: rhs.node,
                };
                assert!($check(self));
                RECORDER.with(|r| {
                    let mut r = r.borrow_mut();
                    Self {
                        node: r.add_node(node),
                        ty: self.ty,
                    }
                })
            }
        }
    };
}
macro_rules! impl_binary_common {
    ($op:ident, $func:ident, $tok:tt) => {
        impl_binary_!($op, $func, $tok, |_| true);
    };
}
macro_rules! impl_binary_scalar {
    ($t:ty,$op:ident, $func:ident, $tok:tt) => {
        impl std::ops::$op<$t> for Var {
            type Output = Self;
            fn $func(self, rhs: $t) -> Self::Output {
                self $tok Self::from(rhs)
            }
        }
        impl std::ops::$op<Var> for $t {
            type Output = Var;
            fn $func(self, rhs: Var) -> Self::Output {
               Var::from(self) $tok rhs
            }
        }
    };
}
macro_rules! impl_func1_f {
    ($func:ident) => {
        impl Var {
            pub fn $func(&self) -> Var {
                assert!(self.is_float());
                let node = Node::Func {
                    ty: self.ty.into(),
                    f: stringify!($func),
                    args: smallvec![self.node],
                };
                RECORDER.with(|r| {
                    let mut r = r.borrow_mut();
                    Var {
                        node: r.add_node(node),
                        ty: self.ty,
                    }
                })
            }
        }
    };
}
macro_rules! impl_cmp1 {
    ($func:ident, $tok:tt) => {
        impl Var {
            pub fn $func(self, rhs: Self) -> Var {
                assert_eq!(self.ty, rhs.ty);
                let node = Node::Binary {
                    ty: "bool".into(),
                    op: stringify!($tok),
                    lhs: self.node,
                    rhs: rhs.node,
                };
                RECORDER.with(|r| {
                    let mut r = r.borrow_mut();
                    Var {
                        node: r.add_node(node),
                        ty: "bool",
                    }
                })
            }
        }
    };
}

impl_cmp1!(cmplt, <);
impl_cmp1!(cmpgt, >);
impl_cmp1!(cmple, <=);
impl_cmp1!(cmpge, >=);
impl_cmp1!(cmpeq, ==);
impl_cmp1!(cmpne, !=);

macro_rules! impl_binary {
    ($t:ty) => {
        impl_binary_scalar!($t, Add, add, +);
        impl_binary_scalar!($t, Sub, sub, -);
        impl_binary_scalar!($t, Mul, mul, *);
        impl_binary_scalar!($t, Div, div, /);
        impl_binary_scalar!($t, Rem, rem, %);

    };
}
macro_rules! impl_var_b {
    ($t:ty) => {
        impl_from!($t);
        impl_binary_scalar!($t, BitAnd, bitand, &);
        impl_binary_scalar!($t, BitOr, bitor, |);
        impl_binary_scalar!($t, BitXor, bitxor, ^);
    };
}
macro_rules! impl_var_f {
    ($t:ty) => {
        impl_from!($t);
        impl_binary!($t);
    };
}
macro_rules! impl_var_i {
    ($t:ty) => {
        impl_from!($t);
        impl_binary!($t);
        impl_binary_scalar!($t, Shl, shl, <<);
        impl_binary_scalar!($t, Shr, shr, >>);
        impl_binary_scalar!($t, BitAnd, bitand, &);
        impl_binary_scalar!($t, BitOr, bitor, |);
        impl_binary_scalar!($t, BitXor, bitxor, ^);
    };
}
impl_binary_common!(Add, add, +);
impl_binary_common!(Sub, sub, -);
impl_binary_common!(Mul, mul, *);
impl_binary_common!(Div, div, /);
impl_binary_common!(Rem, rem, %);

impl_binary_!(Shl, shl, <<,|x:Var|x.is_int());
impl_binary_!(Shr, shr, >>,|x:Var|x.is_int());
impl_binary_!(BitAnd, bitand, &,|x:Var|x.is_int());
impl_binary_!(BitOr, bitor, |,|x:Var|x.is_int());
impl_binary_!(BitXor, bitxor, ^,|x:Var|x.is_int());

impl_func1_f!(sin);
impl_func1_f!(cos);
impl_func1_f!(tan);
impl_func1_f!(log);
impl_func1_f!(sqrt);
impl_var_b!(bool);
impl_var_i!(i32);
impl_var_i!(u32);
impl_var_i!(i64);
impl_var_i!(u64);
impl_var_i!(isize);
impl_var_i!(usize);
impl_var_f!(f32);
impl_var_f!(f64);
