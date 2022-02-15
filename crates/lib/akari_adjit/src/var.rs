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
        let mut new_bb = Rc::new(BasicBlock::new());
        {
            let new_bb = Rc::get_mut(&mut new_bb).unwrap();
            new_bb.outputs = bb.outputs.clone();
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
        {
            let bb = self.bbs.last_mut().unwrap();
            let bb = Rc::get_mut(bb).unwrap();
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
        let exit = prog.collect_bbs(self.bbs.last().unwrap());
        for o in outputs {
            prog.collect(*o);
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
}
impl BasicBlock {
    pub(crate) fn new() -> Self {
        Self {
            preds: vec![],
            nodes: vec![],
            outputs: vec![],
            succs: vec![],
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
pub struct Var<T> {
    pub(crate) node: usize,
    phantom: PhantomData<T>,
}
impl Var<f32> {
    pub fn arg(i: usize) -> Self {
        let node = Node::Arg {
            ty: "f32".into(),
            idx: i,
        };
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            Self {
                node: r.add_node(node),
                phantom: PhantomData {},
            }
        })
    }
}
#[derive(Clone, Copy)]
pub enum AnyVar {
    U32(Var<u32>),
    F32(Var<f32>),
}
impl AnyVar {
    pub fn node(&self) -> usize {
        match self {
            Self::U32(x) => x.node,
            Self::F32(x) => x.node,
        }
    }
}
pub trait Expand {
    fn expand(&self) -> Vec<AnyVar>;
}
pub trait CondStmt<T> {
    fn cond<F: FnOnce() -> T>(&self, then: F, else_: F);
}

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
macro_rules! impl_expand {
    ($($t:ident,)+ ) => {
        impl<$($t:Expand),+> Expand for($($t,)+){
            #[allow(non_snake_case)]
            fn expand(&self)->Vec<AnyVar>{
                let ($($t,)+) = self;
                let mut v = vec![];
                $(
                    v.extend($t.expand());
                )+
                v
            }
        }
    };
}

macro_rules! impl_cond {
    ($($t:ident,)+) => {
        impl_expand!($($t,)+);
        impl<$($t:Expand),+> CondStmt<($($t,)+)> for Var<bool>{
            fn cond<F:FnOnce()->($($t,)+)>(&self, then:F, else_:F){

                let (b0, b1) = RECORDER.with(|r| {
                    let mut r = r.borrow_mut();
                    let b0 = Rc::new(BasicBlock::new());
                    let b1 = Rc::new(BasicBlock::new());
                    let cond = Node::Cond{
                        cond:self.node,
                        x:Rc::downgrade(&b0),
                        y:Rc::downgrade(&b1),
                    };
                    r.add_node(cond.clone());
                    r.bbs.push(b0.clone());
                    (b0,b1)
                });
                let a = then();
                let ea = a.expand();
                RECORDER.with(|r| {
                    let mut r = r.borrow_mut();
                    r.bbs.push(b1.clone());
                });
                let b = else_();
                let eb = b.expand();
                RECORDER.with(|r| {
                    let mut r = r.borrow_mut();
                    r.bbs.push(Rc::new(BasicBlock::new()));
                    for (i, (x,y)) in ea.iter().zip(eb.iter()).enumerate(){
                        let nx = r.nodes[x.node()].clone();
                        let ny = r.nodes[y.node()].clone();
                        assert_eq!(nx.ty(),ny.ty());
                        let phi = Node::Phi{
                            bb0:b0.clone(),
                            bb1:b1.clone(),
                            out0:i,
                            out1:i,
                            ty:nx.ty().into()
                        };
                        r.add_node(phi);
                    }
                });
                (a, b)
            }
        }
    };
}
impl_cond!(T0,);
impl_cond!(T0, T1,);
impl_cond!(T0, T1, T2,);
impl_cond!(T0, T1, T2, T3,);
impl_cond!(T0, T1, T2, T3, T4,);
impl_cond!(T0, T1, T2, T3, T4, T5,);
impl_cond!(T0, T1, T2, T3, T4, T5, T6,);
impl_cond!(T0, T1, T2, T3, T4, T5, T6, T7,);
impl_cond!(T0, T1, T2, T3, T4, T5, T6, T7, T8,);

macro_rules! impl_from {
    ($t:ty) => {
        impl From<$t> for Var<$t> {
            fn from(x: $t) -> Self {
                let node = Node::Const {
                    ty: stringify!($t).into(),
                    val: x.to_string(),
                };
                RECORDER.with(|r| {
                    let mut r = r.borrow_mut();
                    Self {
                        node: r.add_node(node),
                        phantom: PhantomData {},
                    }
                })
            }
        }
    };
}
macro_rules! impl_binary1 {
    ($t:ty, $op:ident, $func:ident, $tok:tt) => {
        impl std::ops::$op<Var<$t>> for Var<$t> {
            type Output = Self;
            fn $func(self, rhs: Self) -> Self::Output {
                let node = Node::Binary {
                    ty: stringify!($t).into(),
                    op: stringify!($tok),
                    lhs: self.node,
                    rhs: rhs.node,
                };
                RECORDER.with(|r| {
                    let mut r = r.borrow_mut();
                    Self {
                        node: r.add_node(node),
                        phantom: PhantomData {},
                    }
                })
            }
        }
        impl std::ops::$op<$t> for Var<$t> {
            type Output = Self;
            fn $func(self, rhs: $t) -> Self::Output {
                self $tok Self::from(rhs)
            }
        }
        impl std::ops::$op<Var<$t>> for $t {
            type Output = Var<$t>;
            fn $func(self, rhs: Var<$t>) -> Self::Output {
               Var::<$t>::from(self) $tok rhs
            }
        }
    };
}
macro_rules! impl_func1 {
    ($t:ty,$func:ident) => {
        impl Var<$t> {
            pub fn $func(&self) -> Var<$t> {
                let node = Node::Func {
                    ty: stringify!($t).into(),
                    f: stringify!($func),
                    args: smallvec![self.node],
                };
                RECORDER.with(|r| {
                    let mut r = r.borrow_mut();
                    Var {
                        node: r.add_node(node),
                        phantom: PhantomData {},
                    }
                })
            }
        }
    };
}
macro_rules! impl_cmp1 {
    ($t:ty,$func:ident, $tok:tt) => {
        impl Var<$t> {
            pub fn $func(self, rhs: Self) -> Var<bool> {
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
                        phantom: PhantomData {},
                    }
                })
            }
        }
    };
}
macro_rules! impl_cmp {
    ($t:ty) => {
        impl_cmp1!($t, cmplt, <);
        impl_cmp1!($t, cmpgt, >);
        impl_cmp1!($t, cmple, <=);
        impl_cmp1!($t, cmpge, >=);
        impl_cmp1!($t, cmpeq, ==);
        impl_cmp1!($t, cmpne, !=);

    };
}
macro_rules! impl_binary {
    ($t:ty) => {
        impl_binary1!($t, Add, add, +);
        impl_binary1!($t, Sub, sub, -);
        impl_binary1!($t, Mul, mul, *);
        impl_binary1!($t, Div, div, /);
        impl_binary1!($t, Rem, rem, %);
        impl_cmp!($t);
    };
}
macro_rules! impl_var_b {
    ($t:ty) => {
        impl_from!($t);
        impl_binary1!($t, BitAnd, bitand, &);
        impl_binary1!($t, BitOr, bitor, |);
        impl_binary1!($t, BitXor, bitxor, ^);
    };
}
macro_rules! impl_var_f {
    ($t:ty) => {
        impl_from!($t);
        impl_binary!($t);
        impl_func1!($t, sin);
        impl_func1!($t, cos);
        impl_func1!($t, tan);
        impl_func1!($t, log);
        impl_func1!($t, sqrt);
    };
}
macro_rules! impl_var_i {
    ($t:ty) => {
        impl_from!($t);
        impl_binary!($t);
        impl_binary1!($t, Shl, shl, <<);
        impl_binary1!($t, Shr, shr, >>);
        impl_binary1!($t, BitAnd, bitand, &);
        impl_binary1!($t, BitOr, bitor, |);
        impl_binary1!($t, BitXor, bitxor, ^);
    };
}
impl_var_b!(bool);
impl_var_i!(i32);
impl_var_i!(u32);
impl_var_i!(i64);
impl_var_i!(u64);
impl_var_i!(isize);
impl_var_i!(usize);
impl_var_f!(f32);
impl_var_f!(f64);
