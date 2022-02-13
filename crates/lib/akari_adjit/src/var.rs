use std::{cell::RefCell, collections::HashMap, marker::PhantomData, rc::Rc, sync::mpsc::Receiver};

use smallvec::{smallvec, SmallVec};
pub(crate) struct Program {
    nodes: HashMap<usize, Node>,
}

pub(crate) struct Recorder {
    nodes: Vec<Node>,
}
impl Recorder {
    pub fn new() -> Self {
        Self { nodes: vec![] }
    }
}
struct ProgramBuilder<'a> {
    recorder: &'a Recorder,
    nodes: HashMap<usize, Node>,
}
impl<'a> ProgramBuilder<'a> {
    fn collect(&mut self, node: usize) {
        if self.nodes.contains_key(&node) {
            return;
        }
        self.nodes.insert(node, self.recorder.nodes[node].clone());
        let deps = self.recorder.nodes[node].depends();
        for dep in deps {
            self.collect(dep);
        }
    }
}
impl Recorder {
    fn add_node(&mut self, node: Node) -> usize {
        let i = self.nodes.len();
        self.nodes.push(node);
        i
    }
    fn collect(&self, outputs: &[usize]) -> Rc<Program> {
        let mut prog = ProgramBuilder {
            recorder: self,
            nodes: HashMap::new(),
        };
        for o in outputs {
            prog.collect(*o);
        }
        Rc::new(Program { nodes: prog.nodes })
    }
}
thread_local! {
    pub(crate) static RECORDER: RefCell<Recorder> = RefCell::new(Recorder::new());
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
    Select {
        ty: String,
        cond: usize,
        x: usize,
        y: usize,
    },
}
impl Node {
    fn depends(&self) -> SmallVec<[usize; 3]> {
        match self {
            Node::Arg { ty, idx } => smallvec![],
            Node::Const { ty, val } => smallvec![],
            Node::Cast { from, to, val } => smallvec![*val],
            Node::Binary { ty, op, lhs, rhs } => smallvec![*lhs, *rhs],
            Node::Unary { ty, op, val } => smallvec![*val],
            Node::Select { ty, cond, x, y } => smallvec![*cond, *x, *y],
        }
    }
}
#[derive(Clone, Copy)]
pub struct Var<T> {
    pub(crate) node: usize,
    phantom: PhantomData<T>,
}
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
