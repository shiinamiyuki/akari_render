use smallvec::SmallVec;

use crate::var::{BasicBlock, Node, Program};
use std::collections::{HashSet, VecDeque};
use std::fmt::Write as FmtWrite;
use std::io::Write as IoWrite;
use std::{collections::HashMap, rc::Rc};

pub(crate) struct CodeGen {
    program: Rc<Program>,
    visited: HashSet<usize>,
    defs: String,
    per_bb: HashMap<*const BasicBlock, String>,
}
impl CodeGen {
    pub(crate) fn new(program: Rc<Program>) -> Self {
        Self {
            program,
            visited: HashSet::new(),
            defs: String::new(),
            per_bb: HashMap::new(),
        }
    }
    fn g(&mut self, node: usize, bb: &BasicBlock) {
        self.visited.insert(node);
        let out = node;
        let node = self.program.nodes.get(&node).unwrap().clone();
        // self.defs.push_str(&format!("{} v{};\n", node.ty(), out));
        for dep in node.depends() {
            assert!(self.visited.contains(&dep));
        }
        let body = self.per_bb.get_mut(&(bb as *const BasicBlock)).unwrap();
        match &node {
            Node::Cond { .. } => {}
            Node::Phi {
                bb0,
                bb1,
                out0,
                out1,
                ..
            } => {
                let n0 = bb0.outputs[*out0];
                let n1 = bb1.outputs[*out1];
                assert!(self.visited.contains(&n0));
                assert!(self.visited.contains(&n1));
                writeln!(&mut self.defs, "\t{} v{};", node.ty(), out).unwrap();
                {
                    let s = self
                        .per_bb
                        .get_mut(&(bb0.as_ref() as *const BasicBlock))
                        .unwrap();
                    writeln!(s, "\tv{} = v{};", out, n0).unwrap();
                }
                {
                    let s = self
                        .per_bb
                        .get_mut(&(bb1.as_ref() as *const BasicBlock))
                        .unwrap();
                    writeln!(s, "\tv{} = v{};", out, n0).unwrap();
                }
            }
            Node::Arg { ty: _, idx } => {
                writeln!(body, "\tconst {} v{} = args[{}];", node.ty(), out, *idx).unwrap();
            }
            Node::Const { ty: _, val } => {
                writeln!(body, "\tconst {} v{} = {};", node.ty(), out, val).unwrap();
            }
            Node::Cast { from: _, to, val } => {
                writeln!(body, "\tconst {} v{} = ({})v{};", node.ty(), out, to, *val).unwrap();
            }
            Node::Binary {
                ty: _,
                op,
                lhs,
                rhs,
            } => {
                writeln!(
                    body,
                    "\tconst {} v{} = v{} {} v{};",
                    node.ty(),
                    out,
                    *lhs,
                    op,
                    *rhs
                )
                .unwrap();
            }
            Node::Unary { ty: _, op, val } => {
                writeln!(body, "\tconst {} v{} = {} v{};", node.ty(), out, op, *val,).unwrap();
            }
            Node::Select { ty: _, cond, x, y } => {
                writeln!(
                    body,
                    "\tconst {} v{} = v{} ? v{} : v{};",
                    node.ty(),
                    out,
                    *cond,
                    *x,
                    *y
                )
                .unwrap();
            }
            Node::Func { ty: _, f, args } => {
                writeln!(
                    body,
                    "\tconst {} v{} = {}({});",
                    node.ty(),
                    out,
                    f,
                    args.iter()
                        .map(|x| format!("v{}", *x))
                        .collect::<SmallVec<[String; 3]>>()
                        .join(",")
                )
                .unwrap();
            }
        }
    }
    fn gen_bbs(&mut self) {
        let mut queue = VecDeque::new();
        queue.push_back(self.program.entry.clone());
        while let Some(bb) = queue.pop_front() {
            self.per_bb.insert(Rc::as_ptr(&bb), String::new());
            for node in &bb.nodes {
                self.g(*node, &bb);
            }
            for succ in bb.succs.clone() {
                let succ = succ.upgrade().unwrap();
                queue.push_back(succ);
            }
        }
    }
    fn assemble_bbs(&mut self, mut bb: Rc<BasicBlock>) -> String {
        let mut s = String::new();
        loop {
            write!(&mut s, "{}", self.per_bb.get(&(Rc::as_ptr(&bb))).unwrap()).unwrap();
            let terminator_ = *bb.nodes.last().unwrap();
            let terminator = self.program.nodes.get(&terminator_).unwrap();
            match terminator {
                Node::Cond { .. } => {
                    write!(&mut s, "if(v{}){{", terminator_).unwrap();
                    assert!(bb.succs.len() == 2);
                    write!(&mut s, "{}", self.assemble_bbs(bb.succs[0].upgrade().unwrap())).unwrap();
                    write!(&mut s, "}}else{{").unwrap();
                    write!(&mut s, "{}", self.assemble_bbs(bb.succs[1].upgrade().unwrap())).unwrap();
                    write!(&mut s, "}}").unwrap();
                    let then = bb.succs[0].upgrade().unwrap();
                    let merge = then.succs[0].upgrade().unwrap();
                    bb = merge;
                }
                _ => {
                    assert!(bb.succs.len() <= 1);
                    break;
                }
            }
        }
        s
    }
    pub(crate) fn gen(mut self) -> String {
        self.gen_bbs();
        for o in self.program.outputs.clone() {
            assert!(self.visited.contains(&o));
        }
        let header = r#"extern AKR_JIT_DLL_EXPORT void jit_main(const float * args, float * outputs) {
"#;
        let mut end = String::new();
        for (i, o) in self.program.outputs.iter().enumerate() {
            writeln!(&mut end, "outputs[{}]=v{};", i, o).unwrap();
        }
        writeln!(&mut end, "{}", "}").unwrap();
        let s = format!(
            "{}{}{}{}",
            header,
            self.defs.clone(),
            self.assemble_bbs(self.program.entry.clone()),
            end
        );
        s
    }
}
