use smallvec::SmallVec;

use crate::var::{Node, Program};
use std::collections::HashSet;
use std::fmt::Write as FmtWrite;
use std::io::Write as IoWrite;
use std::{collections::HashMap, rc::Rc};

pub(crate) struct CodeGen {
    program: Rc<Program>,
    visited: HashSet<usize>,
    // defs: String,
    body: String,
}
impl CodeGen {
    pub(crate) fn new(program: Rc<Program>) -> Self {
        Self {
            program,
            visited: HashSet::new(),
            body: String::new(),
        }
    }
    fn g(&mut self, node: usize) {
        if self.visited.contains(&node) {
            return;
        } else {
            self.visited.insert(node);
            let out = node;
            let node = self.program.nodes.get(&node).unwrap().clone();
            // self.defs.push_str(&format!("{} v{};\n", node.ty(), out));
            for dep in node.depends() {
                self.g(dep);
            }
            match &node {
                Node::Arg { ty: _, idx } => {
                    writeln!(
                        &mut self.body,
                        "\tconst {} v{} = args[{}];",
                        node.ty(),
                        out,
                        *idx
                    )
                    .unwrap();
                }
                Node::Const { ty: _, val } => {
                    writeln!(&mut self.body, "\tconst {} v{} = {};", node.ty(), out, val).unwrap();
                }
                Node::Cast { from: _, to, val } => {
                    writeln!(
                        &mut self.body,
                        "\tconst {} v{} = ({})v{};",
                        node.ty(),
                        out,
                        to,
                        *val
                    )
                    .unwrap();
                }
                Node::Binary {
                    ty: _,
                    op,
                    lhs,
                    rhs,
                } => {
                    writeln!(
                        &mut self.body,
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
                    writeln!(
                        &mut self.body,
                        "\tconst {} v{} = {} v{};",
                        node.ty(),
                        out,
                        op,
                        *val,
                    )
                    .unwrap();
                }
                Node::Select { ty: _, cond, x, y } => {
                    writeln!(
                        &mut self.body,
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
                        &mut self.body,
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
    }
    pub(crate) fn gen(mut self) -> String {
        for o in self.program.outputs.clone() {
            self.g(o);
        }
        let header = r#"extern AKR_JIT_DLL_EXPORT void jit_main(const float * args, float * outputs) {
"#;
        let mut end = String::new();
        for (i, o) in self.program.outputs.iter().enumerate() {
            writeln!(&mut end, "outputs[{}]=v{};", i, o).unwrap();
        }
        writeln!(&mut end, "{}", "}").unwrap();
        format!("{}{}{}", header, self.body, end)
    }
}
