pub mod compile;
use codegen::CodeGen;
pub use libloading;
use var::{CondStmt, Var, RECORDER};
pub mod codegen;
pub mod var;
pub mod vec;

pub type FuncCFnPtr = unsafe extern "C" fn(*const f32, *mut f32);
pub type DFuncCFnPtr = unsafe extern "C" fn(*const f32, *const f32, *const usize, *mut f32);
// pub type D2FuncCFnPtr = unsafe extern "C" fn(*const f32, *const f32, *const usize, *mut f32);
pub struct Func {
    size: (usize, usize),
    _lib: libloading::Library,
    p: libloading::Symbol<'static, FuncCFnPtr>,
}
impl Func {
    pub fn call(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.size.0);
        assert_eq!(output.len(), self.size.1);
        unsafe { (self.p)(input.as_ptr(), output.as_mut_ptr()) }
    }
}
pub struct DFunc {
    size: (usize, usize, usize),
    _lib: libloading::Library,
    p: libloading::Symbol<'static, DFuncCFnPtr>,
}
impl DFunc {
    pub fn call(&self, input: &[f32], output: &[f32], vars: &[usize], grad: &mut [f32]) {
        assert_eq!(input.len(), self.size.0);
        assert_eq!(output.len(), self.size.1);
        assert_eq!(vars.len(), self.size.2);
        assert_eq!(vars.len() * output.len(), grad.len());
        unsafe {
            (self.p)(
                input.as_ptr(),
                output.as_ptr(),
                vars.as_ptr(),
                grad.as_mut_ptr(),
            )
        }
    }
}
pub fn jit<F: FnOnce(Vec<Var<f32>>) -> Vec<Var<f32>>>(
    func_name: &str,
    ninputs: usize,
    f: F,
) -> Func {
    let inputs = (0..ninputs).map(|i| Var::arg(i)).collect();
    let outputs = f(inputs);
    RECORDER.with(|r| {
        let r = r.borrow();
        let outputs: Vec<_> = outputs.iter().map(|v| v.node).collect();
        let program = r.collect(&outputs);
        let cg = CodeGen::new(program);
        let source = cg.gen();
        let path = compile::compile(source, func_name).unwrap();
        unsafe {
            let lib = libloading::Library::new(path).unwrap();
            let p: libloading::Symbol<FuncCFnPtr> = lib.get(b"jit_main\0").unwrap();
            let p: libloading::Symbol<'static, FuncCFnPtr> = std::mem::transmute(p);
            Func {
                size: (ninputs, outputs.len()),
                _lib: lib,
                p,
            }
        }
    })
}
pub fn grad(f: &Func, vars: &[usize]) -> DFunc {
    todo!()
}
// pub fn hessian(f: &Func, df: &DFunc, vars: &[usize]) -> DFunc {
//     todo!()
// }
pub fn if_<F: FnOnce() -> T, T>(cond: Var<bool>, then: F, else_: F)
where
    Var<bool>: CondStmt<T>,
{
    cond.cond(then, else_);
}
mod test {
    #[test]
    fn test_jit() {
        use super::*;
        let f = jit("add2", 2, |args| vec![args[0] + args[1]]);
        let mut out = [0.0f32];
        f.call(&[2.0, 3.0], &mut out);
        assert!((out[0] - 5.0).abs() < 1e-4);
    }
}
