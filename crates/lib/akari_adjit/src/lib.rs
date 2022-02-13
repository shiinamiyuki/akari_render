pub mod compile;
pub use libloading;
use var::Var;
pub mod var;
pub mod codegen;
// use std::ops::Add;
// trait Bool {

// }
// trait Num {
//     fn cmplt(&self, rhs: Self) -> impl Bool;
//     fn cmpgt(&self, rhs: Self) -> impl Bool;
//     fn cmple(&self, rhs: Self) -> impl Bool;
//     fn cmpge(&self, rhs: Self) -> impl Bool;
//     fn cmpeq(&self, rhs: Self) -> impl Bool;
//     fn cmpne(&self, rhs: Self) -> impl Bool;
// }

// trait Int : Num +  {

// }

// pub struct KernelInput<'a> {
//     pub vi64: &'a [i64],
//     pub vf32: &'a [f32],
// }
// pub struct KernelOutput<'a> {
//     pub vf32: &'a mut [f32],
// }
// pub type KernelFn = unsafe extern "C" fn(*const i64, usize, *const f32, usize, *mut f32, usize);
pub type FuncCFnPtr = unsafe extern "C" fn(*const f32, usize, *mut f32, usize);
pub type DFuncCFnPtr =
    unsafe extern "C" fn(*const f32, usize, *const f32, usize, *const usize, *mut f32, usize);
pub struct Func {
    size: (usize, usize),
    _lib: libloading::Library,
    p: FuncCFnPtr,
}
impl Func {
    pub fn call(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.size.0);
        assert_eq!(output.len(), self.size.1);
        unsafe {
            (self.p)(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        }
    }
}
pub struct DFunc {
    size: (usize, usize, usize),
    _lib: libloading::Library,
    p: DFuncCFnPtr,
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
                input.len(),
                output.as_ptr(),
                output.len(),
                vars.as_ptr(),
                grad.as_mut_ptr(),
                grad.len(),
            )
        }
    }
}
pub fn jit<F: FnOnce(Vec<Var<f32>>) -> Vec<Var<f32>>>(
    size:(usize, usize),
    f: F,
) -> Func {
    todo!()
}
pub fn grad(f: &Func, vars: &[usize]) -> DFunc {
    todo!()
}
// pub fn hessian(f: &Func, vars: &[usize]) -> DFunc {
//     todo!()
// }
