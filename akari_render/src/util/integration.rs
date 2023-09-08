use std::sync::Arc;

use crate::*;
#[derive(Clone, Copy, Value)]
#[repr(C)]
struct WorkItem {
    a: f32,
    b: f32,
    c: f32,
    fa: f32,
    fb: f32,
    fc: f32,
    i: f32,
    eps: f32,
    depth: u32,
}
pub fn adaptive_simpson<U: Value>(
    _device: &Device,
    user_data: Expr<U>,
    f: impl Fn(Expr<U>, Expr<f32>) -> Expr<f32>,
    x0: Expr<f32>,
    x1: Expr<f32>,
    eps: f32,
    max_depth: usize,
) -> Expr<f32> {
    let stack = VLArrayVar::<WorkItem>::zero(max_depth * 2 + 1);
    let sp = var!(u32, 0);
    let push = |item: Expr<WorkItem>| {
        stack.write(*sp, item);
        *sp.get_mut() += 1;
    };
    let pop = || {
        *sp.get_mut() -= 1;
        stack.read(*sp)
    };
    let a = x0;
    let b = 0.5 * (x0 + x1);
    let c = x1;
    let fa = f(user_data, a);
    let fb = f(user_data, b);
    let fc = f(user_data, c);
    let i = (c - a) * (1.0f32 / 6.0) * (fa + 4.0 * fb + fc);
    push(WorkItemExpr::new(
        a,
        b,
        c,
        fa,
        fb,
        fc,
        i,
        eps,
        max_depth as u32,
    ));
    let result = var!(f32, 0.0);
    while_!(sp.cmpne(0), {
        let item = pop();
        let a = item.a();
        let b = item.b();
        let c = item.c();
        let fa = item.fa();
        let fb = item.fb();
        let fc = item.fc();
        let i = item.i();
        let d = 0.5 * (a + b);
        let e = 0.5 * (b + c);
        let fd = f(user_data, d);
        let fe = f(user_data, e);
        let eps = item.eps();

        let h = c - a;
        let i0 = (1.0 / 12.0) * h * (fa + 4.0 * fd + fb);
        let i1 = (1.0 / 12.0) * h * (fb + 4.0 * fe + fc);
        let ip = i0 + i1;
        let depth = item.depth();
        if_!(
            depth.cmple(0) | (ip - i).abs().cmplt(15.0 * eps),
            {
                *result.get_mut() += ip + (1.0 / 15.0) * (ip - i);
            },
            else,
            {
                push(WorkItemExpr::new(
                    a,
                    d,
                    b,
                    fa,
                    fd,
                    fb,
                    i0,
                    0.5 * eps,
                    depth - 1,
                ));
                push(WorkItemExpr::new(
                    b,
                    e,
                    c,
                    fb,
                    fe,
                    fc,
                    i1,
                    0.5 * eps,
                    depth - 1,
                ));
            }
        );
    });
    *result
}

pub fn adaptive_simpson_2d<U: Value>(
    device: &Device,
    user_data: Expr<U>,
    f: impl Fn(Expr<U>, Expr<Float2>) -> Expr<f32> + 'static,
    x0: Expr<Float2>,
    x1: Expr<Float2>,
    eps: f32,
    max_depth: usize,
) -> Expr<f32> {
    let integrate_y = |user_data: Expr<U>, x: Expr<Float3>| {
        let y0 = x.y();
        let y1 = x.z();
        let x = x.x();
        adaptive_simpson::<U>(
            device,
            user_data,
            |user_data, y| f(user_data, make_float2(x, y)),
            y0,
            y1,
            eps,
            max_depth,
        )
    };
    adaptive_simpson::<U>(
        device,
        user_data,
        |user_data, x| integrate_y(user_data, make_float3(x, x0.y(), x1.y())),
        x0.x(),
        x1.x(),
        eps,
        max_depth,
    )
}

// #[derive(Clone, Copy, Value)]
// #[repr(C)]
// struct WorkItem2d<U: Value> {
//     user_data: U,
//     y0: f32,
//     y1: f32,
// }

// pub fn adaptive_simpson_2d<U: Value>(
//     device: &Device,
//     user_data: Expr<U>,
//     f: impl Fn(Expr<U>, Expr<Float2>) -> Expr<f32> + 'static,
//     x0: Expr<Float2>,
//     x1: Expr<Float2>,
//     eps: f32,
//     max_depth: usize,
// ) -> Expr<f32>
// where
//     <U as Value>::Expr: CallableParameter,
// {
//     let integrate_y = {
//         let device_ = device.clone();
//         device.create_dyn_callable::<(Expr<U>, Expr<Float3>), Expr<f32>>(Box::new(
//             move |user_data: Expr<U>, x: Expr<Float3>| {
//                 let y0 = x.y();
//                 let y1 = x.z();
//                 let x = x.x();
//                 adaptive_simpson::<U>(
//                     &device_,
//                     user_data,
//                     |user_data, y| f(user_data, make_float2(x, y)),
//                     y0,
//                     y1,
//                     eps,
//                     max_depth,
//                 )
//             },
//         ))
//     };
//     adaptive_simpson::<U>(
//         device,
//         user_data,
//         |user_data, x| integrate_y.call(user_data, make_float3(x, x0.y(), x1.y())),
//         x0.x(),
//         x1.x(),
//         eps,
//         max_depth,
//     )
// }

#[cfg(test)]
mod test {
    use std::env::current_exe;

    use luisa::Context;
    use rand::{thread_rng, Rng};

    use super::*;
    fn test_integration_helper(
        device: &Device,
        inputs: &Buffer<Float2>,
        outputs: &Buffer<Float2>,
        f: impl Fn(Expr<f32>) -> Expr<f32>,
        analytic: impl Fn(Expr<f32>) -> Expr<f32>,
    ) {
        let eps = 1e-6f32;
        let max_depth = 13;
        let kernel = device.create_kernel::<()>(&|| {
            let i = dispatch_id().x();
            let endpoints = inputs.read(i);
            let x0 = endpoints.x();
            let x1 = endpoints.y();
            let int_simpson =
                adaptive_simpson::<bool>(&device, true.into(), |_, x| f(x), x0, x1, eps, max_depth);
            let int_analytic = analytic(x1) - analytic(x0);
            outputs.write(i, make_float2(int_simpson, int_analytic));
        });
        kernel.dispatch([inputs.len() as u32, 1, 1]);
        let inputs = inputs.copy_to_vec();
        let outputs = outputs.copy_to_vec();
        let mut bads = vec![];
        for i in 0..inputs.len() {
            let x0 = inputs[i].x;
            let x1 = inputs[i].y;
            let int_simpson = outputs[i].x;
            let int_analytic = outputs[i].y;
            let error = (int_simpson - int_analytic).abs();
            let rel_error = error / (int_analytic.abs() + 1e-6);
            if error >= 1e-3 || (error >= 1e-6 && rel_error >= 2e-3) {
                bads.push(format!(
                    "x0: {:.8}, x1: {:.8}, int_simpson: {:.8}, int_analytic: {:.8}, error: {:.8}, rel_error: {:.8}",
                    x0, x1, int_simpson, int_analytic, error, rel_error
                ));
                // panic!(
                //     "x0: {:.8}, x1: {:.8}, int_simpson: {:.8}, int_analytic: {:.8}, error: {:.8}, rel_error: {:.8}",
                //     x0, x1, int_simpson, int_analytic, error, rel_error
                // )
            }
        }
        if bads.len() > 5 {
            panic!(
                "too many bads: {} / {}\n{}",
                bads.len(),
                inputs.len(),
                bads.join("\n")
            );
        }
    }
    #[test]
    fn simpson_integration() {
        let ctx = Context::new(current_exe().unwrap());
        let device = ctx.create_cpu_device();
        let mut rng = thread_rng();
        let inputs = device.create_buffer_from_fn(16384, |_| {
            let x0 = rng.gen_range(-2.0f32..2.0f32);
            let x1 = rng.gen_range(-2.0f32..2.0f32);
            Float2::new(x0.min(x1), x0.max(x1))
        });
        let outputs = device.create_buffer(inputs.len());
        test_integration_helper(&device, &inputs, &outputs, |x| x * x, |x| x * x * x / 3.0);
        test_integration_helper(&device, &inputs, &outputs, |x| x.sin(), |x| -x.cos());
        test_integration_helper(
            &device,
            &inputs,
            &outputs,
            |x| x * (-x * x).exp(),
            |x| -(-x * x).exp() * 0.5,
        );
        let k = 4.0f32;
        test_integration_helper(
            &device,
            &inputs,
            &outputs,
            |x| x * (k * x * x).sin(),
            |x| -(k * x * x).cos() / (2.0 * k),
        );
    }
}
