use std::sync::{atomic::AtomicBool, Arc};

use luisa::runtime::Swapchain;
use winit::{
    event::{Event as WinitEvent, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

use crate::*;
struct DisplayContex {
    window: Window,
    stream: Stream,
    screen: Tex2d<Float4>,
    swapchain: Swapchain,
    has_update: AtomicBool,
}
pub struct DisplayWindow {
    event_loop: EventLoop<()>,
    ctx: Arc<DisplayContex>,
}

impl DisplayWindow {
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        let event_loop = EventLoop::new().unwrap();
        let stream = device.create_stream(StreamTag::Graphics);
        let window = winit::window::WindowBuilder::new()
            .with_title("AkariRender")
            .with_inner_size(winit::dpi::LogicalSize::new(width, height))
            .with_resizable(false)
            .build(&event_loop)
            .unwrap();
        let swapchain = device.create_swapchain(&window, &stream, width, height, false, false, 3);
        let display_img =
            device.create_tex2d::<Float4>(swapchain.pixel_storage(), width, height, 1);
        Self {
            event_loop,
            ctx: Arc::new(DisplayContex {
                window,
                stream,
                screen: display_img,
                swapchain,
                has_update: AtomicBool::new(false),
            }),
        }
    }
    pub fn show(self) {
        let Self {
            event_loop, ctx, ..
        } = self;
        event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        event_loop
            .run(move |event, elwt| {
                let window = &ctx.window;
                let stream = &ctx.stream;
                let sc = &ctx.swapchain;
                match event {
                    WinitEvent::AboutToWait => {
                        window.request_redraw();
                    }

                    WinitEvent::WindowEvent { event, window_id } if window_id == window.id() => {
                        match event {
                            WindowEvent::CloseRequested => {
                                elwt.exit();
                            }
                            WindowEvent::RedrawRequested => {
                                if ctx
                                    .has_update
                                    .swap(false, std::sync::atomic::Ordering::Relaxed)
                                {
                                    stream.with_scope(|s| {
                                        s.present(sc, &ctx.screen);
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            })
            .unwrap();
    }
    pub fn channel(&self) -> DisplayChannel {
        DisplayChannel {
            ctx: self.ctx.clone(),
        }
    }
}

#[derive(Clone)]
pub struct DisplayChannel {
    ctx: Arc<DisplayContex>,
}

impl DisplayChannel {
    pub fn set_title(&self, title: &str) {
        self.ctx.window.set_title(title);
    }
    pub fn screen_tex(&self) -> &Tex2d<Float4> {
        &self.ctx.screen
    }
    pub fn notify_update(&self) {
        self.ctx.window.request_redraw();
        self.ctx
            .has_update
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}
