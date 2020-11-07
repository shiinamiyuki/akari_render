// MIT License
//
// Copyright (c) 2020 椎名深雪
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <fstream>
#include <iostream>
#include <akari/core/color.h>
#include <akari/core/application.h>
#include <akari/core/logger.h>
#include <akari/core/parser.h>
#include <akari/core/parallel.h>
#include <akari/render/scenegraph.h>
#include <akari/render/scene.h>
#include <akari/core/comm.h>
using namespace akari;

#ifndef AKR_ENABLE_PYTHON
namespace pybind11 {
    class module {};
} // namespace pybind11
#endif
namespace py = pybind11;
static std::string inputFilename;
static int spp = 0;
static int super_sampling = 0;
static bool denoise = false;
static std::string denoiser;
static bool use_net = false;
static std::list<std::pair<std::string, int>> workers;
static PluginManager<comm::AbstractNetworkWorld> comm_plugins;
static int DEFAULT_PORT = 7856;
int listen_port = DEFAULT_PORT;
std::shared_ptr<comm::World> init_worker() {
    auto pi = comm_plugins.load_plugin("NetworkWorld");
    auto w = pi->make_shared();
    w->listen(listen_port);
    return w;
}
std::shared_ptr<comm::World> init_master() {
    auto pi = comm_plugins.load_plugin("NetworkWorld");
    auto w = pi->make_shared();
    w->connect(workers);
    return w;
}
std::shared_ptr<comm::World> init_local() { return comm::local_world(); }
static auto init_func = &init_local;
static size_t num_threads = std::thread::hardware_concurrency();

void parse(int argc, const char **argv) {
    auto print_help = [] {
        std::cout << R"(
Usage: akari [option] <scene file>

Options:

    -h, --help
        Display this help message

    -v, --verbose
        Enable verbose output

    --spp
        Override spp
    
    --denoise <denoiser>
        Run denoiser after rendering
        <denoiser> is one of:
            default         The default denoiser (see below)
            oidn            Intel OpenImageDenoise (default)
            optix           Optix AI Denoiser
            <other>         load that specific denoiser plugin

    --ss <count>
        Use <count> times super sampling inside render pipeline
        Only useful when --denoise is specified

    -t, --threads <count>
        Render with <count> threads

    -n, --network
        Enable network rendering. Set this process as master

    -w, --worker <hostname[:port]>
        Add a worker for network rendering.
        The worker must be started with --listen <port>
        Default port is 7856

    --listen [-p <port>]
        Set this process as a worker and listen on <port>
        Default port is 7856

    )" << std::endl;
    };
    auto ensure = [=](std::string_view option, int k) {
        if (k >= argc) {
            fprintf(stderr, "%s require more arguments", option.data());
            std::exit(1);
        }
    };
    if (argc < 2) {
        print_help();
        std::exit(1);
    }
    int i = 1;
    for (; i < argc;) {
        std::string_view option(argv[i]);
        if (option == "-h" || option == "--help") {
            print_help();
            exit(0);
        } else if (option == "--threads" || option == "-t") {
            ensure(option, i + 1);
            num_threads = std::stoi(argv[i + 1]);
            i += 2;
        } else if (option == "--spp") {
            ensure(option, i + 1);
            spp = std::stoi(argv[i + 1]);
            i += 2;
        } else if (option == "--denoise") {
            denoise = true;
            ensure(option, i + 1);
            denoiser = argv[i + 1];
            i += 2;
        } else if (option == "--ss") {
            ensure(option, i + 1);
            super_sampling = std::stoi(argv[i + 1]);
            i += 2;
        } else if (option == "-v" || option == "--verbose") {
            GetDefaultLogger()->log_verbose(true);
            i++;
        } else if (option == "--listen") {
            init_func = &init_worker;
            i++;
            option = argv[i + 1];
            if (option == "-p") {
                ensure("-listen -p", i + 1);
                listen_port = std::stoi(argv[i + 1]);
                i += 2;
            }
        } else if (option == "--network") {
            init_func = &init_master;
            i++;
        } else if (option == "--worker" || option == "-w") {
            ensure(option, i + 1);
            std::string worker = argv[i + 1];
            auto pos = worker.find(':');
            int port = DEFAULT_PORT;
            if (pos != std::string::npos) {
                port = std::stoi(worker.substr(pos + 1));
                worker = worker.substr(0, pos);
            }
            workers.emplace_back(worker, port);
            i += 2;
        } else if (inputFilename.empty()) {
            inputFilename = argv[i];
            i++;
        } else {
            std::cerr << "unrecognized option: " << argv[i] << std::endl;
            i++;
        }
    }
}

void parse_and_run() {
    using namespace sdl;
    auto parser = render::SceneGraphParser::create_parser();
    auto module = parser->parse_file(inputFilename, "this");
    auto it = module->exports.find("scene");
    if (it == module->exports.end()) {
        fatal("variable 'scene' not found");
    }
    auto scene = dyn_cast<render::SceneNode>(it->second.object());
    AKR_ASSERT_THROW(scene);
    if (spp > 0) {
        scene->set_spp(spp);
    }
    if (super_sampling > 1) {
        scene->super_sample(super_sampling);
    }
    if (denoise) {
        if (denoiser == "default")
            denoiser = "oidn";
        scene->run_denosier(denoiser);
    }
    info("rendering with {} threads", thread::num_work_threads());
    scene->render();
}
void display_build_info() { std::cout << info_build() << std::endl; }
int main(int argc, const char **argv) {
    try {
        Application app(argc, argv);
        thread::init(num_threads);
        comm::init_comm_world(init_func());
        auto _ = AtScopeExit([&] { comm::finalize_comm_world(); });
        display_build_info();
        parse(argc, argv);
        parse_and_run();
    } catch (std::exception &e) {
        fatal("Exception: {}", e.what());
        exit(1);
    }
}
