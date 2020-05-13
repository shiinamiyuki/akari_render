 // MIT License
//
// Copyright (c) 2019 椎名深雪
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

#pragma warning(disable : 4819)
#include <cxxopts.hpp>
#include <fstream>
#include <akari/core/application.h>
#include <akari/core/logger.h>
#include <akari/render/scene_graph.h>

using namespace akari;

static std::string inputFilename;

void parse(int argc, char **argv) {
    try {
        cxxopts::Options options("akari", " - AkariRender Command Line Interface");
        options.positional_help("input output").show_positional_help();
        {
            auto opt = options.allow_unrecognised_options().add_options();
            opt("i,input", "Input Scene Description File", cxxopts::value<std::string>());
            opt("help", "Show this help");
        }
        options.parse_positional({"input"});
        auto result = options.parse(argc, argv);
        if (!result.count("input")) {
            fatal("Input file must be provided\n");
            std::cout << options.help() << std::endl;
            exit(0);
        }
        if (result.arguments().empty() || result.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }
        inputFilename = result["input"].as<std::string>();
    } catch (const cxxopts::OptionException &e) {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }
}
int main(int argc, char **argv) {
    Application app;
    parse(argc, argv);
    try {
        CurrentPathGuard _guard;
        fs::current_path(fs::absolute(fs::path(inputFilename)).parent_path());
        inputFilename = fs::path(inputFilename).filename().string();
        std::shared_ptr<SceneGraph> graph;
        {
            Info("Loading {}\n", inputFilename);
            std::ifstream in(inputFilename);
            std::string str((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            json data = str.empty() ? json::object() : json::parse(str);

            graph = std::make_shared<SceneGraph>(serialize::load_from_json<SceneGraph>(data));
        }
        graph->commit();
        Info("Start Rendering ...\n");
        for (size_t i = 0; i < graph->render.size(); i++) {
            auto task = graph->create_render_task(i);
            task->Start();
            task->Wait();
            auto film = task->GetFilmUpdate();
            film->write_image(graph->render[i].output);
        }
    } catch (std::exception &e) {
        fatal("Exception: {}", e.what());
        exit(1);
    }
}
