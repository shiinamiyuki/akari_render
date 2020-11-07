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
#include <akari/core/serde.h>
#include <json.hpp>
using namespace akari;

int main(int argc, const char **argv) {
    Application app(argc, argv);
    thread::init(std::thread::hardware_concurrency());
    auto print_help = [] {
        std::cout << R"(
AkariRender serialize utility

Usage: akari-serde <scene file> <binary file>

Options:
)" << std::endl;
    };
    if (argc != 3) {
        print_help();
        exit(1);
    }
    std::string inputFilename = argv[1];
    std::string outputFilename = argv[2];
    using namespace sdl;
    auto parser = render::SceneGraphParser::create_parser();
    auto module = parser->parse_file(inputFilename, "this");
    auto it = module->exports.find("scene");
    if (it == module->exports.end()) {
        fatal("variable 'scene' not found");
    }
    auto scene = dyn_cast<render::SceneNode>(it->second.object());
    auto stream = std::make_unique<FStream>(fs::path(outputFilename), Stream::Mode::Write);
    auto ar = create_output_archive(std::move(stream));
    (*ar)(scene);
    // nlohmann::json j;
    // auto ar = create_output_archive(j);
    // (*ar)(scene);
    // std::ofstream os((fs::path(outputFilename)));
    // os << j.dump(1);
}