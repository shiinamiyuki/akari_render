// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <iostream>
#include <fstream>
#include <sstream>
#include <akari/scenegraph.h>
#include <akari/render.h>
#include <akari/render_ppg.h>
#include <akari/thread.h>
#include <akari/serial.h>
#include <akari/api.h>
#include <cxxopts.hpp>
using namespace akari;
int main(int argc, char **argv) {

    thread::init(std::thread::hardware_concurrency());
    if (argc < 2) {
        fprintf(stderr, "usage: akari-cli scene\n");
        exit(-1);
    }
    try {
        scene::P<scene::SceneGraph> scene_graph;
        {
            std::ifstream t(argv[1]);
            std::stringstream buffer;
            buffer << t.rdbuf();
            cereal::JSONInputArchive ar(buffer);
            ar(scene_graph);
        }
        render_scenegraph(scene_graph);

    } catch (std::exception &e) {
        std::cerr << "error: " << e.what() << std::endl;
    }
    thread::finalize();
}