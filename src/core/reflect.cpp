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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <akari/core/plugin.h>
#include <akari/core/reflect.hpp>
#include <fmt/format.h>
namespace akari {

    detail::reflection_manager &detail::reflection_manager::instance() {
        static detail::reflection_manager mgr;
        mgr.name_not_found_callback = [](detail::reflection_manager &mgr, const char *name) -> meta_instance & {
            if (get_plugin_manager()->load_plugin(name)) {
                try {

                    return mgr.instances.at(mgr.name_map.at(name));
                } catch (std::out_of_range &e) {
                    (void)e;
                    throw std::runtime_error(
                        fmt::format("type named: `{}` not found after trying loading plugins", name));
                }
            } else {
                throw std::runtime_error(fmt::format("type named: `{}` not found", name));
            }
        };
        return mgr;
    }
} // namespace akari