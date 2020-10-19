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

#include <set>
#include <fstream>
#include "string-utils.h"
#include <fmt/format.h>
int main(int argc, char **argv) {
    if (argc != 3) {
        fmt::print(stderr, "Usage: akari-soac conf out\n");
        std::exit(1);
    }
    std::ifstream in(argv[1]);
    std::string str((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    str = akari::tool::jsonc_to_json(str);
    std::ofstream out(argv[2]);
    using namespace nlohmann;
    try {
        json cfg = json::parse(str);
        json headers = cfg["headers"];
        auto flats = std::set<std::string>();
        for (auto flat : cfg["flat"]) {
            flats.insert(flat.get<std::string>());
        }
        json soas = cfg["soa"];
        auto gen_struct = [&](std::string name, json desc) {
            json fields = desc["fields"];
            int indent = 4;
            struct Block {
                int &indent;
                Block(int &indent) : indent(indent) { indent += 4; }
                ~Block() { indent -= 4; }
            };
            auto wl = [&](const std::string &s) { out << std::string(indent, ' ') << s << "\n"; };
            json templates = json::array();
            if (desc.contains("template")) {
                templates = desc["template"];
                std::vector<std::string> v;
                for (auto t : desc["template"]) {
                    v.emplace_back("typename " + t.get<std::string>());
                }
                out << "template<" << akari::tool::join(",", v.begin(), v.end()) << ">\n";
            }
            wl("struct SOA<" + name + ">{");
            {
                Block _(indent);
                if (std::find(templates.begin(), templates.end(), "Float") != templates.end()) {
                    wl(";");
                } else if (std::find(templates.begin(), templates.end(), "C") != templates.end()) {
                    wl("\n");
                }
                wl("size_t _size = 0;");
                for (auto field : fields.items()) {
                    auto f = field.key();
                    auto t = field.value().get<std::string>();
                    if (!(templates.contains(t) || soas.contains(t) || flats.find(t) != flats.end())) {
                        fmt::print(stderr, "{} is illegal\n", f);
                        std::exit(1);
                    }
                    wl(fmt::format("SOA<{}> {};", t, f));
                }
                wl("using Self = SOA<" + name + ">;");
                wl("using value_type = " + name + ";");
                wl("Self() = default;");
                wl("template<class Allocator>");
                wl("Self(size_t s, Allocator & alloc): _size(s)");
                for (auto field : fields.items()) {
                    auto f = field.key();
                    wl(", " + f + "(s, alloc)");
                }
                wl("{}");
                wl("struct IndexHelper{");
                {
                    Block __(indent);
                    wl("Self & self;");
                    wl("int idx;");
                    wl(" operator value_type(){");
                    {
                        Block ___(indent);
                        wl("value_type ret;");
                        for (auto [field, _] : fields.items()) {
                            wl("ret." + field + " = self." + field + "[idx];");
                        }
                        wl("return ret;");
                    }
                    wl("}");
                    wl(" const value_type & operator=(const value_type & rhs){");
                    {
                        Block ___(indent);
                        wl("value_type ret;");
                        for (auto [field, _] : fields.items()) {
                            wl("self." + field + "[idx] = rhs." + field + ";");
                        }
                        wl("return rhs;");
                    }
                    wl("}");
                }
                wl("};");
                wl("struct ConstIndexHelper{");
                {
                    Block __(indent);
                    wl("Self & self;");
                    wl("int idx;");
                    wl(" operator value_type(){");
                    {
                        Block ___(indent);
                        wl("value_type ret;");
                        for (auto [field, _] : fields.items()) {
                            wl("ret." + field + " = self." + field + "[idx];");
                        }
                        wl("return ret;");
                    }
                    wl("}");
                }
                wl("};");
                wl(" IndexHelper operator[](int idx){return IndexHelper{*this, idx};}");
                wl(" ConstIndexHelper operator[](int idx)const{return ConstIndexHelper{*this, idx};}");
                wl(" size_t size()const{return _size;}");
            }
            wl("};");
        };
        for (auto h : headers) {
            out << ("#pragma once\n#include <" + h.get<std::string>() + ">\n");
        }
        out << "namespace akari{\n";
        for (auto [name, desc] : soas.items()) {
            gen_struct(name, desc);
            out << "\n";
        }
        out << "}\n";
    } catch (std::exception &e) {
        fmt::print(stderr, "{}\n", e.what());
        std::exit(1);
    }
}