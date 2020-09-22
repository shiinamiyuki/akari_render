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

#include <string>
#include <regex>
#include <json.hpp>
// This pass removes comments

namespace akari::tool {
    inline std::string jsonc_to_json(const std::string &jsonc) {
        std::string json = std::regex_replace(jsonc, std::regex(R"(\/\/.*\n)"), "\n");
        return json;
    }
    template <typename Iter>
    inline std::string join(const std::string &sep, Iter begin, Iter end) {
        std::string out;
        if (begin != end) {
            out.append(*begin);
        }
        for (auto it = begin + 1; it != end; it++) {
            out.append(sep).append(*it);
        }
        return out;
    }

    inline std::string join(const std::string &sep, const nlohmann::json &j) {
        std::vector<std::string> v;
        for (auto i : j) {
            v.emplace_back(i.get<std::string>());
        }
        return join(sep, v.begin(), v.end());
    }
} // namespace akari::tool