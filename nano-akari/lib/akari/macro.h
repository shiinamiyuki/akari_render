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
#pragma once
#include "map.h"
namespace akari {
    template <class Archive, class T>
    void safe_apply(Archive &ar, const char *name, T &val) {
        try {
            ar(CEREAL_NVP_(name, val));
        } catch (cereal::Exception &e) {
            std::string_view msg(e.what());
            if (msg.find("provided NVP") == std::string_view::npos) {
                // not a name not found error
                throw e;
            }
        }
    }
#define AKR_SER_ONE(value) safe_apply(ar, #value, value);
#define AKR_SER_MULT(...)  ENOKI_MAP(AKR_SER_ONE, __VA_ARGS__)
#define AKR_SER(...)                                                                                                   \
    template <class Archive>                                                                                           \
    void save(Archive &ar) const {                                                                                     \
        AKR_SER_MULT(__VA_ARGS__)                                                                                      \
    }                                                                                                                  \
    template <class Archive>                                                                                           \
    void load(Archive &ar) {                                                                                           \
        AKR_SER_MULT(__VA_ARGS__)                                                                                      \
    }
#define AKR_SER_POLY(Base, ...)                                                                                        \
    template <class Archive>                                                                                           \
    void save(Archive &ar) const {                                                                                     \
        ar(CEREAL_NVP_(#Base, cereal::base_class<Base>(this)));                                                        \
        AKR_SER_MULT(__VA_ARGS__)                                                                                      \
    }                                                                                                                  \
    template <class Archive>                                                                                           \
    void load(Archive &ar) {                                                                                           \
        ar(CEREAL_NVP_(#Base, cereal::base_class<Base>(this)));                                                        \
        AKR_SER_MULT(__VA_ARGS__)                                                                                      \
    }
} // namespace akari