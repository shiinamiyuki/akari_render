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

#include <akari/scenegraph.h>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/binary.hpp>

CEREAL_REGISTER_TYPE(akari::scene::Object);

CEREAL_REGISTER_TYPE(akari::scene::Instance);
CEREAL_REGISTER_POLYMORPHIC_RELATION(akari::scene::Object, akari::scene::Instance);

CEREAL_REGISTER_TYPE(akari::scene::Node);
CEREAL_REGISTER_POLYMORPHIC_RELATION(akari::scene::Object, akari::scene::Node);

CEREAL_REGISTER_TYPE(akari::scene::Mesh);
CEREAL_REGISTER_POLYMORPHIC_RELATION(akari::scene::Object, akari::scene::Mesh);

CEREAL_REGISTER_TYPE(akari::scene::Material);
CEREAL_REGISTER_POLYMORPHIC_RELATION(akari::scene::Object, akari::scene::Material);

CEREAL_REGISTER_TYPE(akari::scene::FloatTexture);
CEREAL_REGISTER_POLYMORPHIC_RELATION(akari::scene::Texture, akari::scene::FloatTexture);
CEREAL_REGISTER_TYPE(akari::scene::RGBTexture);
CEREAL_REGISTER_POLYMORPHIC_RELATION(akari::scene::Texture, akari::scene::RGBTexture);
CEREAL_REGISTER_TYPE(akari::scene::ImageTexture);
CEREAL_REGISTER_POLYMORPHIC_RELATION(akari::scene::Texture, akari::scene::ImageTexture);

CEREAL_REGISTER_TYPE(akari::scene::Camera);
CEREAL_REGISTER_POLYMORPHIC_RELATION(akari::scene::Object, akari::scene::Camera);
CEREAL_REGISTER_TYPE(akari::scene::PerspectiveCamera);
CEREAL_REGISTER_POLYMORPHIC_RELATION(akari::scene::Camera, akari::scene::PerspectiveCamera);

CEREAL_REGISTER_TYPE(akari::scene::Integrator);
CEREAL_REGISTER_POLYMORPHIC_RELATION(akari::scene::Object, akari::scene::Integrator);
CEREAL_REGISTER_TYPE(akari::scene::PathTracer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(akari::scene::Integrator, akari::scene::PathTracer);

CEREAL_REGISTER_DYNAMIC_INIT(akari);