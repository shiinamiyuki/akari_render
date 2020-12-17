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
#include <akari/util.h>
#include <akari/macro.h>
#include <cereal/cereal.hpp>
namespace akari::scene {

    template <typename T>
    using P = std::shared_ptr<T>;
    class Texture {
      public:
        Texture() : value(0.0) {}
        Texture(Float v) : value(v) {}
        Texture(Spectrum s) : value(s) {}
        Texture(std::string s) : value(s) {}
        std::variant<Float, Spectrum, std::string> value;
        AKR_SER(value)
        void clear() { value = Float(0.0); }
        void set_image_texture(const std::string &path) { value = path; }
        void set_color(Spectrum color) { value = color; }
        void set_float(Float v) { value = v; }
    };
    class Material {
      public:
        P<Texture> color;
        P<Texture> specular;
        P<Texture> metallic;
        P<Texture> roughness;
        P<Texture> emission;
        P<Texture> transmission;
        Material();
        void commit() {}
        AKR_SER(color, specular, metallic, roughness, emission, transmission)
    };
    struct Mesh {
        bool loaded = false;

      public:
        std::string name;
        std::vector<vec3> vertices;
        std::vector<ivec3> indices;
        std::vector<vec3> normals;
        std::vector<vec2> texcoords;
        std::string path;
        AKR_SER(name, path)

        void save_to_file(const std::string &file) const;
        void load();
        void unload();
    };

    class Instance {
      public:
        std::string name;
        TRSTransform transform;
        P<Mesh> mesh;
        P<Material> material;
        void commit() { material->commit(); }
        AKR_SER(name, transform, mesh, material)
    };
    class Node {
      public:
        TRSTransform transform;
        std::vector<P<Instance>> instances;
        std::vector<P<Node>> children;
        void commit() {
            for (auto &child : children) {
                child->commit();
            }
        }
        AKR_SER(transform, instances, children)
    };
#define AKR_DECL_RTTI(Class)                                                                                           \
    template <class T>                                                                                                 \
    bool isa() const {                                                                                                 \
        return type() == T::static_type;                                                                               \
    }                                                                                                                  \
    template <class T>                                                                                                 \
    const T *as() const {                                                                                              \
        return isa<T>() ? dynamic_cast<const T *>(this) : nullptr;                                                     \
    }                                                                                                                  \
    template <class T>                                                                                                 \
    T *as() {                                                                                                          \
        return isa<T>() ? dynamic_cast<T *>(this) : nullptr;                                                           \
    }
#define AKR_DECL_TYPEID(Class, TypeId)                                                                                 \
    static const Type static_type = Type::TypeId;                                                                      \
    Type type() const override { return Type::TypeId; }
    class Camera {
      public:
        enum class Type { Perspective };
        AKR_DECL_RTTI(Camera)
        virtual Type type() const = 0;
        TRSTransform transform;
        ivec2 resolution = ivec2(512, 512);
        AKR_SER(transform, resolution)
    };
    class PerspectiveCamera final : public Camera {
      public:
        AKR_DECL_TYPEID(PerspectiveCamera, Perspective)
        Float fov = glm::radians(80.0);
        Float lens_radius = 0.0;
        Float focal_distance = 0.0;
        AKR_SER_POLY(Camera, fov, lens_radius, focal_distance)
    };
    class SceneGraph {
      public:
        P<Camera> camera;
        P<Node> root;
        std::vector<P<Mesh>> meshes;
        std::vector<P<Instance>> instances;
        void commit();
        AKR_SER(camera, meshes, instances, root)
    };
} // namespace akari::scene