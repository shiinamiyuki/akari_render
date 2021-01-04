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

    class Object : public std::enable_shared_from_this<Object> {
      public:
        std::string name;
        AKR_SER(name)
        virtual ~Object() = default;
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
    }                                                                                                                  \
    virtual Type type() const = 0;
#define AKR_DECL_TYPEID(Class, TypeId)                                                                                 \
    static const Type static_type = Type::TypeId;                                                                      \
    Type type() const override { return Type::TypeId; }

    // class Texture {
    //   public:
    //     Texture() : value(0.0) {}
    //     Texture(Float v) : value(v) {}
    //     Texture(Spectrum s) : value(s) {}
    //     Texture(std::string s) : value(s) {}
    //     std::variant<Float, Spectrum, std::string> value;
    //     AKR_SER(value)
    //     void clear() { value = Float(0.0); }
    //     void set_image_texture(const std::string &path) { value = path; }
    //     void set_color(Spectrum color) { value = color; }
    //     void set_float(Float v) { value = v; }
    // };

    class Texture : public Object {
      public:
        enum class Type { Float, RGB, Image };
        AKR_DECL_RTTI(Texture)
        AKR_SER_POLY(Object)
    };
    class FloatTexture : public Texture {
      public:
        Float value = 0.0;
        FloatTexture() = default;
        FloatTexture(Float v) : value(v) {}
        AKR_SER_POLY(Texture, value)
        AKR_DECL_TYPEID(FloatTexture, Float)
    };

    class RGBTexture : public Texture {
      public:
        Color3f value;
        RGBTexture() = default;
        RGBTexture(Color3f v) : value(v) {}
        AKR_SER_POLY(Texture, value)
        AKR_DECL_TYPEID(RGBTexture, RGB)
    };
    class ImageTexture : public Texture {
      public:
        std::string path;
        ImageTexture() = default;
        ImageTexture(const std::string &path) : path(path) {}
        AKR_SER_POLY(Texture, path)
        AKR_DECL_TYPEID(ImageTexture, Image)
    };
    class Material : public Object {
      public:
        P<Texture> color;
        P<Texture> specular;
        P<Texture> metallic;
        P<Texture> roughness;
        P<Texture> emission;
        P<Texture> transmission;
        Material();
        void commit() {}
        AKR_SER_POLY(Object, color, specular, metallic, roughness, emission, transmission)
    };
    class Mesh : public Object {
        bool loaded = false;

      public:
        std::vector<vec3> vertices;
        std::vector<ivec3> indices;
        std::vector<vec3> normals;
        std::vector<vec2> texcoords;
        std::string path;
        AKR_SER_POLY(Object, path)

        void save_to_file(const std::string &file) const;
        void load();
        void unload();
    };

    class Instance : public Object {
      public:
        TRSTransform transform;
        P<Mesh> mesh;
        P<Material> material;
        void commit() { material->commit(); }
        AKR_SER_POLY(Object, transform, mesh, material)
    };
    class Node : public Object {
      public:
        TRSTransform transform;
        std::vector<P<Instance>> instances;
        std::vector<P<Node>> children;
        void commit() {
            for (auto &child : children) {
                child->commit();
            }
        }
        AKR_SER_POLY(Object, transform, instances, children)
    };

    class Camera : public Object {
      public:
        enum class Type { Perspective };
        AKR_DECL_RTTI(Camera)

        TRSTransform transform;
        ivec2 resolution = ivec2(512, 512);
        AKR_SER_POLY(Object, transform, resolution)
    };
    class PerspectiveCamera final : public Camera {
      public:
        AKR_DECL_TYPEID(PerspectiveCamera, Perspective)
        Float fov = glm::radians(80.0);
        Float lens_radius = 0.0;
        Float focal_distance = 0.0;
        AKR_SER_POLY(Camera, fov, lens_radius, focal_distance)
    };

    class Integrator : public Object {
      public:
        enum class Type { Path, VPL, SMCMC };
        AKR_DECL_RTTI(Integrator)
        AKR_SER_POLY(Object)
    };

    class PathTracer : public Integrator {
      public:
        uint32_t spp = 16;
        int32_t min_depth = 4;
        int32_t max_depth = 7;
        AKR_DECL_TYPEID(PathTracer, Path)
        AKR_SER_POLY(Integrator, spp, min_depth, max_depth)
    };
    class SMCMC : public Integrator {
      public:
        uint32_t spp = 16;
        int32_t min_depth = 4;
        int32_t max_depth = 7;
        AKR_DECL_TYPEID(SMCMC, SMCMC)
        AKR_SER_POLY(Integrator, spp, min_depth, max_depth)
    };

    class VPL : public Integrator {
      public:
        uint32_t spp = 16;
        int32_t min_depth = 4;
        int32_t max_depth = 7;
        AKR_DECL_TYPEID(VPL, VPL)
        AKR_SER_POLY(Integrator, spp, min_depth, max_depth)
    };
    class SceneGraph {
      public:
        P<Camera> camera;
        P<Integrator> integrator;
        P<Node> root;
        std::vector<P<Mesh>> meshes;
        std::vector<P<Instance>> instances;
        void commit();
        AKR_SER(camera, integrator, meshes, instances, root)

        std::vector<P<Object>> find(const std::string &name);
    };
} // namespace akari::scene