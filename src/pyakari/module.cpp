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

#include <fstream>
#include <sstream>
#include <csignal>
#include <pyakari/module.h>
#include <pybind11/stl_bind.h>
#include <akari/serial.h>
#include <akari/api.h>
#include <akari/gpu/api.h>
#include <akari/thread.h>

PYBIND11_MAKE_OPAQUE(std::vector<akari::scene::P<akari::scene::Object>>);
PYBIND11_MAKE_OPAQUE(std::vector<akari::scene::P<akari::scene::Mesh>>);
PYBIND11_MAKE_OPAQUE(std::vector<akari::scene::P<akari::scene::Instance>>);
PYBIND11_MAKE_OPAQUE(std::vector<akari::scene::P<akari::scene::Node>>);
namespace akari::python {

    template <typename V, typename T, int N>
    void register_vec(py::module &m, const char *name1, const char *name2) {
        // using V = Vector<T, N>;
        auto c = py::class_<V>(m, name1);
        m.def(name2, []() -> V { return V(); });
        if constexpr (N >= 1) {
            m.def(name2, [](T x) -> V { return V(x); });
            c.def_readwrite("x", &V::x);
        }
        if constexpr (N == 2) {
            m.def(name2, [](T x, T y) -> V { return V(x, y); });
        }
        if constexpr (N == 3) {
            m.def(name2, [](T x, T y, T z) -> V { return V(x, y, z); });
        }
        if constexpr (N == 4) {
            m.def(name2, [](T x, T y, T z, T w) -> V { return V(x, y, z, w); });
        }
        if constexpr (N >= 2) {
            c.def_readwrite("y", &V::y);
        }
        if constexpr (N >= 3) {
            c.def_readwrite("z", &V::z);
        }
        if constexpr (N >= 4) {
            c.def_readwrite("w", &V::w);
        }
    }
    void init(py::module &m) {
        using namespace akari::scene;
        m.doc() = "nano-akari python interface";
        register_vec<vec2, float, 2>(m, "_vec2", "vec2");
        register_vec<vec3, float, 3>(m, "_vec3", "vec3");
        register_vec<vec4, float, 4>(m, "_vec4", "vec4");

        register_vec<ivec2, int, 2>(m, "_ivec2", "ivec2");
        register_vec<ivec3, int, 3>(m, "_ivec3", "ivec3");
        register_vec<ivec4, int, 4>(m, "_ivec4", "ivec4");

        register_vec<dvec2, double, 2>(m, "_dvec2", "dvec2");
        register_vec<dvec3, double, 3>(m, "_dvec3", "dvec3");
        register_vec<dvec4, double, 4>(m, "_dvec4", "dvec4");

        register_vec<bvec2, bool, 2>(m, "_bvec2", "bvec2");
        register_vec<bvec3, bool, 3>(m, "_bvec3", "bvec3");
        register_vec<bvec4, bool, 4>(m, "_bvec4", "bvec4");

        register_vec<Color<float, 2>, float, 2>(m, "Color2f", "color2");
        register_vec<Color<float, 3>, float, 3>(m, "Color3f", "color3");
        register_vec<Color<float, 4>, float, 4>(m, "Color4f", "color4");

        py::class_<TRSTransform>(m, "TRSTransform")
            .def(py::init<>())
            .def_readwrite("translation", &TRSTransform::translation)
            .def_readwrite("rotation", &TRSTransform::rotation)
            .def_readwrite("scale", &TRSTransform::scale);
        py::class_<Object, P<Object>>(m, "Object").def_readwrite("name", &Object::name);
        py::class_<Camera, Object, P<Camera>>(m, "Camera")
            .def_readwrite("transform", &Camera::transform)
            .def_readwrite("resolution", &Camera::resolution);
        py::class_<PerspectiveCamera, Camera, P<PerspectiveCamera>>(m, "PerspectiveCamera")
            .def(py::init<>())
            .def_readwrite("fov", &PerspectiveCamera::fov)
            .def_readwrite("lens_radius", &PerspectiveCamera::lens_radius)
            .def_readwrite("focal_distance", &PerspectiveCamera::focal_distance);
        py::class_<Texture, Object, P<Texture>>(m, "Texture");
        py::class_<FloatTexture, Texture, P<FloatTexture>>(m, "FloatTexture")
            .def(py::init<>())
            .def(py::init<Float>())
            .def_readwrite("value", &FloatTexture::value);
        py::class_<RGBTexture, Texture, P<RGBTexture>>(m, "RGBTexture")
            .def(py::init<>())
            .def(py::init<Color3f>())
            .def_readwrite("value", &RGBTexture::value);
        py::class_<ImageTexture, Texture, P<ImageTexture>>(m, "ImageTexture")
            .def(py::init<>())
            .def_readwrite("path", &ImageTexture::path);
        py::class_<Material, Object, P<Material>>(m, "Material")
            .def(py::init<>())
            .def_readwrite("color", &Material::color)
            .def_readwrite("specular", &Material::specular)
            .def_readwrite("metallic", &Material::metallic)
            .def_readwrite("emission", &Material::emission)
            .def_readwrite("roughness", &Material::roughness);
        py::class_<Volume, Object, P<Volume>>(m, "Volume");
        py::class_<HomogeneousVolume, Volume, P<HomogeneousVolume>>(m, "HomogeneousVolume")
            .def(py::init<>())
            .def_readwrite("color", &HomogeneousVolume::color)
            .def_readwrite("absorption", &HomogeneousVolume::absorption)
            .def_readwrite("density", &HomogeneousVolume::density)
            .def_readwrite("anisotropy", &HomogeneousVolume::anisotropy);
        py::class_<Instance, Object, P<Instance>>(m, "Instance")
            .def(py::init<>())
            .def_readwrite("transform", &Instance::transform)
            .def_readwrite("material", &Instance::material)
            .def_readwrite("volume", &Instance::volume)
            .def_readwrite("mesh", &Instance::mesh);
        py::class_<Mesh, Object, P<Mesh>>(m, "Mesh")
            .def(py::init<>())
            .def_readwrite("name", &Mesh::name)
            .def_readwrite("path", &Mesh::path);
        py::class_<Node, Object, P<Node>>(m, "Node")
            .def(py::init<>())
            .def_readwrite("transform", &Node::transform)
            .def_readwrite("instances", &Node::instances)
            .def_readwrite("children", &Node::children);
        py::class_<Integrator, Object, P<Integrator>>(m, "Integrator");
        py::class_<PathTracer, Integrator, P<PathTracer>>(m, "PathTracer")
            .def(py::init<>())
            .def_readwrite("spp", &PathTracer::spp)
            .def_readwrite("min_depth", &PathTracer::min_depth)
            .def_readwrite("max_depth", &PathTracer::max_depth);
        py::class_<UnifiedPathTracer, Integrator, P<UnifiedPathTracer>>(m, "UnifiedPathTracer")
            .def(py::init<>())
            .def_readwrite("spp", &UnifiedPathTracer::spp)
            .def_readwrite("min_depth", &UnifiedPathTracer::min_depth)
            .def_readwrite("max_depth", &UnifiedPathTracer::max_depth);
        py::class_<GuidedPathTracer, Integrator, P<GuidedPathTracer>>(m, "GuidedPathTracer")
            .def(py::init<>())
            .def_readwrite("spp", &GuidedPathTracer::spp)
            .def_readwrite("min_depth", &GuidedPathTracer::min_depth)
            .def_readwrite("max_depth", &GuidedPathTracer::max_depth)
            .def_readwrite("metropolized", &GuidedPathTracer::metropolized);
        py::class_<SMCMC, Integrator, P<SMCMC>>(m, "SMCMC")
            .def(py::init<>())
            .def_readwrite("spp", &SMCMC::spp)
            .def_readwrite("min_depth", &SMCMC::min_depth)
            .def_readwrite("max_depth", &SMCMC::max_depth);
        py::class_<MCMC, Integrator, P<MCMC>>(m, "MCMC")
            .def(py::init<>())
            .def_readwrite("spp", &MCMC::spp)
            .def_readwrite("min_depth", &MCMC::min_depth)
            .def_readwrite("max_depth", &MCMC::max_depth);
        py::class_<VPL, Integrator, P<VPL>>(m, "VPL")
            .def(py::init<>())
            .def_readwrite("spp", &VPL::spp)
            .def_readwrite("min_depth", &VPL::min_depth)
            .def_readwrite("max_depth", &VPL::max_depth);
        py::class_<BDPT, Integrator, P<BDPT>>(m, "BDPT")
            .def(py::init<>())
            .def_readwrite("spp", &BDPT::spp)
            .def_readwrite("min_depth", &BDPT::min_depth)
            .def_readwrite("max_depth", &BDPT::max_depth);
        py::class_<SceneGraph, P<SceneGraph>>(m, "SceneGraph")
            .def(py::init<>())
            .def_readwrite("meshes", &SceneGraph::meshes)
            .def_readwrite("root", &SceneGraph::root)
            .def_readwrite("camera", &SceneGraph::camera)
            .def_readwrite("instances", &SceneGraph::instances)
            .def_readwrite("integrator", &SceneGraph::integrator)
            .def_readwrite("output_path", &SceneGraph::output_path)
            .def("find", &SceneGraph::find);
        m.def("save_json_str", [](P<SceneGraph> scene) -> std::string {
            std::ostringstream os;
            {
                cereal::JSONOutputArchive ar(os);
                ar(scene);
            }
            return os.str();
        });
        m.def("load_json_str", [](const std::string &s) -> P<SceneGraph> {
            std::istringstream in(s);
            cereal::JSONInputArchive ar(in);
            P<SceneGraph> scene;
            ar(scene);
            return scene;
        });
        m.def("save_xml_str", [](P<SceneGraph> scene) -> std::string {
            std::ostringstream os;
            {
                cereal::XMLOutputArchive ar(os);
                ar(scene);
            }
            return os.str();
        });
        m.def("load_xml_str", [](const std::string &s) -> P<SceneGraph> {
            std::istringstream in(s);
            cereal::XMLInputArchive ar(in);
            P<SceneGraph> scene;
            ar(scene);
            return scene;
        });
        m.def("load", [](const std::string &s) -> P<SceneGraph> {
            auto path = fs::path(s);
            P<SceneGraph> scene;
            if (path.extension() == ".json") {
                std::ifstream in(path);
                cereal::JSONInputArchive ar(in);
                ar(scene);
            } else if (path.extension() == ".xml") {
                std::ifstream in(path);
                cereal::XMLInputArchive ar(in);
                ar(scene);
            } else {
                std::ifstream in(path, std::ios::binary);
                cereal::BinaryInputArchive ar(in);
                ar(scene);
            }
            return scene;
        });
        m.def("save", [](const P<SceneGraph> &scene, const std::string &s) {
            auto path = fs::path(s);
            if (path.extension() == ".json") {
                std::ofstream out(path);
                cereal::JSONOutputArchive ar(out);
                ar(scene);
            } else if (path.extension() == ".xml") {
                std::ofstream out(path);
                cereal::XMLOutputArchive ar(out);
                ar(scene);
            } else {
                std::ofstream out(path, std::ios::binary);
                cereal::BinaryOutputArchive ar(out);
                ar(scene);
            }
        });
        m.def("render", [](P<SceneGraph> scenegraph) {
            auto old_sig_handler = std::signal(SIGINT, SIG_DFL);
            astd::scope_exit _([=] { std::signal(SIGINT, old_sig_handler); });
            render_scenegraph(scenegraph);
        });
#ifdef AKR_BACKEND_CUDA
        m.def("render_gpu", [](P<SceneGraph> scenegraph, const std::string &backend) {
            auto old_sig_handler = std::signal(SIGINT, SIG_DFL);
            astd::scope_exit _([=] { std::signal(SIGINT, old_sig_handler); });
            gpu::render_scenegraph(scenegraph, backend);
        });
#endif
        m.def("thread_pool_init", thread::init);
        m.def("thread_pool_finalize", thread::finalize);
        py::bind_vector<std::vector<P<Object>>>(m, "ObjectArray");
        py::bind_vector<std::vector<P<Mesh>>>(m, "MeshArray");
        py::bind_vector<std::vector<P<Instance>>>(m, "InstanceArray");
        py::bind_vector<std::vector<P<Node>>>(m, "NodeArray");
    }
} // namespace akari::python

PYBIND11_MODULE(pyakari, m) { akari::python::init(m); }