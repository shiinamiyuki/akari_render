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
#include <akari/core/nodes/camera.h>
#include <akari/kernel/camera.h>
#include <akari/core/logger.h>
namespace akari {
    AKR_VARIANT class PerspectiveCameraNode : public CameraNode<C> {
      public:
        AKR_IMPORT_TYPES()
        Point3f position;
        Vector3f rotation;
        Point2i resolution = Point2i(512, 512);
        Float fov = radians(80.0f);
        Camera<C> compile(MemoryArena *arena) override {
            Transform3f c2w;
            c2w = Transform3f::rotate_z(rotation.z());
            c2w = Transform3f::rotate_x(rotation.y()) * c2w;
            c2w = Transform3f::rotate_y(rotation.x()) * c2w;
            c2w = Transform3f::translate(position) * c2w;
            return PerspectiveCamera<C>(resolution, c2w, fov);
        }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "fov") {
                fov = radians(value.get<double>().value());
            } else if (field == "rotation") {
                rotation = radians(load_array<Array3f>(value));
            } else if (field == "position") {
                position = load_array<Array3f>(value);
            } else if (field == "resolution") {
                resolution = load_array<Array2i>(value);
            }
        }
    };

    AKR_VARIANT void RegisterCameraNode<C>::register_python_nodes(py::module &m) {
#ifdef AKR_ENABLE_PYTHON
        AKR_IMPORT_TYPES()

        py::class_<CameraNode<C>, SceneGraphNode<C>, std::shared_ptr<CameraNode<C>>>(m, "Camera");
        py::class_<PerspectiveCameraNode<C>, CameraNode<C>, std::shared_ptr<PerspectiveCameraNode<C>>>(
            m, "PerspectiveCamera")
            .def(py::init<>())
            .def_readwrite("position", &PerspectiveCameraNode<C>::position)
            .def_readwrite("rotation", &PerspectiveCameraNode<C>::rotation)
            .def_readwrite("fov", &PerspectiveCameraNode<C>::fov)
            .def_readwrite("resolution", &PerspectiveCameraNode<C>::resolution)
            .def("commit", &PerspectiveCameraNode<C>::commit);
#endif
    }

    AKR_VARIANT void RegisterCameraNode<C>::register_nodes() {
        AKR_IMPORT_TYPES()
        register_node<C, PerspectiveCameraNode<C>>("PerspectiveCamera");
    }
    AKR_RENDER_STRUCT(RegisterCameraNode)

} // namespace akari