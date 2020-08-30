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
#include <akari/kernel/cameras/camera.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
namespace akari {
    AKR_VARIANT class PerspectiveCameraNode : public CameraNode<Float, Spectrum> {
      public:
        AKR_IMPORT_TYPES(Camera)
        Point3f position;
        Vector3f rotation;
        Point2i resolution = Point2i(512, 512);
        Float fov = radians(80.0f);
        ACamera compile(MemoryArena *arena) override {
            AKR_IMPORT_RENDER_TYPES(PerspectiveCamera)
            Transform3f c2w;
            c2w = Transform3f::rotate_z(rotation.z());
            c2w = Transform3f::rotate_x(rotation.y()) * c2w;
            c2w = Transform3f::rotate_y(rotation.x()) * c2w;
            c2w = Transform3f::translate(position) * c2w;
            return arena->alloc<APerspectiveCamera>(resolution, c2w, fov);
        }
    };
    AKR_VARIANT void RegisterCameraNode<Float, Spectrum>::register_nodes(py::module &m) {
        AKR_IMPORT_RENDER_TYPES(CameraNode, SceneGraphNode, SceneGraphNode, PerspectiveCameraNode);
        py::class_<ACameraNode, ASceneGraphNode, std::shared_ptr<ACameraNode>>(m, "Camera");
        py::class_<APerspectiveCameraNode, ACameraNode, std::shared_ptr<APerspectiveCameraNode>>(m, "PerspectiveCamera")
            .def(py::init<>())
            .def_readwrite("position", &APerspectiveCameraNode::position)
            .def_readwrite("rotation", &APerspectiveCameraNode::rotation)
            .def_readwrite("fov", &APerspectiveCameraNode::fov)
            .def_readwrite("resolution", &APerspectiveCameraNode::resolution)
            .def("commit", &APerspectiveCameraNode::commit);
    }
    AKR_RENDER_STRUCT(RegisterCameraNode)
} // namespace akari