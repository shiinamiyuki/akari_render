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
#include <akari/core/logger.h>
#include <akari/render/scenegraph.h>
#include <akari/render/camera.h>
#include <akari/render/common.h>
namespace akari::render {

    class PerspectiveCameraNode final : public CameraNode {
      public:
        vec3 position;
        vec3 rotation;
        ivec2 resolution = ivec2(512, 512);
        double fov = glm::radians(80.0f);
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "fov") {
                fov = glm::radians(value.get<double>().value());
            } else if (field == "rotation") {
                rotation = radians(load<vec3>(value));
            } else if (field == "position") {
                position = load<vec3>(value);
            } else if (field == "resolution") {
                resolution = load<ivec2>(value);
            }
        }
        Camera *create_camera(Allocator<> *allocator) override {
            // Transform c2w;
            // c2w = Transform::rotate_z(rotation.z);
            // c2w = Transform::rotate_x(rotation.y) * c2w;
            // c2w = Transform::rotate_y(rotation.x) * c2w;
            // c2w = Transform::translate(position) * c2w;
            TRSTransform TRS{position, rotation, Vec3(1.0)};
            auto c2w = TRS();
            return allocator->new_object<Camera>(allocator->new_object<const PerspectiveCamera>(resolution, c2w, fov));
        }
    };
    AKR_EXPORT std::shared_ptr<CameraNode> create_perspective_camera() {
        return std::make_shared<PerspectiveCameraNode>();
    }
} // namespace akari::render