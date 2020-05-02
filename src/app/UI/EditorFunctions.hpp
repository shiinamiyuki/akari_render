// MIT License
//
// Copyright (c) 2019 椎名深雪
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

#ifndef AKARIRENDER_EDITORFUNCTIONS_HPP
#define AKARIRENDER_EDITORFUNCTIONS_HPP
#include <akari/Core/Math.h>
#include <akari/Core/Spectrum.h>
#include <imgui.h>
namespace akari::Gui {
    template <typename T> inline bool Edit(const char *label, T &value) = delete;

    template <> inline bool Edit(const char *label, int &value) {
        return ImGui::InputInt(label, &value, 1, 100, ImGuiInputTextFlags_EnterReturnsTrue);
    }

    template <> inline bool Edit(const char *label, float &value) {
        return ImGui::InputFloat(label, &value, 0.1, 1, 3, ImGuiInputTextFlags_EnterReturnsTrue);
    }
    template <> inline bool Edit(const char *label, ivec2 &value) {
        return ImGui::InputInt2(label, reinterpret_cast<int *>(&value), ImGuiInputTextFlags_EnterReturnsTrue);
    }
    template <> inline bool Edit(const char *label, ivec3 &value) {
        return ImGui::InputInt3(label, reinterpret_cast<int *>(&value), ImGuiInputTextFlags_EnterReturnsTrue);
    }
    template <> inline bool Edit(const char *label, vec2 &value) {
        return ImGui::InputFloat2(label, reinterpret_cast<float *>(&value), 3, ImGuiInputTextFlags_EnterReturnsTrue);
    }

    template <> inline bool Edit(const char *label, vec3 &value) {
        return ImGui::InputFloat3(label, reinterpret_cast<float *>(&value), 3, ImGuiInputTextFlags_EnterReturnsTrue);
    }
    template <> inline bool Edit(const char *label, bool &value) { return ImGui::Checkbox(label, &value); }

    template <> inline bool Edit(const char *label, Spectrum &value) {
        //        ImGui::PushID(&value);
        int flags = 0;
        bool ret = ImGui::ColorPicker3(label, reinterpret_cast<float *>(&value), flags);
        //        ImGui::PopID();
        return ret;
    }

    template <typename T> inline bool Edit(const char *label, Angle<T> &value) {
        T degrees = RadiansToDegrees(value.value);
        bool ret = Edit<T>(label, degrees);
        if (ret) {
            value.value = DegreesToRadians(degrees);
        }
        return ret;
    }
    template <> inline bool Edit(const char *label, AffineTransform &value) {
        bool ret = false;
        if (ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_DefaultOpen)) {
            ret = ret | Edit("translation", value.translation);
            ret = ret | Edit("rotation", value.rotation);
            ret = ret | Edit("scale", value.scale);
            ImGui::TreePop();
        }
        return ret;
    }

} // namespace akari::Gui

#endif // AKARIRENDER_EDITORFUNCTIONS_HPP
