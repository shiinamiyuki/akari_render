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

#ifndef AKARIRENDER_MIXED_MATERIAL_H
#define AKARIRENDER_MIXED_MATERIAL_H

#include <akari/render/material.h>
#include <akari/render/texture.h>

namespace akari {
    AKR_EXPORT std::shared_ptr<Material> create_disney_material(
        std::shared_ptr<Texture> base_color, std::shared_ptr<Texture> subsurface, std::shared_ptr<Texture> metallic,
        std::shared_ptr<Texture> specular, std::shared_ptr<Texture> specularTint, std::shared_ptr<Texture> roughness,
        std::shared_ptr<Texture> anisotropic, std::shared_ptr<Texture> sheen, std::shared_ptr<Texture> sheen_tint,
        std::shared_ptr<Texture> clearcoat, std::shared_ptr<Texture> clearcoat_gloss, std::shared_ptr<Texture> ior,
        std::shared_ptr<Texture> spec_trans);
} // namespace akari
#endif // AKARIRENDER_MATTE_H
