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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef AKARIRENDER_RESOURCE_H
#define AKARIRENDER_RESOURCE_H

#include <akari/core/akari.h>
#include <akari/core/platform.h>
#include <akari/core/error.hpp>
#include <akari/core/image.h>
#include <akari/core/serde.h>
namespace akari {
    class AKR_EXPORT Resource {
      public:
        virtual Expected<bool> load(const fs::path &) = 0;
        virtual ~Resource() = default;
    };
    class AKR_EXPORT ResourceManager {
        virtual void cache_resource(const fs::path &path, const std::shared_ptr<Resource> &) = 0;
        virtual std::shared_ptr<Resource> lookup(const fs::path &path) = 0;

      public:
        Expected<std::shared_ptr<Resource>> load_resource(const fs::path &);
        template <typename T>
        Expected<std::shared_ptr<T>> load_path(const fs::path &path, bool force_reload = false) {
            if (!force_reload) {
                auto cache = lookup(path);
                if (cache) {
                    auto res = dyn_cast<T>(cache);
                    if (!res) {
                        return Error("dyn_cast failed; cached resource of different type");
                    }
                    return res;
                }
            }
            auto res = std::make_shared<T>();
            auto exp = res->load(path);
            if (exp) {
                cache_resource(path, res);
                return res;
            }
            return exp.extract_error();
        }
        virtual ~ResourceManager() = default;
        static void finalize();
    };
    AKR_EXPORT std::shared_ptr<ResourceManager> resource_manager();

    class AKR_EXPORT ImageResource : public Resource, public Serializable {
        std::shared_ptr<Image> _image;

      public:
        AKR_SER_CLASS("ImageResource")
        void save(OutputArchive &ar) const override;
        void load(InputArchive &ar) override;
        Expected<bool> load(const fs::path &) override;
        const std::shared_ptr<Image> &image() const { return _image; }
    };
} // namespace akari

#endif