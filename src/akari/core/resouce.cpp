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

#include <mutex>
#include <fmt/format.h>
#include <akari/core/resource.h>
#include <akari/core/image.h>

namespace akari {

    class ResourceManagerImpl : public ResourceManager {
        std::mutex mutex;
        struct Record {
            std::shared_ptr<Resource> resouce;
            //            std::filesystem::file_time_type lwt;
        };
        std::unordered_map<std::string, Record> cache;

      public:
        void cache_resource(const fs::path &path, const std::shared_ptr<Resource> &resource) override {
            std::lock_guard<std::mutex> _(mutex);
            cache.emplace(fs::absolute(path).string(), Record{resource});
        }
        std::shared_ptr<Resource> lookup(const fs::path &path) override {
            std::lock_guard<std::mutex> _(mutex);
            auto it = cache.find(fs::absolute(path).string());
            if (it != cache.end()) {
                return it->second.resouce;
            }
            return nullptr;
        }
    };
    namespace _resource_internal {
        static std::shared_ptr<ResourceManagerImpl> mgr;
        static std::once_flag flag;
    } // namespace _resource_internal
    AKR_EXPORT void ResourceManager::finalize() {
        using namespace _resource_internal;
        mgr = nullptr;
    }
    std::shared_ptr<ResourceManager> resource_manager() {
        using namespace _resource_internal;
        std::call_once(flag, [&]() { mgr = std::make_shared<ResourceManagerImpl>(); });
        return mgr;
    }
    void ImageResource::save(OutputArchive &ar) const {
        ar(_image->channels());
        for (int i = 0; i < _image->channels(); i++) {
            ar(_image->channel_name(i));
        }
        ar(_image->resolution());
        ar.save_bytes(reinterpret_cast<const char *>(_image->data()),
                      sizeof(float) * hprod(_image->array3d().dimension()));
    }
    void ImageResource::load(InputArchive &ar) {
        std::vector<std::string> channel_names;
        int channels;
        ar(channels);
        channel_names.resize(channels);
        for (int i = 0; i < channels; i++) {
            ar(channel_names[i]);
        }
        ivec2 resolution;
        ar(resolution);
        _image = std::make_shared<Image>(channel_names, resolution);
        ar.load_bytes(reinterpret_cast<char *>(_image->data()), sizeof(float) * hprod(_image->array3d().dimension()));
    }
    Expected<bool> ImageResource::load(const fs::path &path) {
        auto reader = default_image_reader();
        _image = reader->read(path);
        if (_image)
            return true;
        else {
            return Error(fmt::format("failed to read {}", path.string()));
        }
    }
} // namespace akari