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
#include <core/resource.h>
#include <core/image.hpp>

namespace akari{

    class ResourceManagerImpl : public ResourceManager {
        std::mutex mutex;
        struct Record {
            std::shared_ptr<Resource> resouce;
            //            std::filesystem::file_time_type lwt;
            
        };
        std::unordered_map<std::string, Record> cache;
    public:
        void cache_resource(const fs::path & path, const std::shared_ptr<Resource> &resource) override {
            std::lock_guard<std::mutex> _(mutex);
            cache.emplace(fs::absolute(path).string(), Record{resource});
        }
        std::shared_ptr<Resource> lookup(const fs::path & path) override {
            std::lock_guard<std::mutex> _(mutex);
            auto it = cache.find(fs::absolute(path).string());
            if(it != cache.end()){
                return it->second.resouce;
            }
            return nullptr;
        }
    };
    std::shared_ptr<ResourceManager> resource_manager() {
        static std::shared_ptr<ResourceManagerImpl> mgr;
        static std::once_flag flag;
        std::call_once(flag, [&]() { mgr = std::make_shared<ResourceManagerImpl>(); });
        return mgr;
    }

    Expected<bool> ImageResource::load(const fs::path & path){
        auto reader = default_image_reader();
        _image = reader->read(path);
        if(_image)
            return true;
        else{
            return Error(fmt::format("failed to read {}", path.string()));
        }
    }
}