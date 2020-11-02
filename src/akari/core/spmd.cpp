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

#include <akari/core/spmd.h>
#include <deque>
#include <mutex>
namespace akari::spmd {
    class AKR_EXPORT LocalWorld : public World {
        std::shared_ptr<Node> local_;
        // std::mutex m;
        // std::unordered_map<std::shared_ptr<Node>, ChannelRecord> records;
        struct ChannelRecord {
            std::shared_ptr<Node> a, b;
            std::weak_ptr<ByteChannel> channel;
        };
        std::unordered_map<uint32_t, ChannelRecord> channels;
        uint32_t tag_counter = 0;

      public:
        std::shared_ptr<Node> local() const override { return local_; }
        void foreach_nodes(const NodeVisitor &vis) override { vis(local_); }
        std::shared_ptr<ByteChannel> channel(const std::shared_ptr<Node> &a, const std::shared_ptr<Node> &b) override {
            auto channel = std::make_shared<ByteChannel>(tag_counter++, shared_from_this());
            channels.emplace(channel->tag(), ChannelRecord{a, b, channel});
            return channel;
        }
        void send(const std::shared_ptr<ByteChannel> &channel, std::string_view msg) override {
            AKR_ASSERT(channel->world().get() == this);
            channel->__push(Message(msg.data(), msg.data() + msg.size()));
        }
        Message receive(const std::shared_ptr<ByteChannel> &channel) override {
            // std::lock_guard<std::mutex> lock(m);
            return channel->__pop();
        }
        void initialize() override {}
        size_t size() const override { return 1; }
        void finalize() override {
            for (auto &&[_, rec] : channels) {
                if (auto ch = rec.channel.lock()) {
                    ch->__close();
                }
            }
            channels.clear();
        }
    };
    class AKR_EXPORT LocalNode : public Node {
      public:
        int rank() const override { return 0; }
        std::shared_ptr<FileResolver> file_resolver() override { return std::make_shared<TrivialFileResolver>(); }
    };
    AKR_EXPORT std::shared_ptr<Node> local_node() {
        static std::shared_ptr<LocalNode> node;
        static std::once_flag flag;
        std::call_once(flag, [&] { node.reset(new LocalNode()); });
        return node;
    }
    AKR_EXPORT std::shared_ptr<World> local_world() {
        static std::shared_ptr<World> w;
        static std::once_flag flag;
        std::call_once(flag, [&] { w.reset(new LocalWorld()); });
        return w;
    }
} // namespace akari::spmd