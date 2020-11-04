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

#include <akari/core/comm.h>
#include <deque>
#include <mutex>
namespace akari::comm {
    class AKR_EXPORT LocalWorld : public World {
        std::shared_ptr<Node> local_;
        // std::mutex m;
        // std::unordered_map<std::shared_ptr<Node>, ChannelRecord> records;
        std::unordered_map<ChannelRecord, std::weak_ptr<ByteChannel>, ChannelRecord::Hash, ChannelRecord::Eq> channels;

      public:
        std::shared_ptr<Node> local() const override { return local_; }
        void foreach_node(const NodeVisitor &vis) override { vis(local_); }
        std::shared_ptr<ByteChannel> channel(uint32_t tag, const std::shared_ptr<Node> &a,
                                             const std::shared_ptr<Node> &b) override {
            ChannelRecord rec;
            rec.tag = tag;
            rec.nodes.first = a->rank();
            rec.nodes.second = b->rank();
            if (rec.nodes.first > rec.nodes.second) {
                std::swap(rec.nodes.first, rec.nodes.second);
            }
            if (channels.find(rec) != channels.end()) {
                AKR_ASSERT("channel already created");
            }
            auto channel = std::make_shared<ByteChannel>(rec, shared_from_this());
            channels.emplace(rec, channel);
            return channel;
        }
        void send(const std::shared_ptr<ByteChannel> &channel, std::string_view msg) override {
            AKR_ASSERT(channel->world().get() == this);
            channel->__push(Message(msg.data(), msg.data() + msg.size()));
        }
        std::optional<Message> receive(const std::shared_ptr<ByteChannel> &channel) override {
            // std::lock_guard<std::mutex> lock(m);
            return channel->__pop();
        }
        void initialize() override {}
        size_t size() const override { return 1; }
        void finalize() override {
            for (auto &rec : channels) {
                if (auto ch = rec.second.lock()) {
                    ch->__close();
                }
            }
            channels.clear();
        }
        void barrier() override {}
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

    static std::shared_ptr<World> comm_world_;
    AKR_EXPORT std::shared_ptr<World> comm_world() { return comm_world_; }
    AKR_EXPORT void init_comm_world(const std::shared_ptr<World> &world) { comm_world_ = world; }
    AKR_EXPORT void finalize_comm_world() { comm_world_->finalize(); }
} // namespace akari::comm