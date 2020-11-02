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

#pragma once
#include <akari/core/akari.h>
#include <akari/core/file.h>
#include <deque>
#include <mutex>
#include <type_traits>
namespace akari::spmd {
    class Node;
    class World;
    using Message = std::vector<char>;
    template <class T>
    class MessageQueue {
        std::deque<T> queue;
        mutable std::mutex m;
        mutable std::condition_variable cv;

      public:
        void push(T msg) {
            std::unique_lock<std::mutex> lock(m);
            queue.emplace_back(std::move(msg));
            cv.notify_all();
        }
        bool has_message() const {
            std::unique_lock<std::mutex> lock(m);
            return !queue.empty();
        }
        T pop() {
            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [=] { return queue.empty(); });
            auto msg = std::move(queue.front());
            queue.pop_front();
            return msg;
        }
    };

    class AKR_EXPORT Node {
      public:
        virtual int rank() const = 0;
        virtual std::shared_ptr<FileResolver> file_resolver() = 0;
    };

    using NodeVisitor = std::function<void(const std::shared_ptr<Node> &)>;
    class ByteChannel;
    class AKR_EXPORT World : public std::enable_shared_from_this<World> {
      public:
        virtual std::shared_ptr<Node> local() const = 0;
        virtual size_t size() const = 0;
        virtual void foreach_nodes(const NodeVisitor &) = 0;
        virtual std::shared_ptr<ByteChannel> channel(const std::shared_ptr<Node> &a,
                                                          const std::shared_ptr<Node> &b) = 0;
        virtual void send(const std::shared_ptr<ByteChannel> &channel, std::string_view) = 0;
        virtual Message receive(const std::shared_ptr<ByteChannel> &channel) = 0;
        virtual void initialize() = 0;
        virtual void finalize() = 0;
    };
    class ByteChannel : public std::enable_shared_from_this<ByteChannel> {
        std::shared_ptr<World> world_;
        std::shared_ptr<MessageQueue<Message>> message;
        uint32_t tag_;

      public:
        ByteChannel(uint32_t tag_, std::shared_ptr<World> world_)
            : tag_(tag_), world_(world_), message(new MessageQueue<Message>()) {}
        uint32_t tag() const { return tag_; }
        std::shared_ptr<World> world() const { return world_; }
        void __push(Message &&msg) { message->push(std::move(msg)); }
        Message __pop() { return message->pop(); }
        void __close() { world_ = nullptr; }
        void send_raw(std::string_view msg) { world_->send(shared_from_this(), msg); }
        Message receive_raw() { return world_->receive(shared_from_this()); }
        bool has_message() const { return message->has_message(); }
        bool closed() const { return world_ != nullptr; }
        template <class T>
        void send(const T &object) {
            static_assert(std::is_trivially_copyable_v<T>);
            send_raw(std::string_view(reinterpret_cast<const char *>(&object), sizeof(T)));
        }
        template <class T>
        T receive() {
            static_assert(std::is_trivially_copyable_v<T>);
            auto msg = receive_raw();
            T object;
            AKR_ASSERT(sizeof(T) == msg.size());
            std::memcpy(&object, msg.data(), sizeof(T));
            return object;
        }
    };

    class AKR_EXPORT AbstractNetworkWorld : public World {
      public:
        virtual void listen(int port) = 0;
        virtual void connect(const std::list<std::pair<std::string, int>> &workers) = 0;
    };
    AKR_EXPORT std::shared_ptr<World> local_world();
} // namespace akari::spmd