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
#include <json.hpp>
#include <asio.hpp>
#include <optional>
#include <akari/core/file.h>
#include <akari/core/spmd.h>
#include <unordered_set>
namespace akari::spmd {
    using asio::ip::tcp;
    static constexpr uint32_t FILE_SERVER_TAG = 4000;
    static constexpr uint32_t INTERNAL_TAG = 32765;
    class FileServer : public std::enable_shared_from_this<FileServer> {
        // std::unordered_set<std::shared_ptr<std::unique_ptr<FileResolver>>> resolvers;

      public:
        std::shared_ptr<ByteChannel> channel;
        FileServer(std::shared_ptr<ByteChannel> channel) : channel(channel) {}
        void run() {
            debug("file server listening on {}", channel->record.tag);
            std::thread([=] {
                auto _ = shared_from_this();
                (void)_;
                while (!channel->closed()) {
                    auto msg = channel->receive();
                    if (!msg)
                        break;
                    fs::path path(msg->begin(), msg->end());
                    debug("reading {}", path.string());
                    auto stream = resolve_file(path);
                    auto content = stream->read_all();
                    channel->send(std::string_view(content.data(), content.size()));
                }
            }).detach();
        }
    };
    class NetworkFileResolver final : public FileResolver {
      public:
        std::shared_ptr<ByteChannel> channel;
        NetworkFileResolver(std::shared_ptr<ByteChannel> channel) : channel(channel) {}
        std::unique_ptr<FileStream> resolve(const fs::path &path) override {
            auto s = path.string();
            channel->send(std::string_view(s.data(), s.size()));
            auto msg = channel->receive();
            if (!msg)
                return nullptr;
            return std::make_unique<ByteFileStream>(std::move(*msg));
        }
    };
    class AKR_EXPORT NetworkNode : public Node {
      public:
        int rank_;
        int rank() const override { return rank_; }
        NetworkNode(int rank_) : rank_(rank_) {}
    };
    class AKR_EXPORT NetworkLocalNode : public NetworkNode {
      public:
        using NetworkNode::NetworkNode;
        std::shared_ptr<FileResolver> file_resolver() override { return std::make_shared<TrivialFileResolver>(); }
    };
    class AKR_EXPORT NetworkRemoteNode : public NetworkNode {
      public:
        asio::ip::tcp::socket sock;
        std::shared_ptr<FileResolver> resolver;
        NetworkRemoteNode(int rank, asio::ip::tcp::socket &&sock) : NetworkNode(rank), sock(std::move(sock)) {}
        std::shared_ptr<FileResolver> file_resolver() override { return resolver; }
    };
    // class AKR_EXPORT NetworkRemoteNodeMaster : public NetworkRemoteNode {
    //   public:
    //     asio::ip::tcp::socket sock;
    //     NetworkRemoteNodeMaster(int rank_, asio::io_service &ios, const asio::ip::tcp::endpoint &ep)
    //         : NetworkRemoteNode(rank_), sock(ios, ep.protocol()) {
    //         sock.connect(ep);
    //     }

    //     asio::ip::tcp::socket &get_socket() override { return sock; }
    // };
    // class AKR_EXPORT NetworkRemoteNodeWorker : public NetworkRemoteNode {
    //   public:
    //     asio::ip::tcp::acceptor acceptor_;
    //     asio::ip::tcp::socket sock;
    //     NetworkRemoteNodeWorker(int rank_, asio::io_service &ios, const asio::ip::tcp::endpoint &ep)
    //         : NetworkRemoteNode(rank_), acceptor_(ios, ep), sock(ios) {
    //         acceptor_.accept(sock);
    //     }
    //     asio::ip::tcp::socket &get_socket() override { return sock; }
    // };
    class AKR_EXPORT NetworkWorld final : public AbstractNetworkWorld {
        asio::io_service ios;
        std::shared_ptr<NetworkLocalNode> local_;
        std::list<std::shared_ptr<NetworkRemoteNode>> remotes;
        // std::mutex m;
        // std::unordered_map<std::shared_ptr<Node>, ChannelRecord> records;
        std::mutex ch_mutex;
        std::unordered_map<ChannelRecord, std::pair<tcp::socket *, std::weak_ptr<ByteChannel>>, ChannelRecord::Hash,
                           ChannelRecord::Eq>
            channels;
        std::optional<tcp::acceptor> acceptor_;
        std::vector<std::shared_ptr<FileServer>> file_servers;
        std::vector<std::shared_ptr<ByteChannel>> internals;
        enum InternalMessage { request_sync, ack_sync };
        std::atomic_bool is_shutdown = false;

      public:
        std::shared_ptr<Node> local() const override { return local_; }
        size_t size() const override { return 1 + remotes.size(); }
        void foreach_node(const NodeVisitor &vis) override {
            vis(local_);
            for (auto &n : remotes) {
                vis(n);
            }
        }
        std::shared_ptr<ByteChannel> channel(uint32_t tag, const std::shared_ptr<Node> &a,
                                             const std::shared_ptr<Node> &b) override {
            std::unique_lock<std::mutex> lock(ch_mutex);
            if (local()->rank() != 0) {
                if (a != remotes.front() && b != remotes.front()) {
                    AKR_ASSERT("worker node cannot connect to peer");
                }
            }
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

            tcp::socket *sock = nullptr;
            if (a != b) {
                if (a == local()) {
                    sock = &dyn_cast<NetworkRemoteNode>(b)->sock;
                } else {
                    sock = &dyn_cast<NetworkRemoteNode>(a)->sock;
                }
            }
            channels.emplace(rec, std::make_pair(sock, channel));
            return channel;
        }
        void send(const std::shared_ptr<ByteChannel> &channel, std::string_view msg) override {
            AKR_ASSERT(channel->world().get() == this);
            std::unique_lock<std::mutex> lock(ch_mutex);
            auto rec = channel->record;
            auto sock = channels[rec].first;
            lock.unlock();
            if (rec.nodes.first == rec.nodes.second) {
                channel->__push(Message(msg.data(), msg.data() + msg.size()));
            } else {
                auto dest = rec.nodes.first == local()->rank() ? rec.nodes.second : rec.nodes.first;
                uint64_t sz = msg.size();
                uint32_t tag = rec.tag;
                debug("write {} bytes; {} -> {}; tag={}", sz, local()->rank(), dest, tag);
                asio::write(*sock, asio::buffer(&tag, sizeof(tag)));
                asio::write(*sock, asio::buffer(&sz, sizeof(uint64_t)));
                asio::write(*sock, asio::buffer(msg.data(), msg.size()));
            }
        }

        std::optional<Message> receive(const std::shared_ptr<ByteChannel> &channel) override {
            AKR_ASSERT(channel->world().get() == this);
            debug("trying to receive from {}", channel->record.tag);
            auto rec = channel->record;
            return channel->__pop();
        }
        void handle_message(int a, int b, tcp::socket *sock) {
            if (a > b) {
                std::swap(a, b);
            }
            while (!is_shutdown && sock->is_open()) {
                debug("reading from socket");
                uint32_t tag;
                asio::read(*sock, asio::buffer(&tag, sizeof(tag)));
                // debug("reading from socket done tag");
                uint64_t sz;
                asio::read(*sock, asio::buffer(&sz, sizeof(uint64_t)));
                // debug("reading from socket done size");
                Message msg(sz);
                asio::read(*sock, asio::buffer(msg.data(), msg.size()));
                debug("receiving {} bytes from {}", sz, tag);
                ChannelRecord rec{std::make_pair(a, b), tag};
                std::unique_lock<std::mutex> lock(ch_mutex);
                channels.at(rec).second.lock()->__push(std::move(msg));
            }
        }
        void handle_messages() {
            for (auto &remote : remotes) {
                std::thread([=] { handle_message(local()->rank(), remote->rank(), &remote->sock); }).detach();
            }
        }
        void listen(int port) override {
            tcp::endpoint ep(tcp::v4(), port);
            acceptor_.emplace(ios, ep);
            tcp::socket sock(ios);
            acceptor_->accept(sock);
            int rank = 0;
            asio::read(sock, asio::buffer(&rank, sizeof(int)));
            info("rank={}", rank);
            AKR_ASSERT(rank > 0);
            local_.reset(new NetworkLocalNode(rank));
            remotes.emplace_back(new NetworkRemoteNode(0, std::move(sock)));
            auto ch = channel(FILE_SERVER_TAG, remotes.front(), local());
            remotes.front()->resolver = std::make_shared<NetworkFileResolver>(ch);
            ch = channel(INTERNAL_TAG, remotes.front(), local());
            internals.push_back(ch);
            handle_messages();
        }
        void connect(const std::list<std::pair<std::string, int>> &workers) override {
            local_.reset(new NetworkLocalNode(0));
            int r = 1;
            for (auto &w : workers) {
                tcp::endpoint ep(asio::ip::address::from_string(w.first), w.second);
                tcp::socket sock(ios, ep.protocol());
                sock.connect(ep);
                asio::write(sock, asio::buffer(&r, sizeof(int)));
                remotes.emplace_back(new NetworkRemoteNode(r, std::move(sock)));
                r++;
            }
            for (auto &remote : remotes) {
                auto ch = channel(FILE_SERVER_TAG, remote, local());
                file_servers.emplace_back(new FileServer(ch));
                ch = channel(INTERNAL_TAG, remote, local());
                internals.push_back(ch);
            }
            for (auto &server : file_servers) {
                server->run();
            }
            handle_messages();
        }
        void barrier() override {
            if (local()->rank() == 0) {
                for (auto &internal : internals) {
                    internal->send_object(InternalMessage::request_sync);
                }
                for (auto &internal : internals) {
                    while (true) {
                        auto msg = *internal->receive_object<InternalMessage>();
                        if (msg == InternalMessage::ack_sync) {
                            break;
                        }
                    }
                }
            } else {
                AKR_ASSERT(internals.size() == 1);
                for (auto &internal : internals) {
                    while (true) {
                        auto msg = *internal->receive_object<InternalMessage>();
                        if (msg == InternalMessage::request_sync) {
                            internal->send_object(InternalMessage::ack_sync);
                            break; // good
                        }
                    }
                }
            }
        }
        void finalize() override {
            barrier();
            for (auto &rec : channels) {
                if (auto p = rec.second.second.lock()) {
                    p->__close();
                }
            }
            channels.clear();
            if (local()->rank() == 0) {
                file_servers.clear();
            }
            remotes.clear();
        }
        void initialize() override {}
    };
    AKR_EXPORT_PLUGIN(NetworkWorld, NetworkWorld, AbstractNetworkWorld)
} // namespace akari::spmd