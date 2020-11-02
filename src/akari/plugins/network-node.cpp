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

    class FileServer : public std::enable_shared_from_this<FileServer> {
        // std::unordered_set<std::shared_ptr<std::unique_ptr<FileResolver>>> resolvers;

      public:
        std::shared_ptr<ByteChannel> channel;
        FileServer(std::shared_ptr<ByteChannel> channel) : channel(channel) {}
        void run() {
            std::thread([=] {
                auto _ = shared_from_this();
                (void)_;
                while (!channel->closed()) {
                    auto msg = channel->receive_raw();
                    fs::path path(msg.begin(), msg.end());
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
            auto msg = channel->receive_raw();
            return std::make_unique<ByteFileStream>(std::move(msg));
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
        struct ChannelRecord {
            std::shared_ptr<Node> a, b;
            std::weak_ptr<ByteChannel> channel;
            tcp::socket *sock = nullptr;
        };

        std::unordered_map<uint32_t, ChannelRecord> channels;
        std::optional<tcp::acceptor> acceptor_;
        uint32_t tag_counter = 0;
        std::vector<std::shared_ptr<FileServer>> file_servers;

      public:
        std::shared_ptr<Node> local() const override { return local_; }
        size_t size() const override { return 1 + remotes.size(); }
        void foreach_nodes(const NodeVisitor &vis) override {
            vis(local_);
            for (auto &n : remotes) {
                vis(n);
            }
        }

        ChannelRecord &get_channel(const std::shared_ptr<ByteChannel> &ch) { return channels.at(ch->tag()); }
        std::shared_ptr<ByteChannel> channel(const std::shared_ptr<Node> &a, const std::shared_ptr<Node> &b) override {
            if (local()->rank() != 0) {
                if (a != remotes.front() && b != remotes.front()) {
                    AKR_ASSERT("worker node cannot connect to peer");
                }
            }
            auto channel = std::make_shared<ByteChannel>(tag_counter++, shared_from_this());
            tcp::socket *sock = nullptr;
            if (a != b) {
                if (a == local()) {
                    sock = &dyn_cast<NetworkRemoteNode>(b)->sock;
                } else {
                    sock = &dyn_cast<NetworkRemoteNode>(a)->sock;
                }
            }
            channels.emplace(channel->tag(), ChannelRecord{a, b, channel, sock});
            return channel;
        }
        void send(const std::shared_ptr<ByteChannel> &channel, std::string_view msg) override {
            AKR_ASSERT(channel->world().get() == this);
            auto rec = get_channel(channel);
            if (rec.a == rec.b) {
                channel->__push(Message(msg.data(), msg.data() + msg.size()));
            } else {
                uint64_t sz = msg.size();
                uint32_t tag = channel->tag();
                asio::write(*rec.sock, asio::buffer(&tag, sizeof(tag)));
                asio::write(*rec.sock, asio::buffer(&sz, sizeof(uint64_t)));
                asio::write(*rec.sock, asio::buffer(msg.data(), msg.size()));
            }
        }

        Message receive(const std::shared_ptr<ByteChannel> &channel) override {
            AKR_ASSERT(channel->world().get() == this);
            auto rec = get_channel(channel);
            if (rec.a == rec.b || channel->has_message()) {
                return channel->__pop();
            } else {
                while (true) {
                    uint64_t sz;
                    asio::read(*rec.sock, asio::buffer(&sz, sizeof(uint64_t)));
                    uint32_t tag;
                    asio::read(*rec.sock, asio::buffer(&tag, sizeof(tag)));
                    Message msg(sz);
                    asio::read(*rec.sock, asio::buffer(msg.data(), msg.size()));
                    if (channel->tag() == tag) {
                        return msg;
                    } else {
                        channels.at(tag).channel.lock()->__push(std::move(msg));
                    }
                }
            }
        }
        void listen(int port) override {
            tcp::endpoint ep(tcp::v4(), port);
            acceptor_.emplace(ios, ep);
            tcp::socket sock(ios);
            acceptor_->accept(sock);
            int rank = 0;
            asio::read(sock, asio::buffer(&rank, sizeof(int)));
            AKR_ASSERT(rank > 0);
            local_.reset(new NetworkLocalNode(rank));
            remotes.emplace_back(new NetworkRemoteNode(0, std::move(sock)));
            auto ch = channel(remotes.front(), local());
            remotes.front()->resolver = std::make_shared<NetworkFileResolver>(ch);
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
                auto ch = channel(remote, local());
                file_servers.emplace_back(new FileServer(ch));
            }
        }
        void finalize() override {
            for (auto &[_, rec] : channels) {
                if (auto p = rec.channel.lock()) {
                    p->__close();
                }
            }
            channels.clear();
        }
        void initialize() override {}
    };
    AKR_EXPORT_PLUGIN(NetworkWorld, NetworkWorld, AbstractNetworkWorld)
} // namespace akari::spmd