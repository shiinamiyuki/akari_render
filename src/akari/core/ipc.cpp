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
#include <akari/core/ipc.h>

// #ifdef AKR_PLATFORM_WINDOWS
// #    ifndef NOMINMAX
// #        define NOMINMAX
// #    endif
// #    include <Ws2tcpip.h>
// #    include <winsock2.h>
// #    undef NOMINMAX
// using socket_t = SOCKET;
// #else
// using socket_t = int;
// #    include <arpa/inet.h>
// #    include <errno.h>
// #    include <netdb.h>
// #    include <netinet/in.h>
// #    include <signal.h>
// #    include <sys/socket.h>
// #    include <unistd.h>
// #    define SOCKET_ERROR   (-1)
// #    define INVALID_SOCKET (-1)
// #endif

// namespace akari {
//     enum SocketError : int {
// #ifdef AKR_PLATFORM_WINDOWS
//         Again = EAGAIN,
//         ConnRefused = WSAECONNREFUSED,
//         WouldBlock = WSAEWOULDBLOCK,
// #else
//         Again = EAGAIN,
//         ConnRefused = ECONNREFUSED,
//         WouldBlock = EWOULDBLOCK,
// #endif
//     };

//     static int closeSocket(socket_t socket) {
// #ifdef AKR_PLATFORM_WINDOWS
//         return closesocket(socket);
// #else
//         return close(socket);
// #endif
//     }
//     static std::atomic<int> numActiveChannels{0};
//     class IPCChannelImpl : public IPCChannel {
//       public:
//         IPCChannelImpl(const std::string &hostname, size_t port) {
//             if (numActiveChannels++ == 0) {
// #ifdef AKR_PLATFORM_WINDOWS
//                 WSADATA wsaData;
//                 int err = WSAStartup(MAKEWORD(2, 2), &wsaData);
//                 if (err != NO_ERROR)
//                     LOG_FATAL("Unable to initialize WinSock: %s", ErrorString(err));
// #else
                
// #endif
//             }
//         }
//         size_t send(const uint8_t *data, size_t size) override {}
//         size_t receive(uint8_t *data, size_t size) override {}
//     };
// } // namespace akari