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

#ifndef AKARIRENDER_LOGGER_H
#define AKARIRENDER_LOGGER_H

#include <akari/common/platform.h>
#include <akari/common/color.h>
#include <fmt/format.h>
template <typename T, int N>
struct fmt::formatter<akari::Array<T, N>> {

    template <typename FormatContext>
    auto format(const akari::Array<T, N> &a, FormatContext &ctx) {
        // auto format(const point &p, FormatContext &ctx) -> decltype(ctx.out()) // c++11
        // ctx.out() is an output iterator to write to.
        auto it = format_to(ctx.out(), "[");
        for (int i = 0; i < N; i++) {
            it = format_to(it, "{}", a[i]);
            if (i != N - 1) {
                it = format_to(it, ", ");
            }
        }
        it = format_to(it, "]");
        return it;
    }
    constexpr auto parse(format_parse_context &ctx) {
        // auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) // c++11
        // [ctx.begin(), ctx.end()) is a character range that contains a part of
        // the format string starting from the format specifications to be parsed,
        // e.g. in
        //
        //   fmt::format("{:f} - point of interest", point{1, 2});
        //
        // the range will contain "f} - point of interest". The formatter should
        // parse specifiers until '}' or the end of the range. In this example
        // the formatter should parse the 'f' specifier and return an iterator
        // pointing to '}'.

        // Parse the presentation format and store it in the formatter:
        auto it = ctx.begin(), end = ctx.end();
        // Check if reached the end of the range:
        if (it != end && *it != '}')
            throw format_error("invalid format");

        // Return an iterator past the end of the parsed range:
        return it;
    }
};
template <typename T, int N>
struct fmt::formatter<akari::Color<T, N>> : fmt::formatter<akari::Array<T, N>> {
    template <typename FormatCtx>
    auto format(const akari::Color<T, N> &a, FormatCtx &ctx) {
        return fmt::formatter<akari::Array<T, N>>::format(a, ctx);
    }
};
namespace akari {

    enum class LogLevel { Info, Debug, Warning, Error, Fatal };
    class LogHandler {
      public:
        virtual void AddMessage(LogLevel level, const std::string &msg) = 0;
    };
    class AKR_EXPORT Logger {
      public:
        virtual void RegisterHandler(const std::shared_ptr<LogHandler> &) = 0;
        virtual void RemoveHandler(const std::shared_ptr<LogHandler> &) = 0;
        virtual void Warning(const std::string &msg) = 0;
        virtual void Error(const std::string &msg) = 0;
        virtual void Info(const std::string &msg) = 0;
        virtual void Debug(const std::string &msg) = 0;
        virtual void Fatal(const std::string &msg) = 0;
        virtual ~Logger() = default;
    };

    AKR_EXPORT Logger *GetDefaultLogger();

    template <typename... Args>
    void warning(const char *fmt, Args &&... args) {
        auto logger = GetDefaultLogger();
        logger->Warning(fmt::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void error(const char *fmt, Args &&... args) {
        auto logger = GetDefaultLogger();
        logger->Error(fmt::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void info(const char *fmt, Args &&... args) {
        auto logger = GetDefaultLogger();
        logger->Info(fmt::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void debug(const char *fmt, Args &&... args) {
        auto logger = GetDefaultLogger();
        logger->Debug(fmt::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void fatal(const char *fmt, Args &&... args) {
        auto logger = GetDefaultLogger();
        logger->Fatal(fmt::format(fmt, std::forward<Args>(args)...));
    }
} // namespace akari
#endif // AKARIRENDER_LOGGER_H
