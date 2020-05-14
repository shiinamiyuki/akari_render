// MIT License
//
// Copyright (c) 2019 椎名深雪
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

#include <akari/core/component.h>
#include <fmt/format.h>
namespace akari {
    enum class LogLevel {
        Info,
        Debug,
        Warning,
        Error,
        Fatal
    };
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
    };

    AKR_EXPORT Logger *GetDefaultLogger();

    template <typename... Args> void warning(const char *fmt, Args &&... args) {
        auto logger = GetDefaultLogger();
        logger->Warning(fmt::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args> void error(const char *fmt, Args &&... args) {
        auto logger = GetDefaultLogger();
        logger->Error(fmt::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args> void info(const char *fmt, Args &&... args) {
        auto logger = GetDefaultLogger();
        logger->Info(fmt::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args> void debug(const char *fmt, Args &&... args) {
        auto logger = GetDefaultLogger();
        logger->Debug(fmt::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args> void fatal(const char *fmt, Args &&... args) {
        auto logger = GetDefaultLogger();
        logger->Fatal(fmt::format(fmt, std::forward<Args>(args)...));
    }
} // namespace akari
#endif // AKARIRENDER_LOGGER_H
