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

#include <akari/core/logger.h>
#include <chrono>
#include <mutex>
#include <cstdio>
#include <vector>

namespace akari {
    class DefaultLogger : public Logger {
        decltype(std::chrono::system_clock::now()) start = std::chrono::system_clock::now();
        void DoLogMessage(LogLevel level, const std::string &msg, FILE *fp = stdout) {
            std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;

            if (level == LogLevel::Fatal || level == LogLevel::Warning) {
                fprintf(fp, "\u001b[31m");
            } else if (level == LogLevel::Warning) {
                fprintf(fp, "\u001b[33m");
            } else if (level == LogLevel::Info) {
                fprintf(fp, "\u001b[32m");
            }
            switch (level) {
            case LogLevel::Fatal:
                fprintf(fp, "[FATAL] ");
                break;
            case LogLevel::Error:
                fprintf(fp, "[ERROR] ");
                break;
            case LogLevel::Warning:
                fprintf(fp, "[WARNING] ");
                break;
            case LogLevel::Info:
                fprintf(fp, "[INFO] ");
                break;
            case LogLevel::Debug:
                fprintf(fp, "[DEBUG] ");
                break;
            }

            if (level != LogLevel::Fatal) {
                fprintf(fp, "\u001b[0m");
                fmt::print("[{:.3f}] ", elapsed.count());
                fprintf(fp, "%s\n", msg.c_str());

            } else {
                fmt::print("[{:.3f}] ", elapsed.count());
                fprintf(fp, "%s\n", msg.c_str());
                fprintf(fp, "\u001b[0m");
            }
            fflush(fp);
            for (auto it = handlers.begin(); it != handlers.end();) {
                auto &wp = *it;
                if (wp.expired()) {
                    it = handlers.erase(it);
                } else {
                    auto sp = wp.lock();
                    sp->AddMessage(level, msg);
                    it++;
                }
            }
        }
        std::vector<std::weak_ptr<LogHandler>> handlers;

      public:
        void RegisterHandler(const std::shared_ptr<LogHandler> &ptr) override { handlers.emplace_back(ptr); }
        void RemoveHandler(const std::shared_ptr<LogHandler> &ptr) override {
            for (auto it = handlers.begin(); it != handlers.end();) {
                auto &wp = *it;
                if (wp.expired() || wp.lock() == ptr) {
                    it = handlers.erase(it);
                } else {
                    it++;
                }
            }
        }
        void Warning(const std::string &msg) override { DoLogMessage(LogLevel::Warning, msg, stderr); }

        void Error(const std::string &msg) override { DoLogMessage(LogLevel::Error, msg); }

        void Info(const std::string &msg) override { DoLogMessage(LogLevel::Info, msg); }

        void Debug(const std::string &msg) override { DoLogMessage(LogLevel::Debug, msg); }

        void Fatal(const std::string &msg) override { DoLogMessage(LogLevel::Fatal, msg, stderr); }
    }; // namespace akari

    Logger *GetDefaultLogger() {
        static std::unique_ptr<DefaultLogger> logger;
        static std::once_flag flag;
        std::call_once(flag, [&]() { logger = std::make_unique<DefaultLogger>(); });
        return logger.get();
    }
} // namespace akari
