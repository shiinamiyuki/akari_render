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

#include <Akari/Core/Logger.h>
#include <chrono>
#include <mutex>

namespace Akari {
    class DefaultLogger : public Logger {
        decltype(std::chrono::system_clock::now()) start = std::chrono::system_clock::now();
        std::string GetMessageHeader(const std::string &level) {
            std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;
            return fmt::format("[{}] [{:.3f}] ", level, elapsed.count());
        }
        void DoLogMessage(const std::string &msg, FILE *fp = stdout) {
            fprintf(fp, "%s", msg.c_str());
            fflush(fp);
        }

      public:
        void Warning(const std::string &msg) override { DoLogMessage(GetMessageHeader("WARNING").append(msg), stderr); }

        void Error(const std::string &msg) override { DoLogMessage(GetMessageHeader("ERROR").append(msg), stderr); }

        void Info(const std::string &msg) override { DoLogMessage(GetMessageHeader("INFO").append(msg)); }

        void Debug(const std::string &msg) override { DoLogMessage(GetMessageHeader("DEBUG").append(msg)); }

        void Fatal(const std::string &msg) override { DoLogMessage(GetMessageHeader("FATAL").append(msg), stderr); }
    };

    Logger *GetDefaultLogger() {
        static std::unique_ptr<DefaultLogger> logger;
        static std::once_flag flag;
        std::call_once(flag, [&]() { logger = std::make_unique<DefaultLogger>(); });
        return logger.get();
    }
} // namespace Akari
