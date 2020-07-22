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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef AKARIRENDER_ERROR_HPP
#define AKARIRENDER_ERROR_HPP

#include <cstring>
#include <type_traits>
#include <memory>
#include <variant>
#include <optional>
#include <cstring>

namespace akari {
    struct Error {
        explicit Error(const char *msg) {
            _message = std::unique_ptr<char[]>(new char[strlen(msg) + 1]);
            std::memcpy(_message.get(), msg, strlen(msg));
            _message.get()[strlen(msg)] = 0;
        }
        explicit Error(const std::string &msg) : Error(msg.c_str()) {}
        const char *what() const { return _message.get(); }

      private:
        std::unique_ptr<char[]> _message;
    };
    template <typename T = void> struct Expected {
        static_assert(!std::is_same_v<Error, T>, "T cannot be Error");
        using value_type = T;
        using reference = T &;
        using const_reference = const T &;
        Expected(value_type v) : _storage(std::move(v)) {}
        Expected(Error err) : _storage(std::move(err)) {}
        operator bool() const { return has_value(); }
        bool has_value() const { return std::get_if<T>(&_storage) != nullptr; }
        const T *operator->() const { return std::get_if<T>(&_storage); }
        T *operator->() { return std::get_if<T>(&_storage); }
        const Error &error() const { return *std::get_if<Error>(&_storage); }
        Error extract_error() { return std::move(*std::get_if<Error>(&_storage)); }
        const_reference operator*() const { return *std::get_if<T>(&_storage); }
        reference operator*() { return *std::get_if<T>(&_storage); }
        const T &value() const { return *std::get_if<T>(&_storage); }
        T extract_value() const { return std::move(*std::get_if<T>(&_storage)); }

      private:
        std::variant<T, Error> _storage;
    };
    template <> struct Expected<void> {
        Expected() = default;
        Expected(Error err) : _storage(std::move(err)) {}
        operator bool() const { return has_value(); }
        bool has_value() const { return !_storage.has_value(); }
        const Error &error() const { return _storage.value(); }
        Error extract_error() { return std::move(_storage).value(); }

      private:
        std::optional<Error> _storage;
    };
} // namespace akari

#endif