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

#ifndef AKARIRENDER_PLUGIN_H
#define AKARIRENDER_PLUGIN_H

#include <Akari/Core/Akari.h>
#include <Akari/Core/Component.h>
#include <memory>

namespace Akari {

    typedef void (*SharedLibraryFunc)();
    class AKR_EXPORT SharedLibraryLoader {
        void *handle = nullptr;
        void Load(const char *path);

      public:
        explicit SharedLibraryLoader(const char *path) { Load(path); }
        SharedLibraryFunc GetFuncPointer(const char *name);

        ~SharedLibraryLoader();
    };

    class AKR_EXPORT IPlugin {
      public:
        virtual TypeInfo *GetTypeInfo() = 0;
        virtual const char *GetInterface() = 0;
    };

    class AKR_EXPORT IPluginManager {
      public:
        virtual void SetPluginPath(const char *path) = 0;
        virtual bool LoadPath(const char *path) = 0;
        virtual IPlugin *LoadPlugin(const char *name) = 0;
        virtual void ForeachPlugin(const std::function<void(IPlugin *)> &func) = 0;
        virtual CreateComponentFunc GetCreateFunc(const char *name) = 0;
    };

    AKR_EXPORT IPluginManager *GetPluginManager();

    namespace detail {
        template <std::size_t N, typename T, typename... types> struct get_Nth_type {
            using type = typename get_Nth_type<N - 1, types...>::type;
        };

        template <typename T, typename... types> struct get_Nth_type<0, T, types...> { using type = T; };

        template <typename U, int I> auto _AnyCastV(const std::vector<Any> &args) {
            auto p = (args.at(I)).cast<std::decay_t<U>>();
            if(!p){
                fprintf(stderr, "Invalid conversion of arg %d %s to %s\n", I, args.at(I).type_name(),
                        typeid(U).name());
                throw std::bad_cast();
            }
            return *p;
        }

    } // namespace detail
    template <typename F> struct CreateFuncWrapper {};

#define _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(i) detail::_AnyCastV<typename detail::get_Nth_type<i, Args...>::type, i>(args)
    template <typename U, typename... Args> struct CreateFuncWrapper<std::shared_ptr<U>(Args...)> {
        static std::enable_if_t<sizeof...(Args) <= 8, std::shared_ptr<U>> func(std::shared_ptr<U> (*F)(Args ... args),
                                                                               std::vector<Any> args) {
            static_assert(std::is_base_of_v<Component, U>);
            const std::size_t nArgs = sizeof...(Args);
            if constexpr (nArgs == 0) {
                return F().get();
            } else if constexpr (nArgs == 1) {
                return F(_AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(0));
            } else if constexpr (nArgs == 2) {
                return F(_AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(0), _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(1));
            } else if constexpr (nArgs == 3) {
                return F(_AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(0), _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(1),
                         _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(2));
            } else if constexpr (nArgs == 4) {
                return F(_AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(0), _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(1),
                         _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(2), _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(3));
            } else if constexpr (nArgs == 5) {
                return F(_AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(0), _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(1),
                         _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(2), _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(3),
                         _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(4));
            } else if constexpr (nArgs == 6) {
                return F(_AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(0), _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(1),
                         _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(2), _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(3),
                         _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(4), _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(5));
            } else if constexpr (nArgs == 7) {
                return F(_AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(0), _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(1),
                         _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(2), _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(3),
                         _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(4), _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(5),
                         _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(6));
            } else if constexpr (nArgs == 8) {
                return F(_AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(0), _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(1),
                         _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(2), _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(3),
                         _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(4), _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(5),
                         _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(6), _AKR_DETAIL_PLUGIN_CAST_NTH_TYPE_(7));
            } else {
                panic("No there are too many arguments\n");
                return nullptr;
            }
        }
    };

#define AKR_EXPORT_CREATE(Func)                                                                                        \
    extern "C" AKR_EXPORT void AKARI_PLUGIN_CREATE_FUNC(std::vector<Any> args, std::shared_ptr<Component> &p) {                            \
        p = CreateFuncWrapper<decltype(Func)>::func(Func, std::move(args));                                   \
    }

} // namespace Akari
#endif // AKARIRENDER_PLUGIN_H
