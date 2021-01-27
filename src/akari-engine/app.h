// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>
#include <akari/util.h>
#include <akari-engine/util.h>
#include <akari-engine/ext/imgui/imgui.h>
#include <akari-engine/ext/imgui/imgui_impl_glfw.h>
#include <akari-engine/ext/imgui/imgui_impl_vulkan.h>
// #include <akari-engine/ext/vulkan-hpp/utils.hpp>

namespace akari::engine {
#if 0
    class AKR_EXPORT AppWindow {
        const std::string title;
        ivec2 size;
        GLFWwindow *window = nullptr;

        VkSurfaceKHR surface;
        VkAllocationCallbacks *g_Allocator     = NULL;
        VkInstance g_Instance                  = VK_NULL_HANDLE;
        VkPhysicalDevice g_PhysicalDevice      = VK_NULL_HANDLE;
        VkDevice g_Device                      = VK_NULL_HANDLE;
        uint32_t g_QueueFamily                 = (uint32_t)-1;
        VkQueue g_Queue                        = VK_NULL_HANDLE;
        VkDebugReportCallbackEXT g_DebugReport = VK_NULL_HANDLE;
        VkPipelineCache g_PipelineCache        = VK_NULL_HANDLE;
        VkDescriptorPool g_DescriptorPool      = VK_NULL_HANDLE;

        ImGui_ImplVulkanH_Window g_MainWindowData;
        int g_MinImageCount     = 2;
        bool g_SwapChainRebuild = false;

        
        void setup_vulkan(const char **extensions, uint32_t extensions_count);
        void setup_vulkan_window(int width, int height);
        void frame_render(ImDrawData* draw_data);
        void frame_present();
      public:
        AppWindow(const char *title, ivec2 size) : title(title), size(size) {}
        void show();
    };
#endif
    class AKR_EXPORT AppWindow {
        const std::string title;
        ivec2 size;
        bool enable_validation_layer = true;
        GLFWwindow *window           = nullptr;
        vk::UniqueInstance instance;
        vk::DebugUtilsMessengerEXT debug_messenger;
        vk::PhysicalDevice physical_device;
        uint32_t queue_family = (uint32_t)-1;
        vk::UniqueDevice device;
        vk::Queue graphic_queues;
        void init_window();
        void init_vulkan();
        void cleanup();

      public:
        AppWindow(const char *title, ivec2 size) : title(title), size(size) {
            init_window();
            init_vulkan();
        }
        void show();
        ~AppWindow() { cleanup(); }
    };
} // namespace akari::engine