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
#include <app.h>
#include <algorithm>
#include <GLFW/glfw3.h>
static void glfw_error_callback(int error, const char *description) {
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}
namespace akari::engine {
    void AppWindow::setup_vulkan(const char **extensions, uint32_t extensions_count) {
        {
            vk::InstanceCreateInfo create_info;
            create_info.enabledExtensionCount   = extensions_count;
            create_info.ppEnabledExtensionNames = extensions;
            g_instance                          = vk::createInstanceUnique(create_info, g_allocator);
        }
        // Select GPU
        g_physical_device = g_instance->enumeratePhysicalDevices().front();

        // Select graphics queue family
        auto queueFamilyProperties = g_physical_device.getQueueFamilyProperties();

        size_t graphicsQueueFamilyIndex = std::distance(
            queueFamilyProperties.begin(), std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
                                                        [](vk::QueueFamilyProperties const &qfp) {
                                                            return qfp.queueFlags & vk::QueueFlagBits::eGraphics;
                                                        }));
        AKR_ASSERT(graphicsQueueFamilyIndex < queueFamilyProperties.size());
        g_queue_family = graphicsQueueFamilyIndex;
        // create a UniqueDevice
        float queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo(
            vk::DeviceQueueCreateFlags(), static_cast<uint32_t>(graphicsQueueFamilyIndex), 1, &queuePriority);
        
        vk::DeviceCreateInfo create_info = vk::DeviceCreateInfo(vk::DeviceCreateFlags());
        
        g_device =
            g_physical_device.createDeviceUnique(create_info, deviceQueueCreateInfo));
        {
            std::array<vk::DescriptorPoolSize, 11> pool_sizes = {
                vk::DescriptorPoolSize(vk::DescriptorType::eSampler, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eUniformTexelBuffer, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eStorageTexelBuffer, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eUniformBufferDynamic, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eStorageBufferDynamic, 1000),
                vk::DescriptorPoolSize(vk::DescriptorType::eInputAttachment, 1000)};
            g_descriptor_pool = g_device->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo(
                vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, pool_sizes.size(), pool_sizes));
        }
    }
    void AppWindow::setup_vulkan_window(int width, int height) {
        auto wd     = &g_mainwindow_data;
        wd->Surface = surface;
        // Check for WSI support
        VkBool32 res;
        vkGetPhysicalDeviceSurfaceSupportKHR(g_physical_device, g_queue_family, wd->Surface, &res);
        if (res != VK_TRUE) {
            fprintf(stderr, "Error no WSI support on physical device 0\n");
            exit(-1);
        }
        const VkFormat requestSurfaceImageFormat[]     = {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
                                                      VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM};
        const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
        wd->SurfaceFormat                              = ImGui_ImplVulkanH_SelectSurfaceFormat(
            g_physical_device, wd->Surface, requestSurfaceImageFormat, (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat),
            requestSurfaceColorSpace);
        VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_FIFO_KHR};
        wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(g_physical_device, wd->Surface, &present_modes[0],
                                                              IM_ARRAYSIZE(present_modes));
        IM_ASSERT(g_MinImageCount >= 2);
        ImGui_ImplVulkanH_CreateOrResizeWindow(g_instance.get(), g_physical_device, g_device.get(), wd, g_queue_family,
                                               &(const VkAllocationCallbacks &)g_allocator, width, height,
                                               g_MinImageCount);
    }
    void AppWindow::init() {
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
            exit(1);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(size.x, size.y, title.c_str(), nullptr, nullptr);
        // Setup Vulkan
        if (!glfwVulkanSupported()) {
            printf("GLFW: Vulkan Not Supported\n");
            exit(1);
        }
        uint32_t extensions_count = 0;
        const char **extensions   = glfwGetRequiredInstanceExtensions(&extensions_count);
        setup_vulkan(extensions, extensions_count);
        VkSurfaceKHR _surface;
        const VkAllocationCallbacks &cb = g_allocator;
        CHECK_VK(glfwCreateWindowSurface(g_instance.get(), window, &cb, &_surface));
        surface = vk::SurfaceKHR(_surface);

        // Create Framebuffers
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        setup_vulkan_window(w, h);

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        (void)io;
        // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        // ImGui::StyleColorsClassic();

        // ImGui_ImplGlfw_InitForVulkan(window, true);
        // ImGui_ImplVulkan_InitInfo init_info = {};
        // init_info.Instance                  = g_instance.get();
        // init_info.PhysicalDevice            = g_physical_device;
        // init_info.Device                    = g_device.get();
        // init_info.QueueFamily               = g_queue_family;
        // init_info.Queue                     = g_Queue;
        // init_info.PipelineCache             = g_PipelineCache;
        // init_info.DescriptorPool            = g_DescriptorPool;
        // init_info.Allocator                 = g_Allocator;
        // init_info.MinImageCount             = g_MinImageCount;
        // init_info.ImageCount                = wd->ImageCount;
        // init_info.CheckVkResultFn           = check_vk_result;
        // ImGui_ImplVulkan_Init(&init_info, wd->RenderPass);
    }
    AppWindow::~AppWindow() {
        ImGui_ImplVulkanH_DestroyWindow(g_instance.get(), g_device.get(), &g_mainwindow_data,
                                        &(const VkAllocationCallbacks &)g_allocator);
    }
} // namespace akari::engine