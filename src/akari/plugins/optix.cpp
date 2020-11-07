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
#include <iostream>
#include <akari/render/denoiser.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
namespace akari::render {
#define OPTIX_CHECK(call)                                                                                              \
    [&] {                                                                                                              \
        if (auto res = call; res != OPTIX_SUCCESS) {                                                                   \
            std::cerr << "OptiX call [ " << #call << " ] "                                                             \
                      << "failed with error: " << optixGetErrorString(res) << ": " << __FILE__ << ":" << __LINE__      \
                      << std::endl;                                                                                    \
            throw std::runtime_error("optix error");                                                                   \
        }                                                                                                              \
    }()

#define CUDA_CHECK(call)                                                                                               \
    [&] {                                                                                                              \
        if (auto res = call; res != 0) {                                                                               \
            std::cerr << "CUDA call [ " << #call << " ] "                                                              \
                      << "failed with error: " << cudaGetErrorString(res) << ": " << __FILE__ << ":" << __LINE__       \
                      << std::endl;                                                                                    \
            throw std::runtime_error("cuda error");                                                                    \
        }                                                                                                              \
    }()
    struct OptixDenoiserInstance {
        unsigned int width;
        unsigned int height;
        OptixDenoiser denoiser;
        float *intensity_buffer;
        void *denoiser_state;
        void *denoiser_scratch;
        OptixDenoiserSizes denoiser_sizes;
        OptixDeviceContext context;
        OptixDenoiserInstance(OptixDeviceContext context, unsigned int width, unsigned int height)
            : width(width), height(height), context(context) {
            OptixDenoiserOptions denoiser_options;

            denoiser_options.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;

            OPTIX_CHECK(optixDenoiserCreate(context, &denoiser_options, &denoiser));
            OPTIX_CHECK(optixDenoiserSetModel(denoiser, OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0));

            OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, width, height, &denoiser_sizes));

            CUDA_CHECK(cudaMalloc(&denoiser_state, denoiser_sizes.stateSizeInBytes));
            CUDA_CHECK(cudaMalloc(&denoiser_scratch, denoiser_sizes.withoutOverlapScratchSizeInBytes));
            OPTIX_CHECK(
                optixDenoiserSetup(denoiser, nullptr, width, height, reinterpret_cast<CUdeviceptr>(denoiser_state),
                                   denoiser_sizes.stateSizeInBytes, reinterpret_cast<CUdeviceptr>(denoiser_scratch),
                                   denoiser_sizes.withoutOverlapScratchSizeInBytes));

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&intensity_buffer), sizeof(float)));
        }

        static OptixImage2D create_image(int width, int height) noexcept {

            auto pixel_stride = sizeof(float3);
            auto row_stride = pixel_stride * width;
            auto buffer_size = row_stride * height;

            void *buffer = nullptr;
            CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

            OptixImage2D image;
            image.data = reinterpret_cast<CUdeviceptr>(buffer);
            image.width = width;
            image.height = height;
            image.pixelStrideInBytes = 0;
            image.rowStrideInBytes = row_stride;
            image.format = OPTIX_PIXEL_FORMAT_FLOAT3;

            return image;
        }
        static OptixImage2D create_image(Image &image) noexcept {
            unsigned int width = image.resolution()[0];
            unsigned int height = image.resolution()[1];
            auto pixel_stride = sizeof(float3);
            auto row_stride = pixel_stride * width;
            auto buffer_size = row_stride * height;

            void *buffer = nullptr;
            CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

            OptixImage2D gpu_image;
            gpu_image.data = reinterpret_cast<CUdeviceptr>(buffer);
            gpu_image.width = width;
            gpu_image.height = height;
            gpu_image.pixelStrideInBytes = 0;
            gpu_image.rowStrideInBytes = row_stride;
            gpu_image.format = OPTIX_PIXEL_FORMAT_FLOAT3;
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(gpu_image.data), image.data(),
                                  gpu_image.rowStrideInBytes * gpu_image.height, cudaMemcpyDefault));
            return gpu_image;
        }
        std::optional<Image> denoise3(Image &color, Image &albedo, Image &normal) {
            OptixDenoiserParams denoiser_params;
            denoiser_params.hdrIntensity = reinterpret_cast<CUdeviceptr>(intensity_buffer);
            denoiser_params.blendFactor = 0.0f;
            OptixImage2D output = create_image(width, height);
            OptixImage2D inputs[] = {create_image(color), create_image(albedo), create_image(normal)};
            OPTIX_CHECK(optixDenoiserComputeIntensity(
                denoiser, nullptr, &inputs[0], reinterpret_cast<CUdeviceptr>(intensity_buffer),
                reinterpret_cast<CUdeviceptr>(denoiser_scratch), denoiser_sizes.withoutOverlapScratchSizeInBytes));

            OPTIX_CHECK(optixDenoiserInvoke(denoiser, nullptr, &denoiser_params, 0, 0, inputs, 3, 0, 0, &output,
                                            reinterpret_cast<CUdeviceptr>(denoiser_scratch),
                                            denoiser_sizes.withoutOverlapScratchSizeInBytes));
            Image result = rgb_image(color.resolution());
            CUDA_CHECK(cudaMemcpy(result.data(), reinterpret_cast<void *>(output.data),
                                  output.rowStrideInBytes * output.height, cudaMemcpyDefault));
            for (OptixImage2D &img : inputs) {
                CUDA_CHECK(cudaFree(reinterpret_cast<void *>(img.data)));
            }
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(output.data)));
            return result;
        }

        ~OptixDenoiserInstance() {
            CUDA_CHECK(cudaFree(intensity_buffer));
            CUDA_CHECK(cudaFree(denoiser_state));
            CUDA_CHECK(cudaFree(denoiser_scratch));
            OPTIX_CHECK(optixDenoiserDestroy(denoiser));
        }
    };
    class OptixAIDenoiser final : public Denoiser {
        OptixDeviceContext context;

      public:
        OptixAIDenoiser() {
            CUDA_CHECK(cudaFree(nullptr));
            OPTIX_CHECK(optixInit());
            OPTIX_CHECK(optixDeviceContextCreate(nullptr, nullptr, &context));
        }
        ~OptixAIDenoiser() { OPTIX_CHECK(optixDeviceContextDestroy(context)); }
        void add_aov_requests(RenderInput &inputs) {
            inputs.requested_aovs["normal"].required_variance = false;
            inputs.requested_aovs["albedo"].required_variance = false;
        }

        std::optional<Image> denoise(const Scene *scene, RenderOutput &aov) override {

            std::optional<Image> output;
            Image color = aov.aovs["color"].value->to_rgb_image();
            Image albedo = aov.aovs["albedo"].value->to_rgb_image();
            Image normal = aov.aovs["normal"].value->to_rgb_image();
            OptixDenoiserInstance denoiser(context, color.resolution()[0], color.resolution()[1]);
            AKR_ASSERT(is_rgb_image(color));
            AKR_ASSERT(is_rgb_image(albedo));
            AKR_ASSERT(is_rgb_image(normal));
            if (aov.aovs.find("first_hit_normal") != aov.aovs.end() &&
                aov.aovs.find("first_hit_albedo") != aov.aovs.end()) {
                info("full denoising (normal, albedo, color)");
                Image first_hit_normal = aov.aovs["first_hit_normal"].value->to_rgb_image();
                Image first_hit_albedo = aov.aovs["first_hit_albedo"].value->to_rgb_image();
                std::optional<Image> filtered_normal = denoiser.denoise3(normal, first_hit_albedo, first_hit_normal);
                std::optional<Image> filtered_albedo = denoiser.denoise3(albedo, first_hit_albedo, first_hit_normal);
                if (!filtered_albedo || !filtered_normal) {
                    return std::nullopt;
                }
                normal = aov.aovs["normal"].value->to_rgb_image();
                output = denoiser.denoise3(color, *filtered_albedo, *filtered_normal);
            } else {
                output = denoiser.denoise3(color, albedo, color);
            }
            info("optix denoise complete");
            return output;
        }
    };
    AKR_EXPORT_PLUGIN(OptixDenoiser, OptixAIDenoiser, Denoiser);
} // namespace akari::render