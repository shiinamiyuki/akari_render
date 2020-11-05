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
#define _CRT_SECURE_NO_WARNINGS
#include <ImfRgbaFile.h>
#include <ImfHeader.h>
#include <ImfChannelList.h>
#include <ImfStringAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfArray.h>
#include <ImfOutputFile.h>
#include <akari/core/image.h>
#include <akari/core/plugin.h>
namespace akari {
    class HDRImageWriter final : public ImageWriter {
      public:
        bool write(const Image &image, const fs::path &path) {
            auto width = image.resolution()[0];
            // AKR_ASSERT(is_rgb_image(image));
            AKR_ASSERT(path.extension().string() == ".exr");
            Imf::Header header(image.resolution()[0], image.resolution()[1]);
            for (int ch = 0; ch < image.channels(); ch++) {
                header.channels().insert(image.channel_name(ch), Imf::Channel(Imf::FLOAT));
            }
            Imf::OutputFile file(path.string().c_str(), header);
            Imf::FrameBuffer frameBuffer;
            for (int ch = 0; ch < image.channels(); ch++) {
                frameBuffer.insert(image.channel_name(ch),                                // name
                                   Imf::Slice(Imf::FLOAT,                                 // type
                                              (char *)(image.data() + ch),                // base
                                              sizeof(float) * image.channels(),           // xStride
                                              sizeof(float) * image.channels() * width)); // yStride
            }
            file.setFrameBuffer(frameBuffer);
            file.writePixels(image.resolution()[1]);
            return true;
        }
    };
    AKR_EXPORT_PLUGIN(HDRImageWriter, HDRImageWriter, ImageWriter)
} // namespace akari