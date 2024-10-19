#include <rust-api.h>
#include <util.h>
#include <openimageio/imageio.h>
#include <memory>
#include <filesystem>

namespace akari::image {
size_t pixel_size(PixelFormat fmt) noexcept {
    switch (fmt) {
        case PixelFormat::R8:
            return 1;
        case PixelFormat::RGBA8:
        case PixelFormat::RF32:
            return 4;
        case PixelFormat::RGBF32:
            return 12;
        case PixelFormat::RGBAF32:
            return 16;
        default:
            AKR_PANIC("Unknown pixel format");
            return 0;
    }
}
size_t pixel_channels(PixelFormat fmt) noexcept {
    switch (fmt) {
        case PixelFormat::R8:
        case PixelFormat::RF32:
            return 1;
        case PixelFormat::RGBA8:
        case PixelFormat::RGBF32:
            return 3;
        case PixelFormat::RGBAF32:
            return 4;
        default:
            AKR_PANIC("Unknown pixel format");
            return 0;
    }
}
extern "C" {
AKR_API extern ImageApi create_image_api() {
    ImageApi api{};
    api.read = [](const char *path, PixelFormat format) -> Image {
        using namespace OIIO;
        auto inp = ImageInput::open(path);
        if (!inp) {
            return Image{};
        }
        const ImageSpec &spec = inp->spec();
        const int xres = spec.width;
        const int yres = spec.height;
        const int nchannels = spec.nchannels;

        auto px_size = pixel_size(format);
        auto pixels = new uint8_t[xres * yres * px_size];
        switch (format) {
            case PixelFormat::R8: {
                inp->read_image(0, 0, 0, 1, TypeDesc::UINT8, pixels);
                break;
            }
            case PixelFormat::RGBA8: {
                inp->read_image(0, 0, 0, 4, TypeDesc::UINT8, pixels);
                break;
            }
            case PixelFormat::RF32: {
                inp->read_image(0, 0, 0, 1, TypeDesc::FLOAT, pixels);
                break;
            }
            case PixelFormat::RGBF32: {
                inp->read_image(0, 0, 0, 3, TypeDesc::FLOAT, pixels);
                break;
            }
            case PixelFormat::RGBAF32: {
                inp->read_image(0, 0, 0, 4, TypeDesc::FLOAT, pixels);
                break;
            }
        }
        inp->close();
        return Image{pixels, static_cast<size_t>(xres), static_cast<size_t>(yres), format};
    };
    api.write = [](const char *path_, const Image &image) -> bool {
        using namespace OIIO;
        const auto path = std::filesystem::path(path_);
        const auto ext = path.extension().string();
        const bool is_hdr = ext == ".hdr" || ext == ".exr";
        auto out = ImageOutput::create(path);
        if (!out) {
            return false;
        }
        const int xres = static_cast<int>(image.width);
        const int yres = static_cast<int>(image.height);
        const int nchannels = static_cast<int>(pixel_channels(image.format));
        TypeDesc type = is_hdr ? TypeDesc::FLOAT : TypeDesc::UINT8;
        ImageSpec spec(xres, yres, nchannels, type);
        out->open(path, spec);
        out->write_image(type, image.data);
        out->close();
        return true;
    };
    api.destroy_image = [](const Image &img) {
        delete[] img.data;
    };
    return api;
}
}
}// namespace akari::image