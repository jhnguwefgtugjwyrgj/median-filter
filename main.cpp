#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>
#include <cstdint>
#include <cassert>

#include "medianFilter.h"
#include "medianFilterSIMD.h"
#include "medianFilterGPU.h"

namespace {

constexpr size_t image_width = 4096;
constexpr size_t image_height = 4096;
constexpr size_t channels = 3;
constexpr size_t image_stride_bytes = image_width * channels;

std::vector<uint8_t> generate_test_image_rgb(size_t width, size_t height, size_t stride_bytes) {
    std::vector<uint8_t> image(height * stride_bytes, 0);

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> noise(-12, 12);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            const int base_r = static_cast<int>((x * 3 + y * 5) % 256);
            const int base_g = static_cast<int>((x * 7 + y * 2) % 256);
            const int base_b = static_cast<int>((x * 11 + y * 13) % 256);

            int vr = base_r + noise(gen);
            int vg = base_g + noise(gen);
            int vb = base_b + noise(gen);

            if (((x + y) % 29) == 0) { vr += 80; vg -= 40; vb += 60; }
            if (((x * 2 + y) % 31) == 0) { vr -= 80; vg += 30; vb -= 70; }

            vr = std::max(0, std::min(255, vr));
            vg = std::max(0, std::min(255, vg));
            vb = std::max(0, std::min(255, vb));

            const size_t p = y * stride_bytes + x * channels;
            image[p + 0] = static_cast<uint8_t>(vr);
            image[p + 1] = static_cast<uint8_t>(vg);
            image[p + 2] = static_cast<uint8_t>(vb);
        }
    }

    return image;
}

bool compare_images_rgb(const uint8_t* lhs, const uint8_t* rhs, size_t width, size_t height, size_t stride_bytes) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t i = 0; i < width * channels; ++i) {
            if (lhs[y * stride_bytes + i] != rhs[y * stride_bytes + i]) return false;
        }
    }
    return true;
}

template<typename Func>
long long measure_ms(Func&& func) {
    const auto start = std::chrono::high_resolution_clock::now();
    func();
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

} // namespace

int main() {
    std::cout << "2D median filter benchmark 3x3 (RGB)\n";
    std::cout << "Image: " << image_width << " x " << image_height << " x " << channels << "\n";

    auto source_image = generate_test_image_rgb(image_width, image_height, image_stride_bytes);

    std::vector<uint8_t> src_r(image_width * image_height);
    std::vector<uint8_t> src_g(image_width * image_height);
    std::vector<uint8_t> src_b(image_width * image_height);
    for (size_t y = 0; y < image_height; ++y) {
        for (size_t x = 0; x < image_width; ++x) {
            const size_t p = y * image_stride_bytes + x * channels;
            const size_t q = y * image_width + x;
            src_r[q] = source_image[p + 0];
            src_g[q] = source_image[p + 1];
            src_b[q] = source_image[p + 2];
        }
    }

    std::vector<uint8_t> single_thread(image_height * image_stride_bytes);
    std::vector<uint8_t> simd(image_height * image_stride_bytes);
    std::vector<uint8_t> gpu(image_height * image_stride_bytes);

    const auto single_time = measure_ms([&] {
        MedianFilter::median_filter_3x3_rgb(
            source_image.data(),
            single_thread.data(),
            image_width,
            image_height,
            image_stride_bytes
        );
    });
    std::cout << "Single thread version: " << single_time << " ms\n";

    const auto simd_time = measure_ms([&] {
        std::vector<uint8_t> out_r(image_width * image_height);
        std::vector<uint8_t> out_g(image_width * image_height);
        std::vector<uint8_t> out_b(image_width * image_height);
        MedianFilterSIMD::median_filter_3x3(src_r.data(), out_r.data(), image_width, image_height, image_width);
        MedianFilterSIMD::median_filter_3x3(src_g.data(), out_g.data(), image_width, image_height, image_width);
        MedianFilterSIMD::median_filter_3x3(src_b.data(), out_b.data(), image_width, image_height, image_width);

        for (size_t y = 0; y < image_height; ++y) {
            for (size_t x = 0; x < image_width; ++x) {
                const size_t p = y * image_stride_bytes + x * channels;
                const size_t q = y * image_width + x;
                simd[p + 0] = out_r[q];
                simd[p + 1] = out_g[q];
                simd[p + 2] = out_b[q];
            }
        }
    });
    std::cout << "SIMD version: " << simd_time << " ms\n";

    assert(compare_images_rgb(single_thread.data(), simd.data(), image_width, image_height, image_stride_bytes));
    std::cout << "CPU and SIMD results are equal\n";

    if (!MedianFilterGPU::has_gpu()) {
        std::cout << "GPU device not found. GPU benchmark skipped.\n";
        return 0;
    }
    std::cout << "GPU found\n";

    // Прогрев GPU (не учитывается в измерении времени GPU)
    for (int i = 0; i < 2; ++i) {
        MedianFilterGPU::median_filter_3x3_rgb(
            source_image.data(),
            gpu.data(),
            image_width,
            image_height,
            image_stride_bytes
        );
    }

    const auto gpu_time = measure_ms([&] {
        MedianFilterGPU::median_filter_3x3_rgb(
            source_image.data(),
            gpu.data(),
            image_width,
            image_height,
            image_stride_bytes
        );
    });
    std::cout << "GPU version: " << gpu_time << " ms\n";

    assert(compare_images_rgb(single_thread.data(), gpu.data(), image_width, image_height, image_stride_bytes));
    std::cout << "GPU result is equal\n";

    return 0;
}
