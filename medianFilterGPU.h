#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include "utils.h"



class MedianFilterGPU {
private:
    static float median_7(float arr[7]);
    static uint8_t median_9(uint8_t window[9]);
    static sycl::queue create_gpu_queue();

public:
    static bool has_gpu();
    static void median_filter_7(const float* input, float* output, size_t length);
    static void median_filter_3x3(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride);
    static void median_filter_3x3_rgb(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride_bytes);
};

float MedianFilterGPU::median_7(float arr[7]) {
    cond_swap(arr[0], arr[6]);
    cond_swap(arr[2], arr[3]);
    cond_swap(arr[4], arr[5]);

    cond_swap(arr[0], arr[2]);
    cond_swap(arr[1], arr[4]);
    cond_swap(arr[3], arr[6]);

    arr[1] = get_max(arr[0], arr[1]);
    cond_swap(arr[2], arr[5]);
    cond_swap(arr[3], arr[4]);

    arr[2] = get_max(arr[1], arr[2]);
    arr[4] = get_min(arr[4], arr[6]);

    arr[3] = get_max(arr[2], arr[3]);
    arr[4] = get_min(arr[4], arr[5]);

    arr[3] = get_min(arr[3], arr[4]);

    return arr[3];
}

uint8_t MedianFilterGPU::median_9(uint8_t window[9]) {
    cond_swap(window[0], window[3]);
    cond_swap(window[1], window[7]);
    cond_swap(window[2], window[5]);
    cond_swap(window[4], window[8]);

    cond_swap(window[0], window[7]);
    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[8]);
    cond_swap(window[5], window[6]);

    window[2] = get_max(window[0], window[2]);
    cond_swap(window[1], window[3]);
    cond_swap(window[4], window[5]);
    window[7] = get_min(window[7], window[8]);

    window[4] = get_max(window[1], window[4]);
    window[3] = get_min(window[3], window[6]);
    window[5] = get_min(window[5], window[7]);

    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[5]);

    window[3] = get_max(window[2], window[3]);
    window[4] = get_min(window[4], window[5]);

    window[4] = get_max(window[3], window[4]);

    return window[4];
}

sycl::queue MedianFilterGPU::create_gpu_queue() {
    return sycl::queue(sycl::gpu_selector_v);
}

bool MedianFilterGPU::has_gpu() {
    for (const auto& platform : sycl::platform::get_platforms()) {
        for (const auto& device : platform.get_devices()) {
            if (device.is_gpu()) return true;
        }
    }
    return false;
}

void MedianFilterGPU::median_filter_7(const float* input, float* output, size_t length) {
    if (length == 0) return;

    sycl::queue q = create_gpu_queue();

    size_t N = length;
    float* d_input = sycl::malloc_shared<float>(N, q);
    float* d_output = sycl::malloc_shared<float>(N, q);

    q.memcpy(d_input, input, N * sizeof(float)).wait();

    if (N < 7) {
        float window[7];
        for (size_t i = 0; i < N; ++i) {
            for (int j = -3; j <= 3; ++j) {
                int idx = static_cast<int>(i) + j;
                if (idx < 0) window[j + 3] = d_input[0];
                else if (idx >= static_cast<int>(N)) window[j + 3] = d_input[N - 1];
                else window[j + 3] = d_input[idx];
            }
            d_output[i] = median_7(window);
        }

        q.memcpy(output, d_output, N * sizeof(float)).wait();
        sycl::free(d_input, q);
        sycl::free(d_output, q);
        return;
    }

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(N - 6), [=](sycl::id<1> idx) {
            size_t i = idx[0] + 3;
            float window[7];

            for (int j = -3; j <= 3; ++j) window[j + 3] = d_input[i + j];

            d_output[i] = median_7(window);
        });
    });
    q.wait();

    float window[7];

    for (size_t i = 0; i < 3 && i < N; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = static_cast<int>(i) + j;
            if (idx < 0) window[j + 3] = d_input[0];
            else if (idx >= static_cast<int>(N)) window[j + 3] = d_input[N - 1];
            else window[j + 3] = d_input[idx];
        }
        d_output[i] = median_7(window);
    }

    for (size_t i = (N > 3 ? N - 3 : 0); i < N; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = static_cast<int>(i) + j;
            if (idx < 0) window[j + 3] = d_input[0];
            else if (idx >= static_cast<int>(N)) window[j + 3] = d_input[N - 1];
            else window[j + 3] = d_input[idx];
        }
        d_output[i] = median_7(window);
    }

    q.memcpy(output, d_output, N * sizeof(float)).wait();

    sycl::free(d_input, q);
    sycl::free(d_output, q);
}

void MedianFilterGPU::median_filter_3x3(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride) {
    if (width == 0 || height == 0) return;
    if (stride < width) return;

    sycl::queue q = create_gpu_queue();

    const size_t total_size = height * stride;
    uint8_t* d_input = sycl::malloc_shared<uint8_t>(total_size, q);
    uint8_t* d_output = sycl::malloc_shared<uint8_t>(total_size, q);

    q.memcpy(d_input, input, total_size * sizeof(uint8_t)).wait();
    q.memcpy(d_output, input, total_size * sizeof(uint8_t)).wait();

    constexpr size_t block_x = 16;
    constexpr size_t block_y = 16;
    const size_t local_width = block_x + 2;
    const size_t local_height = block_y + 2;

    const size_t grid_x = ((width + block_x - 1) / block_x) * block_x;
    const size_t grid_y = ((height + block_y - 1) / block_y) * block_y;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 2> tile(sycl::range<2>(local_height, local_width), h);

        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(grid_y, grid_x), sycl::range<2>(block_y, block_x)),
            [=](sycl::nd_item<2> item) {
                const size_t global_y = item.get_global_id(0);
                const size_t global_x = item.get_global_id(1);
                const size_t local_y = item.get_local_id(0);
                const size_t local_x = item.get_local_id(1);

                const size_t clamped_y = global_y < height ? global_y : height - 1;
                const size_t clamped_x = global_x < width ? global_x : width - 1;

                tile[local_y + 1][local_x + 1] = d_input[clamped_y * stride + clamped_x];

                if (local_x == 0) {
                    const size_t left_x = clamped_x > 0 ? clamped_x - 1 : 0;
                    tile[local_y + 1][0] = d_input[clamped_y * stride + left_x];
                }
                if (local_x + 1 == block_x) {
                    const size_t right_x = clamped_x + 1 < width ? clamped_x + 1 : width - 1;
                    tile[local_y + 1][local_width - 1] = d_input[clamped_y * stride + right_x];
                }
                if (local_y == 0) {
                    const size_t top_y = clamped_y > 0 ? clamped_y - 1 : 0;
                    tile[0][local_x + 1] = d_input[top_y * stride + clamped_x];
                }
                if (local_y + 1 == block_y) {
                    const size_t bottom_y = clamped_y + 1 < height ? clamped_y + 1 : height - 1;
                    tile[local_height - 1][local_x + 1] = d_input[bottom_y * stride + clamped_x];
                }

                if (local_x == 0 && local_y == 0) {
                    const size_t top_y = clamped_y > 0 ? clamped_y - 1 : 0;
                    const size_t left_x = clamped_x > 0 ? clamped_x - 1 : 0;
                    tile[0][0] = d_input[top_y * stride + left_x];
                }
                if (local_x + 1 == block_x && local_y == 0) {
                    const size_t top_y = clamped_y > 0 ? clamped_y - 1 : 0;
                    const size_t right_x = clamped_x + 1 < width ? clamped_x + 1 : width - 1;
                    tile[0][local_width - 1] = d_input[top_y * stride + right_x];
                }
                if (local_x == 0 && local_y + 1 == block_y) {
                    const size_t bottom_y = clamped_y + 1 < height ? clamped_y + 1 : height - 1;
                    const size_t left_x = clamped_x > 0 ? clamped_x - 1 : 0;
                    tile[local_height - 1][0] = d_input[bottom_y * stride + left_x];
                }
                if (local_x + 1 == block_x && local_y + 1 == block_y) {
                    const size_t bottom_y = clamped_y + 1 < height ? clamped_y + 1 : height - 1;
                    const size_t right_x = clamped_x + 1 < width ? clamped_x + 1 : width - 1;
                    tile[local_height - 1][local_width - 1] = d_input[bottom_y * stride + right_x];
                }

                item.barrier(sycl::access::fence_space::local_space);

                if (global_x < width && global_y < height) {
                    uint8_t window[9];
                    size_t idx = 0;
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            window[idx++] = tile[local_y + 1 + dy][local_x + 1 + dx];
                        }
                    }

                    d_output[global_y * stride + global_x] = median_9(window);
                }
            }
        );
    }).wait();

    q.memcpy(output, d_output, total_size * sizeof(uint8_t)).wait();

    sycl::free(d_input, q);
    sycl::free(d_output, q);
}

void MedianFilterGPU::median_filter_3x3_rgb(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride_bytes) {
    if (width == 0 || height == 0) return;
    if (stride_bytes < width * 3) return;

    sycl::queue q = create_gpu_queue();

    const size_t total_size = height * stride_bytes;
    uint8_t* d_input = sycl::malloc_shared<uint8_t>(total_size, q);
    uint8_t* d_output = sycl::malloc_shared<uint8_t>(total_size, q);

    q.memcpy(d_input, input, total_size * sizeof(uint8_t)).wait();
    q.memcpy(d_output, input, total_size * sizeof(uint8_t)).wait();

    // На некоторых iGPU стабильнее наивный kernel (без local_accessor).
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
            const size_t y = id[0];
            const size_t x = id[1];

            const size_t y0 = (y > 0) ? (y - 1) : 0;
            const size_t y1 = y;
            const size_t y2 = (y + 1 < height) ? (y + 1) : (height - 1);

            const size_t x0 = (x > 0) ? (x - 1) : 0;
            const size_t x1 = x;
            const size_t x2 = (x + 1 < width) ? (x + 1) : (width - 1);

            constexpr size_t channels = 3;
            const size_t p0 = x0 * channels;
            const size_t p1 = x1 * channels;
            const size_t p2 = x2 * channels;

            for (size_t c = 0; c < channels; ++c) {
                uint8_t window[9];
                window[0] = d_input[y0 * stride_bytes + p0 + c];
                window[1] = d_input[y0 * stride_bytes + p1 + c];
                window[2] = d_input[y0 * stride_bytes + p2 + c];
                window[3] = d_input[y1 * stride_bytes + p0 + c];
                window[4] = d_input[y1 * stride_bytes + p1 + c];
                window[5] = d_input[y1 * stride_bytes + p2 + c];
                window[6] = d_input[y2 * stride_bytes + p0 + c];
                window[7] = d_input[y2 * stride_bytes + p1 + c];
                window[8] = d_input[y2 * stride_bytes + p2 + c];
                d_output[y * stride_bytes + x * channels + c] = median_9(window);
            }
        });
    }).wait();

    q.memcpy(output, d_output, total_size * sizeof(uint8_t)).wait();

    sycl::free(d_input, q);
    sycl::free(d_output, q);
}

