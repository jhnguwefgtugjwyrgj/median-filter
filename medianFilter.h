#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include "utils.h"




class MedianFilter {
private:
    static float median_7(float arr[7]);
    static uint8_t median_9(uint8_t window[9]);
public:
    static void median_filter_7(const float* input, float* output, size_t length);
    static void median_filter_3x3(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride);
    // RGB (interleaved) ??????: stride_bytes >= width*3, ??????? ??????? [R,G,B].
    static void median_filter_3x3_rgb(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride_bytes);
};

float MedianFilter::median_7(float arr[7]) {
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

void MedianFilter::median_filter_7(const float* input, float* output, size_t length) {
    float window[7];

    for (size_t i = 0; i < 3; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = i + j;
            if (idx < 0) window[j + 3] = input[0];
            else if (idx >= length) window[j + 3] = input[length - 1];
            else window[j + 3] = input[idx];
        }

        output[i] = median_7(window);
    }

    for (size_t i = 3; i < length - 3; ++i) {
        for (int j = -3; j <= 3; ++j) window[j + 3] = input[i + j];

        output[i] = median_7(window);
    }

    for (size_t i = length - 3; i < length; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = i + j;
            if (idx < 0) window[j + 3] = input[0];
            else if (idx >= length) window[j + 3] = input[length - 1];
            else window[j + 3] = input[idx];
        }

        output[i] = median_7(window);
    }
}




uint8_t MedianFilter::median_9(uint8_t window[9]) {
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


void MedianFilter::median_filter_3x3(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride) {
    for (size_t y = 0; y < height; ++y) {
        const uint8_t* y0 = input + (y > 0 ? y - 1 : 0) * stride;
        const uint8_t* y1 = input + y * stride;                   
        const uint8_t* y2 = input + (y < height - 1 ? y + 1 : y) * stride;

        for (size_t x = 0; x < width; ++x) {
            uint8_t window[9];

            size_t x0 = (x > 0 ? x - 1 : 0); 
            size_t x1 = x;                        
            size_t x2 = (x < width - 1 ? x + 1 : x); 

            window[0] = y0[x0]; window[1] = y0[x1]; window[2] = y0[x2];
            window[3] = y1[x0]; window[4] = y1[x1]; window[5] = y1[x2];
            window[6] = y2[x0]; window[7] = y2[x1]; window[8] = y2[x2];

            uint8_t median = median_9(window);

            output[y * stride + x] = median;
        }
    }
}

void MedianFilter::median_filter_3x3_rgb(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride_bytes) {
    const size_t row_stride = stride_bytes;
    constexpr size_t channels = 3;

    for (size_t y = 0; y < height; ++y) {
        const size_t y0i = (y > 0 ? y - 1 : 0);
        const size_t y1i = y;
        const size_t y2i = (y + 1 < height ? y + 1 : y);

        const uint8_t* y0 = input + y0i * row_stride;
        const uint8_t* y1 = input + y1i * row_stride;
        const uint8_t* y2 = input + y2i * row_stride;

        uint8_t* out_row = output + y * row_stride;

        for (size_t x = 0; x < width; ++x) {
            const size_t x0 = (x > 0 ? x - 1 : 0);
            const size_t x1 = x;
            const size_t x2 = (x + 1 < width ? x + 1 : x);

            const size_t p0 = x0 * channels;
            const size_t p1 = x1 * channels;
            const size_t p2 = x2 * channels;

            for (size_t c = 0; c < channels; ++c) {
                uint8_t window[9];
                window[0] = y0[p0 + c]; window[1] = y0[p1 + c]; window[2] = y0[p2 + c];
                window[3] = y1[p0 + c]; window[4] = y1[p1 + c]; window[5] = y1[p2 + c];
                window[6] = y2[p0 + c]; window[7] = y2[p1 + c]; window[8] = y2[p2 + c];

                out_row[x * channels + c] = median_9(window);
            }
        }
    }
}