#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <iostream>
#include <vector>
#include <atomic>
#include <sched.h>
#include <unistd.h>
#include <png.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

using namespace std;

// Declare curTask as a global atomic variable
std::atomic<int> curTask(0);
// mutex
struct ThreadData {
    int width, height;
    double minR, maxR, minI, maxI;
    int maxIter;
    int* image;
};

void mandelbrot_simd(const double* x0s, double y0, int maxIter, int64_t* results, __mmask8 valid_mask) {
    __m512d x = _mm512_set1_pd(0.0);
    __m512d y = _mm512_set1_pd(0.0);
    __m512d x0_vec = _mm512_maskz_loadu_pd(valid_mask, x0s);
    __m512d y0_vec = _mm512_set1_pd(y0);
    __m512d two = _mm512_set1_pd(2.0);
    __m512d four = _mm512_set1_pd(4.0);
    __m512i repeats = _mm512_setzero_si512(); // 追蹤每個點的迭代次數，初始為0
    __m512i one = _mm512_set1_epi64(1); // 用來算 repeats

    __mmask8 active_mask = valid_mask;

    for (int i = 0; i < maxIter; ++i) {
        // Compute x^2 and y^2
        __m512d x2 = _mm512_mul_pd(x, x);
        __m512d y2 = _mm512_mul_pd(y, y);

        // Compute length_squared = x2 + y2
        __m512d length_squared = _mm512_add_pd(x2, y2);

        // Check which elements are still active
        __mmask8 lt_mask = _mm512_cmp_pd_mask(length_squared, four, _CMP_LT_OS); // 1 : alive; 0 : byebye
        active_mask = _mm512_kand(active_mask, lt_mask);

        if (active_mask == 0) break;

        // Update repeats count for active elements
        repeats = _mm512_mask_add_epi64(repeats, active_mask, repeats, one);

        // Compute x * y
        __m512d xy = _mm512_mul_pd(x, y);

        // Update y = 2.0 * x * y + y0
        y = _mm512_mask_fmadd_pd(two, active_mask, xy, y0_vec);

        // Update x = x2 - y2 + x0
        __m512d x_new = _mm512_add_pd(_mm512_sub_pd(x2, y2), x0_vec);
        x = _mm512_mask_mov_pd(x, active_mask, x_new);
    }

    // Store results for valid elements
    _mm512_mask_storeu_epi64(results, valid_mask, repeats);
}

void* compute(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int y;

    const int step = 8; // SIMD register的寬度，每次可以處理8個 double
    alignas(64) double x0s[step];
    alignas(64) int64_t results[step];

    while ((y = curTask.fetch_add(1)) < data->height) {
        double y0 = y * ((data->maxI - data->minI) / data->height) + data->minI;
        for (int x = 0; x < data->width; x += step) {
            int valid_steps = std::min(step, data->width - x);
            // Initialize x0s for valid steps
            for (int i = 0; i < valid_steps; ++i) {
                x0s[i] = (x + i) * ((data->maxR - data->minR) / data->width) + data->minR;
            }
            // Zero out remaining elements to avoid using uninitialized data
            for (int i = valid_steps; i < step; ++i) {
                x0s[i] = 0.0;
            }
            // Create valid mask
            __mmask8 valid_mask = (1 << valid_steps) - 1;

            // Run SIMD computation with valid_mask
            mandelbrot_simd(x0s, y0, data->maxIter, results, valid_mask);

            // Store only the valid results
            for (int i = 0; i < valid_steps; ++i) {
                data->image[y * data->width + (x + i)] = (int)results[i];
            }
        }
    }
    return nullptr;
}


// Write the result as a PNG file
void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(
        png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT
    );
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0); // 1 -> 0
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = (p % 16) * 16;
                } else {
                    color[0] = (p % 16) * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    // if (argc != 9) {
    //     cout << "Usage: " << argv[0] << " output.png max_iter minR maxR minI maxI width height" << endl;
    //     return 1;
    // }

    const char* output = argv[1];
    int maxIter = stoi(argv[2]);
    double minR = stod(argv[3]);
    double maxR = stod(argv[4]);
    double minI = stod(argv[5]);
    double maxI = stod(argv[6]);
    int width = stoi(argv[7]);
    int height = stoi(argv[8]);

    // // Detect number of CPUs (threads)
    int ncpus;
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    ncpus = CPU_COUNT(&cpu_set);  // Get number of CPUs available

    // vector<pthread_t> threads(ncpus);
    // vector<ThreadData> threadData(ncpus);
    vector<pthread_t> threads(ncpus);
    vector<ThreadData> threadData(ncpus);

    // Allocate memory for image
    // int* image = (int*)malloc(width * height * sizeof(int));
    // assert(image);
    int* image = nullptr;
    image = new int[width * height];
    
    // Initialize the thread data and create threads
    for (int i = 0; i < ncpus; ++i) {
        threadData[i] = {width, height, minR, maxR, minI, maxI, maxIter, image};
        pthread_create(&threads[i], nullptr, compute, &threadData[i]);
    }

    // Join threads after they finish computation
    for (int i = 0; i < ncpus; ++i) {
        pthread_join(threads[i], nullptr);
    }

    // Write the output image using the custom write_png function
    write_png(output, maxIter, width, height, image);

    // Free allocated memory
    // free(image);
    delete[] image;
    return 0;
}
