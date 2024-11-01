#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <sched.h>
#include <unistd.h>
#include <png.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

using namespace std;

struct RowData {
    int num_rows;
    int row_indices[1];  // Flexible array member
    // row_data will follow immediately after row_indices
};

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
    png_set_compression_level(png_ptr, 0);
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

void mandelbrot_avx512(const double x0s[], const double y0s[], int maxIter, int out[], int n) {
    __mmask8 valid_mask = (1 << n) - 1;

    __m512d vx0 = _mm512_maskz_loadu_pd(valid_mask, x0s);
    __m512d vy0 = _mm512_maskz_loadu_pd(valid_mask, y0s);

    __m512d vx = _mm512_setzero_pd();
    __m512d vy = _mm512_setzero_pd();
    __m512i viter = _mm512_setzero_si512();
    __m512i vone = _mm512_set1_epi32(1);
    __m512d vtwo = _mm512_set1_pd(2.0);
    __m512d vfour = _mm512_set1_pd(4.0);

    __mmask8 active_mask = valid_mask;

    for (int iter = 0; iter < maxIter; ++iter) {
        __m512d vx2 = _mm512_mul_pd(vx, vx);
        __m512d vy2 = _mm512_mul_pd(vy, vy);
        __m512d length_squared = _mm512_add_pd(vx2, vy2);
        __mmask8 lt_mask = _mm512_cmp_pd_mask(length_squared, vfour, _CMP_LT_OS);
        active_mask = _kand_mask8(active_mask, lt_mask);

        if (active_mask == 0) break;

        viter = _mm512_mask_add_epi32(viter, active_mask, viter, vone);
        __m512d vxy = _mm512_mul_pd(vx, vy);
        vy = _mm512_fmadd_pd(vtwo, vxy, vy0);
        vx = _mm512_add_pd(_mm512_sub_pd(vx2, vy2), vx0);
    }

    _mm512_mask_storeu_epi32(out, valid_mask, viter);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char* output = argv[1];
    int maxIter = stoi(argv[2]);
    double minR = stod(argv[3]);
    double maxR = stod(argv[4]);
    double minI = stod(argv[5]);
    double maxI = stod(argv[6]);
    int width = stoi(argv[7]);
    int height = stoi(argv[8]);

    // Calculate number of rows for this process
    int num_my_rows = (height + size - 1) / size;
    int start_row = rank;

    // Allocate memory for rows and row_data
    size_t row_data_size = num_my_rows * width * sizeof(int);
    int* row_data = (int*)malloc(row_data_size);

    // Process assigned rows
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_my_rows; i++) {
        int row = start_row + i * size;
        if (row >= height) continue;

        double y0 = minI + row * (maxI - minI) / height;
        int* current_row = row_data + i * width;

        for (int x = 0; x < width; x += 8) {
            double x0s[8];
            double y0s[8];
            int outs[8];

            int n = (x + 8 <= width) ? 8 : (width - x);
            for (int j = 0; j < n; j++) {
                x0s[j] = minR + (x + j) * (maxR - minR) / width;
                y0s[j] = y0;
            }

            mandelbrot_avx512(x0s, y0s, maxIter, outs, n);
            for (int j = 0; j < n; j++) {
                current_row[x + j] = outs[j];
            }
        }
    }

    if (rank == 0) {
        // Master process: Allocate and initialize final image buffer
        int* image = (int*)malloc(width * height * sizeof(int));

        // Copy own results to image buffer
        for (int i = 0; i < num_my_rows; i++) {
            int row = start_row + i * size;
            if (row < height) {
                memcpy(image + row * width, row_data + i * width, width * sizeof(int));
            }
        }

        // Receive results from other processes
        for (int src = 1; src < size; src++) {
            MPI_Status status;
            int recv_num_rows = (height + size - 1) / size;
            int* recv_row_data = (int*)malloc(recv_num_rows * width * sizeof(int));

            MPI_Recv(recv_row_data, recv_num_rows * width, MPI_INT, src, 0, MPI_COMM_WORLD, &status);

            for (int i = 0; i < recv_num_rows; i++) {
                int row = src + i * size;
                if (row < height) {
                    memcpy(image + row * width, recv_row_data + i * width, width * sizeof(int));
                }
            }

            free(recv_row_data);
        }

        // Write final image
        write_png(output, maxIter, width, height, image);
        free(image);
    } else {
        // Worker processes: Send results to master
        MPI_Send(row_data, num_my_rows * width, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // Cleanup
    free(row_data);
    // MPI_Finalize();
    return 0;
}
