#include <stdio.h>
#include <string>
#include <omp.h>
#include "net.h"

#define IMG_COUNT       1000
#define IMG_HEIGHT      16
#define IMG_WIDTH       16
#define IMG_SIZE        (IMG_HEIGHT * IMG_WIDTH)
#define ALIGN_SIZE      64
#define MAX_THREADS     4

float Inputs[IMG_COUNT * IMG_SIZE]
    __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float Images[MAX_THREADS * IMG_SIZE]
    __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
int Preds[IMG_COUNT]
    __attribute__((aligned(ALIGN_SIZE))) = { 0, };

int LoadArray(const char *filename, float *buffer, const size_t size)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
        return 0;
    for (size_t i = 0; i < size; ++i)
    {
        if (fscanf(file, "%f", &buffer[i]) != 1)
        {
            fclose(file);
            return 0;
        }
    }
    fclose(file);
    return 1;
}

int main(int argc, char *argv[])
{
    // get settings
    if (argc < 3)
    {
        printf("Usage: %s model input [threads]\n", argv[0]);
        return 0;
    }
    int threads = omp_get_num_procs() > MAX_THREADS ? MAX_THREADS : omp_get_num_procs();
    if (argc >= 4 && std::stoi(argv[3]) > 0 && std::stoi(argv[3]) < MAX_THREADS)
        threads = std::stoi(argv[3]);
    printf("Model: %s\n", argv[1]);
    printf("Input: %s\n", argv[2]);
    printf("Threads: %d\n", threads);

    // load model and input
    if (LoadArray(argv[2], Inputs, IMG_COUNT * IMG_SIZE) == 0)
    {
        printf("Failed to load data\n");
        return 1;
    }

    // init ncnn
    ncnn::Net net;
    net.opt.num_threads = threads;
    if (net.load_param((std::string(argv[1]) + ".param").c_str()) ||
        net.load_model((std::string(argv[1]) + ".bin").c_str()))
    {
        printf("Failed to load model\n");
        return 1;
    }

    // reco images
    double start_time = omp_get_wtime();
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < IMG_COUNT; ++i)
    {
        int t_id = omp_get_thread_num();
        float *image_ptr = &Images[t_id * (IMG_SIZE + 1)];
        // norm
        for (int j = 0; j < IMG_SIZE; ++j)
            image_ptr[j] = Inputs[i * IMG_SIZE + j] / 255.0f;

        ncnn::Mat in(IMG_WIDTH, IMG_HEIGHT, image_ptr);
        ncnn::Mat out;
        // inference
        ncnn::Extractor ex = net.create_extractor();
        ex.input("conv2d_2_input", in);
        ex.extract("dense_3", out);
        // argmax
        int pred = 0;
        float max_value = out[0];
        for (int j = 1; j < out.w; ++j)
        {
            if (out[j] > max_value)
            {
                max_value = out[j];
                pred = j;
            }
        }
        Preds[i] = pred;
    }
    printf("Elapsed time: %.2f ms\n", (omp_get_wtime() - start_time) * 1000.0);

#ifdef SHOW_RESULTS
    // show predictions
    for (int i = 0; i < IMG_COUNT; ++i)
    {
        printf("%d ", Preds[i]);
        if ((i + 1) % (IMG_COUNT / 10) == 0)
            printf("\n");
    }
#endif

    return 0;
}
