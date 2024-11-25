#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <omp.h>
#include "config.h"
#include "layers.h"

float ModelParam[MODEL_SIZE]
    __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float Blobs[MAX_THREADS * BLOB_SIZE]
    __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float Inputs[IMG_COUNT * IMG_SIZE]
    __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float Images[MAX_THREADS * (IMG_SIZE + 1)]
    __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
int Preds[IMG_COUNT]
    __attribute__((aligned(ALIGN_SIZE))) = { 0, };
Layer layers[NUM_LAYER]
    __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };

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

void BuildModel()
{
    // This is a simple demo where layers are created directly in the code
    // The model graph could be defined by a config file in a real use case
    layers[0].type = LAYER_CONV;
    layers[0].weights = ModelParam;
    layers[0].filters = 6, layers[0].kernel_size = 5, layers[0].padding = 0;

    layers[1].type = LAYER_RELU;
    layers[1].weights = NULL;
    layers[1].alpha = 0.1f;

    layers[2].type = LAYER_MAXPOOL;
    layers[2].weights = NULL;
    layers[2].kernel_size = 2;
    
    layers[3].type = LAYER_CONV;
    layers[3].weights = ModelParam + 156;
    layers[3].filters = 8, layers[3].kernel_size = 3, layers[3].padding = 1;

    layers[4].type = LAYER_RELU;
    layers[4].weights = NULL;
    layers[4].alpha = 0.1f;

    layers[5].type = LAYER_MAXPOOL;
    layers[5].weights = NULL;
    layers[5].kernel_size = 2;

    layers[6].type = LAYER_FC;
    layers[6].weights = ModelParam + 596;
    layers[6].in_feat = 72, layers[6].out_feat = 128;

    layers[7].type = LAYER_RELU;
    layers[7].weights = NULL;
    layers[7].alpha = 0.1f;

    layers[8].type = LAYER_FC;
    layers[8].weights = ModelParam + 9940;
    layers[8].in_feat = 128, layers[8].out_feat = 10;
}

void Reco(float *image, const int image_i, float *blob)
{
    int top_size = 0;
    float *bottom = image;
    float *top = blob;
    int kernel_size;
    int padding;
    int in_c, in_h, in_w, out_c, out_h, out_w;
    const Layer *layers_ptr = layers;

    memset(blob, 0, BLOB_SIZE * sizeof(float));

    // input layer
    kernel_size = layers_ptr->kernel_size;
    padding = layers_ptr->padding;
    in_c = 1, in_h = IMG_HEIGHT, in_w = IMG_WIDTH;
    out_c = layers_ptr->filters;
    out_h = in_h - kernel_size + 2 * padding + 1;
    out_w = in_w - kernel_size + 2 * padding + 1;
    top_size = ConvLayer(
        bottom, top, in_c, in_h, in_w, out_c, out_h, out_w,
        layers_ptr->weights, kernel_size, padding
    );

    for (int layer_i = 1; layer_i < NUM_LAYER; ++layer_i)
    {
        // get new layer config
        ++layers_ptr;
        // forward propagation
        if (layers_ptr->type == LAYER_CONV)
        {
            bottom = top;
            top = &top[top_size];
            kernel_size = layers_ptr->kernel_size, padding = layers_ptr->padding;
            in_c = out_c, in_h = out_h, in_w = out_w;
            out_c = layers_ptr->filters;
            out_h = in_h - kernel_size + 2 * padding + 1;
            out_w = in_w - kernel_size + 2 * padding + 1;
            top_size = ConvLayer(
                bottom, top, in_c, in_h, in_w, out_c, out_h, out_w,
                layers_ptr->weights, kernel_size, padding
            );
        }
        else if (layers_ptr->type == LAYER_RELU)
        {
            ReLU(top, top_size, layers_ptr->alpha);
        }
        else if (layers_ptr->type == LAYER_MAXPOOL)
        {
            bottom = top;
            top = &top[top_size];
            kernel_size = layers_ptr->kernel_size;
            in_c = out_c, in_h = out_h, in_w = out_w;
            out_h = out_h / kernel_size;
            out_w = out_w / kernel_size;
            top_size = MaxPoolingLayer(
                bottom, top, in_c, in_h, in_w, out_c, out_h, out_w,
                kernel_size, kernel_size
            );
        }
        else if (layers_ptr->type == LAYER_FC)
        {
            bottom = top;
            bottom[top_size] = 1.0f;
            top = &top[top_size + 1];
            in_w = top_size + 1;
            out_w = layers_ptr->out_feat;
            top_size = FCLayer(layers_ptr->weights, bottom, top, out_w, in_w);
        }
        else
        {
            printf("Error: unknown layer\n");
            break;
        }
    }
    
    // argmax
    int pred = 0;
    float max_value = top[0];
    for (int i = 1; i < top_size; ++i)
    {
        if (top[i] > max_value)
        {
            max_value = top[i];
            pred = i;
        }
    }

    Preds[image_i] = pred;
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
    if (argc >= 4 && atoi(argv[3]) > 0 && atoi(argv[3]) < MAX_THREADS)
        threads = atoi(argv[3]);
    printf("Model: %s\n", argv[1]);
    printf("Input: %s\n", argv[2]);
    printf("Threads: %d\n", threads);

    // load model and input
    if (LoadArray(argv[1], ModelParam, MODEL_SIZE) == 0 ||
        LoadArray(argv[2], Inputs, IMG_COUNT * IMG_SIZE) == 0)
    {
        printf("Failed to load data\n");
        return 1;
    }

    BuildModel();

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
        image_ptr[IMG_SIZE] = 1.0f;

        Reco(image_ptr, i, &Blobs[t_id * BLOB_SIZE]);
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
