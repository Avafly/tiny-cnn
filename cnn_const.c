#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <cblas.h>
#include <omp.h>

#define IMG_COUNT       1000
#define IMG_HEIGHT      16
#define IMG_WIDTH       16
#define IMG_SIZE        (IMG_HEIGHT * IMG_WIDTH)
#define ALIGN_SIZE      64
#define MODEL_SIZE      11230
#define MAX_THREADS     4
#define BLOB_SIZE       1580
#define IM2COL_BUF_SIZE 3744

float ModelParam[MODEL_SIZE]
    __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float Blobs[MAX_THREADS * BLOB_SIZE]
    __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float Inputs[IMG_COUNT * IMG_SIZE]
    __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float Images[MAX_THREADS * (IMG_SIZE + 1)]
    __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float Im2Col_Buf[MAX_THREADS * IM2COL_BUF_SIZE]
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

int Im2Col(
    const float *data_im, float *data_col,
    const int in_c, const int in_h, const int in_w,
    const int out_c, const int out_h, const int out_w,
    const int kernel_size, const int padding, const int stride
)
{
    int col_i = 0;
    const int kk = kernel_size * kernel_size;
    const int data_col_size = kk * in_c * out_h * out_w;
    for (int ch = 0; ch < in_c; ++ch)
    {
        for (int kh = 0; kh < kernel_size; ++kh)
        {
            for (int kw = 0; kw < kernel_size; ++kw)
            {
                int row_i = ch * kk + kh * kernel_size + kw;
                for (int oh = 0; oh < out_h; ++oh)
                {
                    int ih = oh * stride + kh - padding;
                    for (int ow = 0; ow < out_w; ++ow)
                    {
                        int iw = ow * stride + kw - padding;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                            data_col[row_i * (out_h * out_w) + col_i++] = data_im[ch * (in_h * in_w) + ih * in_w + iw];
                        else
                            data_col[row_i * (out_h * out_w) + col_i++] = 0.0f;
                    }
                }
                col_i = 0;
            }
        }
    }
    data_col += data_col_size;
    for (int col = out_h * out_w; col--; *data_col++ = 1.0f);
    return data_col_size + out_h * out_w;
}

int ConvLayer(
    const float *bottom, float *top,
    const int in_c, const int in_h, const int in_w,
    const int out_c, const int out_h, const int out_w,
    const float *weights, const int kernel_size, const int padding
)
{
    const int t_id = omp_get_thread_num();
    float *data_col = &Im2Col_Buf[t_id * IM2COL_BUF_SIZE];
    Im2Col(
        bottom, data_col, in_c, in_h, in_w, out_c, out_h, out_w, kernel_size, padding, 1
    );
    const int m = out_c;
    const int n = out_h * out_w;
    const int k = kernel_size * kernel_size * in_c + 1;
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k, 1.0f, weights, k, data_col, n, 0.0f, top, n
    );
    return m * n;
}

int MaxPoolingLayer(
    const float *bottom, float *top,
    const int in_c, const int in_h, const int in_w,
    const int out_c, const int out_h, const int out_w,
    const int kernel_size, const int stride
)
{
    int top_size = 0;
    for (int ch = 0; ch < in_c; ++ch)
    {
        for (int oh = 0; oh < out_h; ++oh)
        {
            for (int ow = 0; ow < out_w; ++ow)
            {
                int in_pos = ch * in_h * in_w + oh * kernel_size * in_w + ow * kernel_size;
                float max_value = bottom[in_pos];
                for (int m = 0; m < kernel_size; ++m)
                {
                    for (int n = 0; n < kernel_size; ++n)
                    {
                        int index = in_pos + m * in_w + n;
                        // same mode
                        if ((ow >= in_w / stride && n > 0) ||
                            (oh >= in_h / stride && m > 0))
                        {
                            continue;
                        }
                        max_value = bottom[index] > max_value ? bottom[index] : max_value;
                    }
                }
                top[top_size++] = max_value;
            }
        }
    }
    return top_size;
}

inline void ReLU(float *data, const int size, const float alpha)
{
    for (int i = 0; i < size; ++i)
        data[i] = data[i] > 0.0f ? data[i] : data[i] * alpha;
}

inline int FCLayer(
    const float *Fc1, const float *bottom, float *top,
    int out_feat, int in_feat
)
{
    cblas_sgemv(
        CblasRowMajor, CblasNoTrans,
        out_feat, in_feat, 1.0f, Fc1, in_feat,
        bottom, 1, 0.0f, top, 1
    );
    return out_feat;
}

void Reco(float *image, const int image_i, float *blob)
{
    int top_size = 0;
    float *bottom = image;
    float *top = blob;
    int kernel_size;
    int padding;
    const float alpha = 0.1f;
    int in_c, in_h, in_w, out_c, out_h, out_w;

    memset(blob, 0, BLOB_SIZE * sizeof(float));

    // parse model
    const float *Conv1 = ModelParam;
    const float *Conv2 = ModelParam + 156;
    const float *Fc1 = ModelParam + 596;
    const float *Fc2 = ModelParam + 9940;

    // Conv
    kernel_size = 5, padding = 0;
    in_c = 1, in_h = IMG_HEIGHT, in_w = IMG_WIDTH;
    out_c = 6;
    out_h = in_h - kernel_size + 1;
    out_w = in_w - kernel_size + 1;
    top_size = ConvLayer(
        bottom, top, in_c, in_h, in_w, out_c, out_h, out_w,
        Conv1, kernel_size, padding
    );

    // ReLU
    ReLU(top, top_size, alpha);
    
    // Max Pooling
    bottom = top;
    top = &top[top_size];
    kernel_size = 2;
    in_c = out_c, in_h = out_h, in_w = out_w;
    out_h = out_h / kernel_size;
    out_w = out_w / kernel_size;
    top_size = MaxPoolingLayer(
        bottom, top, in_c, in_h, in_w, out_c, out_h, out_w,
        kernel_size, kernel_size
    );

    // Conv
    bottom = top;
    top = &top[top_size];
    kernel_size = 3, padding = 1;
    in_c = out_c, in_h = out_h, in_w = out_w;
    out_c = 8;
    out_h = in_h - kernel_size + 2 * padding + 1;
    out_w = in_w - kernel_size + 2 * padding + 1;
    top_size = ConvLayer(
        bottom, top, in_c, in_h, in_w, out_c, out_h, out_w,
        Conv2, kernel_size, padding
    );

    // ReLU
    ReLU(top, top_size, alpha);

    // Max Pooling
    bottom = top;
    top = &top[top_size];
    kernel_size = 2;
    in_c = out_c, in_h = out_h, in_w = out_w;
    out_h = out_h / kernel_size;
    out_w = out_w / kernel_size;
    top_size = MaxPoolingLayer(
        bottom, top, in_c, in_h, in_w, out_c, out_h, out_w,
        kernel_size, kernel_size
    );

    // FC
    bottom = top;
    bottom[top_size] = 1.0f;
    top = &top[top_size + 1];
    in_c = 0, in_h = 0, in_w = top_size + 1;
    out_c = 0, out_h = 0, out_w = 128;
    top_size = FCLayer(Fc1, bottom, top, out_w, in_w);

    // ReLU
    ReLU(top, top_size, alpha);

    // FC
    bottom = top;
    bottom[top_size] = 1.0f;
    top = &top[top_size + 1];
    in_w = out_w + 1;
    out_w = 10;
    top_size = FCLayer(Fc2, bottom, top, out_w, in_w);
    
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
    int threads = omp_get_num_procs();
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
