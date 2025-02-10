#include "layers.h"
#include "config.h"
#include <cblas.h>
#include <omp.h>

static float Im2Col_Buf[MAX_THREADS * IM2COL_BUF_SIZE]
    __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };

static int Im2Col(
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

void ReLU(float *data, const int size, const float alpha)
{
    for (int i = 0; i < size; ++i)
        data[i] = data[i] > 0.0f ? data[i] : data[i] * alpha;
}

int FCLayer(
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