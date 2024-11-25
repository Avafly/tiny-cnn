#ifndef LAYERS_H_
#define LAYERS_H_

typedef enum {
    LAYER_CONV,
    LAYER_MAXPOOL,
    LAYER_RELU,
    LAYER_FC
} LayerType;

typedef struct {
    LayerType type;
    float *weights;
    // conv
    int kernel_size;
    int filters;
    int padding;
    // relu
    float alpha;
    // fc
    int in_feat;
    int out_feat;
} Layer;

int ConvLayer(
    const float *bottom, float *top,
    const int in_c, const int in_h, const int in_w,
    const int out_c, const int out_h, const int out_w,
    const float *weights, const int kernel_size, const int padding
);

int MaxPoolingLayer(
    const float *bottom, float *top,
    const int in_c, const int in_h, const int in_w,
    const int out_c, const int out_h, const int out_w,
    const int kernel_size, const int stride
);

void ReLU(float *data, const int size, const float alpha);

int FCLayer(
    const float *Fc1, const float *bottom, float *top,
    int out_feat, int in_feat
);

#endif  // LAYERS_H_