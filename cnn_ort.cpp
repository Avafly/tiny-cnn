#include <cstdio>
#include <string>
#include <vector>
#include <omp.h>
#include <onnxruntime_cxx_api.h>

#define IMG_COUNT       1000
#define IMG_HEIGHT      16
#define IMG_WIDTH       16
#define IMG_SIZE        (IMG_HEIGHT * IMG_WIDTH)
#define ALIGN_SIZE      64

float Inputs[IMG_COUNT * IMG_SIZE]
    __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
int Preds[IMG_COUNT]
    __attribute__((aligned(ALIGN_SIZE))) = { 0, };

std::vector<std::string> input_names, output_names;
std::vector<const char *> input_names_ptr, output_names_ptr;

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
        std::printf("Usage: %s model input [threads]\n", argv[0]);
        return 0;
    }
    int threads = omp_get_num_procs();
    if (argc >= 4 && std::stoi(argv[3]) > 0 && std::stoi(argv[3]) < threads)
        threads = std::stoi(argv[3]);
    std::printf("Model: %s\n", argv[1]);
    std::printf("Input: %s\n", argv[2]);
    std::printf("Threads: %d\n", threads);

    // load model and input
    if (LoadArray(argv[2], Inputs, IMG_COUNT * IMG_SIZE) == 0)
    {
        std::printf("Failed to load data\n");
        return 1;
    }

    // init onnxruntime
    auto env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "TINY-CNN");
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info{nullptr};
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    try
    {
        session = Ort::Session(env, argv[1], session_options);
    }
    catch (const Ort::Exception &e)
    {
        std::printf("Failed to load model: %s\n", e.what());
        return 1;
    }
    memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
    );
    Ort::AllocatorWithDefaultOptions allocator;
    auto in_count = session.GetInputCount(), out_count = session.GetOutputCount();
    for (size_t i = 0; i < in_count; ++i)
        input_names.emplace_back(std::string(session.GetInputNameAllocated(i, allocator).get()));
    for (size_t i = 0; i < out_count; ++i)
        output_names.emplace_back(std::string(session.GetOutputNameAllocated(i, allocator).get()));
    for (const auto &name : input_names)
        input_names_ptr.emplace_back(name.c_str());
    for (const auto &name : output_names)
        output_names_ptr.emplace_back(name.c_str());

    // reco images
    double start_time = omp_get_wtime();
    // norm
    for (int i = 0; i < IMG_COUNT * IMG_HEIGHT * IMG_WIDTH; ++i)
        Inputs[i] = Inputs[i] / 255.0f;
    std::vector<int64_t> in_shape = {IMG_COUNT, IMG_HEIGHT, IMG_WIDTH, 1};
    Ort::Value input_tensors = Ort::Value::CreateTensor<float>(
        memory_info, Inputs, IMG_COUNT * IMG_HEIGHT * IMG_WIDTH, in_shape.data(), in_shape.size());
    std::vector<Ort::Value> output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names_ptr.data(),
        &input_tensors,
        input_names_ptr.size(),
        output_names_ptr.data(),
        output_names_ptr.size()
    );
    for (int i = 0; i < IMG_COUNT; ++i)
    {
        const float *ptr = &output_tensors.front().GetTensorData<float>()[i * 10];
        int pred = 0;
        float max_value = ptr[0];
        for (int i = 1; i < 10; ++i)
        {
            if (ptr[i] > max_value)
            {
                max_value = ptr[i];
                pred = i;
            }
        }
        Preds[i] = pred;
    }
    std::printf("Elapsed time: %.2f ms\n", (omp_get_wtime() - start_time) * 1000.0);

#ifdef SHOW_RESULTS
    // show predictions
    for (int i = 0; i < IMG_COUNT; ++i)
    {
        std::printf("%d ", Preds[i]);
        if ((i + 1) % (IMG_COUNT / 10) == 0)
            std::printf("\n");
    }
#endif

    return 0;
}
