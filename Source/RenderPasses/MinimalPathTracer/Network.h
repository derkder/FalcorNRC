#pragma once
#include <string>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include "Config.h"
#include "Params.slang"

class NRCNetwork
{
public:

    NRCNetwork(const uint32_t width, const uint32_t height);
    ~NRCNetwork();

    void Test();
    //void forward(Falcor::RadianceQuery* queries, cudaSurfaceObject_t output);
    void forward(Falcor::RadianceQuery* queries, Falcor::RadianceTarget* thps, Falcor::RadianceTarget* ptResults, cudaSurfaceObject_t output);
    void train(
        Falcor::RadianceQuery* queries,
        Falcor::RadianceTarget* targets,
        float& loss, uint32_t* trainCounts
    );

private:
    uint32_t frame_width;
    uint32_t frame_height;
    //uint32_t seed = 7272u;
    float learning_rate = 1e-4f;
    uint32_t batch_size = 1 << 14;
    uint32_t n_train_batch = 4;
    unsigned int max_training_query_size = 1 << 16;                   // ~57,600
    std::vector<float> random_seq_host;
    unsigned int seed = 3407u;//重要指示，改成3407
    //unsigned int seed = 43256;
    bool isRandom = false;
};
