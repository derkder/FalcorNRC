#pragma once
#include <string>
#include <cuda_runtime.h>
#include "Config.h"
#include "Params.slang"

class RadiosityNetwork
{
public:
    RadiosityNetwork(const uint32_t width, const uint32_t height);
    ~RadiosityNetwork();

    void forward(Falcor::RadiosityQuery* queries, cudaSurfaceObject_t output);
    void train(Falcor::RadiosityQuery* queries, cudaSurfaceObject_t output, float& loss);

private:
    uint32_t frame_width;
    uint32_t frame_height;
    uint32_t seed = 7272u;
    float learning_rate = 1e-4f;
};
