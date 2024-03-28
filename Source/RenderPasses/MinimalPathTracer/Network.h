#pragma once
#include <string>
#include <cuda_runtime.h>
#include "Config.h"
#include "Params.slang"

class NRCNetwork
{
public:

    NRCNetwork(const uint32_t width, const uint32_t height);
    ~NRCNetwork();

    void Test();
    void forward(Falcor::RadianceQuery* queries, cudaSurfaceObject_t output);
    void train(Falcor::RadianceQuery* queries, cudaSurfaceObject_t output, float& loss);

private:
    uint32_t frame_width;
    uint32_t frame_height;
    uint32_t seed = 7272u;
    float learning_rate = 1e-4f;
};
