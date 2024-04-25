#include "Network.h"
#include "tiny-cuda-nn/common_host.h"

#include <cuda.h>
#include <curand.h>

#include <cstdint>
#include <memory>
#include <fstream>
#include <iostream>
#include <filesystem/path.h>

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/common.h>
#include <json/json.hpp>


using namespace tcnn;
using precision_t = network_precision_t;

namespace
{

struct NetworkComponents
{
    std::shared_ptr<Loss<precision_t>> loss = nullptr;
    std::shared_ptr<Optimizer<precision_t>> optimizer = nullptr;
    std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = nullptr;
    std::shared_ptr<Trainer<float, precision_t, precision_t>> trainer = nullptr;
};

struct IOData
{
    GPUMatrix<float>* render_input_mat = nullptr;
    GPUMatrix<float>* render_output_mat = nullptr;

    GPUMatrix<float>* training_input_mat = nullptr;
    GPUMatrix<float>* training_output_mat = nullptr;

    GPUMemory<float>* random_seq = nullptr;
};

cudaStream_t inference_stream = nullptr;
cudaStream_t training_stream = nullptr;
cudaStream_t temp_stream = nullptr;
curandGenerator_t rng;


NetworkComponents* mNetworkComponents = nullptr;

IOData* mIOData = nullptr;
json loaded_weights;

} // namespace


uint32_t showMsg_counter(uint32_t* dataOnDevice)
{
    uint32_t* dataOnHost = new uint32_t[1];
    cudaMemcpy(dataOnHost, dataOnDevice, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("wow %u\n", dataOnHost[0]);
    uint32_t res = dataOnHost[0];
    delete[] dataOnHost;
    return res;
}


template<typename T, uint32_t input_stride, uint32_t output_stride>
__global__ void formatInputTarget(uint32_t n_elements, Falcor::RadianceQuery* queries, Falcor::RadianceTarget* targets,
    T* input, T* output, uint32_t* trainCount)
{
    n_elements = trainCount[0]; // woc居然只有这样可以读到，根本不能在外面读到一点儿
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;
    

    Falcor::RadianceQuery query = queries[i];
    Falcor::RadianceTarget target = targets[i];

    input[i * input_stride + 0] = query.pos.x, input[i * input_stride + 1] = query.pos.y, input[i * input_stride + 2] = query.pos.z;
    input[i * input_stride + 3] = query.dir.x, input[i * input_stride + 4] = query.dir.y;

    input[i * input_stride + 5] = query.roughness;
    input[i * input_stride + 6] = query.normal.x, input[i * input_stride + 7] = query.normal.y;
    input[i * input_stride + 8] = query.diffuse.x, input[i * input_stride + 9] = query.diffuse.y, input[i * input_stride + 10] = query.diffuse.z;
    input[i * input_stride + 11] = query.specular.x, input[i * input_stride + 12] = query.specular.y,
    input[i * input_stride + 13] = query.specular.z;

    output[i * output_stride + 0] = target.radiance.x, output[i * output_stride + 1] = target.radiance.y, output[i * output_stride + 2] = target.radiance.z;
}

template<typename T, uint32_t input_stride, uint32_t output_stride>
__global__ void formatInputTargetRandom(uint32_t n_elements, uint32_t offset,Falcor::RadianceQuery* queries, Falcor::RadianceTarget* targets,
    T* input, T* output, uint32_t* trainCount, float* random_indices)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + offset > n_elements)
        return;
    int data_index = i * input_stride;
    int sample_index = i + offset;
    //sample_index = (1 - random_indices[sample_index]) * trainCount[0]; //在0 - trainCount[0]之中随机取了一个index

    Falcor::RadianceQuery query = queries[sample_index];
    Falcor::RadianceTarget target = targets[sample_index];

    input[i * input_stride + 0] = query.pos.x, input[i * input_stride + 1] = query.pos.y, input[i * input_stride + 2] = query.pos.z;
    input[i * input_stride + 3] = query.dir.x, input[i * input_stride + 4] = query.dir.y;

    input[i * input_stride + 5] = query.roughness;
    input[i * input_stride + 6] = query.normal.x, input[i * input_stride + 7] = query.normal.y;
    input[i * input_stride + 8] = query.diffuse.x, input[i * input_stride + 9] = query.diffuse.y, input[i * input_stride + 10] = query.diffuse.z;
    input[i * input_stride + 11] = query.specular.x, input[i * input_stride + 12] = query.specular.y,
    input[i * input_stride + 13] = query.specular.z;

    output[i * output_stride + 0] = target.radiance.x, output[i * output_stride + 1] = target.radiance.y, output[i * output_stride + 2] = target.radiance.z;
}

template<typename T, uint32_t input_stride>
__global__ void formatRenderInput(uint32_t n_elements, Falcor::RadianceQuery* queries, T* input)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;

    Falcor::RadianceQuery query = queries[i];

    input[i * input_stride + 0] = query.pos.x, input[i * input_stride + 1] = query.pos.y, input[i * input_stride + 2] = query.pos.z;
    input[i * input_stride + 3] = query.dir.x, input[i * input_stride + 4] = query.dir.y;

    input[i * input_stride + 5] = query.roughness;
    input[i * input_stride + 6] = query.normal.x, input[i * input_stride + 7] = query.normal.y;
    input[i * input_stride + 8] = query.diffuse.x, input[i * input_stride + 9] = query.diffuse.y, input[i * input_stride + 10] = query.diffuse.z;
    input[i * input_stride + 11] = query.specular.x, input[i * input_stride + 12] = query.specular.y, input[i * input_stride + 13] = query.specular.z;
}

template<typename T, uint32_t stride>
__global__ void mapToOutSurf(uint32_t n_elements, uint32_t width, Falcor::RadianceTarget* targets, T* output, cudaSurfaceObject_t outSurf)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;

    Falcor::RadianceTarget target = targets[i];

    uint32_t x = i % width;
    uint32_t y = i / width;

    float4 color = {0, 0, 0, 1};

    color.x = output[i * stride + 0] * 0.1 + target.radiance.x;
    color.y = output[i * stride + 1] * 0.1 + target.radiance.y;
    color.z = output[i * stride + 2] * 0.1 + target.radiance.z;
    //color.x = target.radiance.x;
    //color.y = target.radiance.y;
    //color.z = target.radiance.z;

    surf2Dwrite(color, outSurf, x * sizeof(float4), y);
}


NRCNetwork :: NRCNetwork(const uint32_t width, const uint32_t height)
{
    std::cout << "Hello World!" << width << height << std::endl;

    CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
    CUDA_CHECK_THROW(cudaStreamCreate(&training_stream));

    //curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    //curandSetPseudoRandomGeneratorSeed(rng, 7272ULL);
    //curandSetStream(rng, training_stream);

    mNetworkComponents = new NetworkComponents();
    mIOData = new IOData();

    filesystem::path c_path(NetConfig::netConfigPath);
    if (!c_path.exists())
    {
        std::cout << "Cannot find the network config!" << std::endl;
        return;
    }
    else
    {
        std::cout << "Successfully find the network config!" << std::endl;
    }

    std::ifstream f(c_path.str());
    json config = json::parse(f, nullptr, true, true);

    json encoding_opts = config.value("encoding", json::object());
    json loss_opts = config.value("loss", json::object());
    json optimizer_opts = config.value("optimizer", json::object());
    json network_opts = config.value("network", json::object());

    mNetworkComponents->loss = std::shared_ptr<Loss<precision_t>>(create_loss<precision_t>(loss_opts));
    mNetworkComponents->optimizer = std::shared_ptr<Optimizer<precision_t>>(create_optimizer<precision_t>(optimizer_opts));
    mNetworkComponents->network = std::make_shared<NetworkWithInputEncoding<precision_t>>(
        NetConfig::n_input_dims, NetConfig::n_output_dims, encoding_opts, network_opts
    );
    mNetworkComponents->trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(
        mNetworkComponents->network, mNetworkComponents->optimizer, mNetworkComponents->loss
    );

    //filesystem::path w_path(NetConfig::weightsPath);
    //if (!w_path.exists())
    //{
    //    std::cout << "Cannot find the weights!" << std::endl;
    //    return;
    //}
    //else
    //{
    //    std::cout << "Successfully find the weights!" << std::endl;
    //}
    //std::ifstream wf(w_path.str());
    //json loaded_weights = json::parse(wf, nullptr, true, true);
    mIOData->render_input_mat = new GPUMatrix<float>(NetConfig::n_input_dims, width * height);
    mIOData->render_output_mat = new GPUMatrix<float>(NetConfig::n_output_dims, width * height);
    mIOData->training_input_mat = new GPUMatrix<float>(NetConfig::n_input_dims, width * height);
    mIOData->training_output_mat = new GPUMatrix<float>(NetConfig::n_output_dims, width * height);
    mIOData->random_seq = new GPUMemory<float>(n_train_batch * batch_size);
    //curandGenerateUniform(rng, mIOData->random_seq->data(), n_train_batch * batch_size);

    frame_width = width;
    frame_height = height;
}

NRCNetwork::~NRCNetwork()
{
    delete mNetworkComponents;
    delete mIOData;
}

void NRCNetwork::Test()
{
    //std::cout << "Hello World!" << std::endl;
}


void NRCNetwork ::train(Falcor::RadianceQuery* queries, Falcor::RadianceTarget* targets, float& loss, uint32_t* trainCounts)
{
    //std::cout << "Hello World!" << std::endl;
    //uint32_t n_elements = trainCounts[0].trainCounter;//targets能读到这里怎么会读不到呢
    //uint32_t n_elements = *trainCounts;
    showMsg_counter(trainCounts);//很好，测出来总算对了吐血版
    uint32_t n_elements = frame_width * frame_height;
    mNetworkComponents->optimizer->set_learning_rate(learning_rate);

    linear_kernel(formatInputTarget<float, NetConfig::n_input_dims, NetConfig::n_output_dims>, 0, training_stream, n_elements, queries, targets,
        mIOData->training_input_mat->data(), mIOData->training_output_mat->data(), trainCounts
    );


    auto ctx = mNetworkComponents->trainer->training_step(training_stream, *mIOData->training_input_mat, *mIOData->training_output_mat);
    float tmp_loss = 0;
    tmp_loss = mNetworkComponents->trainer->loss(training_stream, *ctx);
    std::cout << tmp_loss << std::endl;


    //curandGenerateUniform(rng, mIOData->random_seq->data(), n_train_batch * batch_size);
    //for (uint32_t i = 0; i < n_train_batch; i++)
    //{
    //    
    //    linear_kernel(
    //        formatInputTargetRandom<float, NetConfig::n_input_dims, NetConfig::n_output_dims>,
    //        0,
    //        training_stream,
    //        batch_size,
    //        i * batch_size,
    //        queries,
    //        targets,
    //        mIOData->training_input_mat->data(),
    //        mIOData->training_output_mat->data(),
    //        trainCounts,
    //        mIOData->random_seq->data()
    //    );
    //    auto ctx = mNetworkComponents->trainer->training_step(training_stream, *mIOData->training_input_mat, *mIOData->training_output_mat);
    //    float tmp_loss = 0;
    //    tmp_loss = mNetworkComponents->trainer->loss(training_stream, *ctx);
    //    std::cout << tmp_loss << std::endl;
    //}

    //我真的不理解为什么w加了这个，应该是如果网络里放过了就不重复喂了？
    //mNetworkComponents->network->inference(training_stream, *mIOData->training_input_mat, *mIOData->training_output_mat);

    //auto ctx = mNetworkComponents->trainer->training_step(training_stream, *mIOData->training_input_mat, *mIOData->training_output_mat);
    //float tmp_loss = 0;
    //tmp_loss = mNetworkComponents->trainer->loss(training_stream, *ctx);
    //std::cout << tmp_loss << std::endl;
    //json loaded_weights;
    //loaded_weights = mNetworkComponents->trainer->serialize(false);
    //std::cout << loaded_weights.dump(4) << std::endl;
    //if (loaded_weights) std::cout << "Hello World!" << std::endl;
    //else std::cout << "MD World!" << std::endl;
    //std::string network_config_save_path = "network_weights.json";
    //std::ofstream of(network_config_save_path);
    //of << loaded_weights.dump(4);
    //of.close();
    CUDA_CHECK_THROW(cudaStreamSynchronize(training_stream));
}



void NRCNetwork ::forward(Falcor::RadianceQuery* queries, Falcor::RadianceTarget* ptResults, cudaSurfaceObject_t output)
{
    //这里应该可以累加吧
    //json loaded_weights;
    //loaded_weights = mNetworkComponents->trainer->serialize(false);
    //std::cout << loaded_weights.dump(4) << std::endl;

    uint32_t n_elements = frame_width * frame_height;
    //mNetworkComponents->trainer->deserialize(loaded_weights);

    linear_kernel(formatRenderInput<float, NetConfig::n_input_dims>, 0, inference_stream, n_elements, queries, mIOData->render_input_mat->data());
    mNetworkComponents->network->inference(inference_stream, *mIOData->render_input_mat, *mIOData->render_output_mat);
    linear_kernel(
        mapToOutSurf<float, NetConfig::n_output_dims>, 0, inference_stream, n_elements, frame_width, ptResults,
        mIOData->render_output_mat->data(), output
    );
    CUDA_CHECK_THROW(cudaStreamSynchronize(inference_stream));
}

