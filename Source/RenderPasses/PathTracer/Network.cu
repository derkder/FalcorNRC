﻿#include "Network.h"

#include <fstream>
#include <iostream>
#include <filesystem/path.h>

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
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
    GPUMatrix<float>* input_mat = nullptr;
    GPUMatrix<float>* output_mat = nullptr;

    GPUMatrixDynamic<float>* training_input_mat = nullptr;
    GPUMatrix<float>* training_output_mat = nullptr;
};

cudaStream_t inference_stream = nullptr;
cudaStream_t training_stream = nullptr;

NetworkComponents* mNetworkComponents = nullptr;

IOData* mIOData = nullptr;

} // namespace

// template<uint32_t stride, typename T = float>
//__global__ void generateTrainingDataFromSamples(
//     uint32_t n_elements,
//     uint32_t offset,
//     Falcor::RadianceSample* samples,
//     Falcor::RadianceQuery* self_queries,
//     T* self_query_pred,
//     T* training_data,
//     T* training_target,
//     uint32_t* training_sample_counter,
//     uint32_t* self_query_counter,
//     float* random_indices = nullptr
//)
//{
//     uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i + offset > n_elements)
//         return;
//     int data_index = i * stride, sample_index = i + offset;
//     if (random_indices)
//         sample_index = (1 - random_indices[sample_index]) * *training_sample_counter;
//
//     int pred_index = samples[sample_index].idx; // pred_index == -1 if a self-query is not needed.
//
//     if (sample_index < *training_sample_counter)
//     {
//         float3 factor = samples[sample_index].a, bias = samples[sample_index].b;
//         uint32_t output_index = i * 3;
//
//         copyQuery(&training_data[data_index], &samples[sample_index].query);
//
//         float3 pred_radiance = {0, 0, 0};
//         if (pred_index >= 0) // else the sample doesn't contain a self query.
//             pred_radiance = {self_query_pred[pred_index * 3], self_query_pred[pred_index * 3 + 1], self_query_pred[pred_index * 3 + 2]};
// #if REFLECTANCE_FACT
//         float3 reflectance = samples[sample_index].query.diffuse + samples[sample_index].query.specular;
//         if (pred_index >= 0)
//             // restore self-query from reflectance factorization...
//             pred_radiance = pred_radiance * (self_queries[pred_index].diffuse + self_queries[pred_index].specular);
//         float3 radiance = safe_div(pred_radiance * factor + bias, reflectance);
// #else
//         float3 radiance = pred_radiance * factor + bias;
// #endif
//         *(float3*)&training_target[output_index] = radiance;
//     }
// }
//
//template<typename T, uint32_t stride>
//__global__ void formatInput(uint32_t n_elements, Falcor::RadianceQuery* queries, T* input)
//{
//    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i >= n_elements)
//        return;
//
//    Falcor::RadianceQuery query = queries[i];
//
//    input[i * stride + 0] = query.posW.x;
//    input[i * stride + 1] = query.posW.y;
//    input[i * stride + 2] = query.posW.z;
//    input[i * stride + 3] = query.normalW.x;
//    input[i * stride + 4] = query.normalW.y;
//    input[i * stride + 5] = query.normalW.z;
//    input[i * stride + 6] = query.wiW.x;
//    input[i * stride + 7] = query.wiW.y;
//    input[i * stride + 8] = query.wiW.z;
//    input[i * stride + 9] = query.diff.x;
//    input[i * stride + 10] = query.diff.y;
//    input[i * stride + 11] = query.diff.z;
//}
//
//template<typename T, uint32_t stride>
//__global__ void mapToOutSurf(uint32_t n_elements, uint32_t width, T* output, cudaSurfaceObject_t outSurf)
//{
//    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i >= n_elements)
//        return;
//
//    uint32_t x = i % width;
//    uint32_t y = i / width;
//
//    float4 color = {0, 0, 0, 1};
//
//    color.x = output[i * stride + 0];
//    color.y = output[i * stride + 1];
//    color.z = output[i * stride + 2];
//
//    surf2Dwrite(color, outSurf, x * sizeof(float4), y);
//}

NRCNetwork :: NRCNetwork(const uint32_t width, const uint32_t height)
{
    CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
    CUDA_CHECK_THROW(cudaStreamCreate(&training_stream));

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

    filesystem::path w_path(NetConfig::weightsPath);
    if (!w_path.exists())
    {
        std::cout << "Cannot find the weights!" << std::endl;
        return;
    }
    else
    {
        std::cout << "Successfully find the weights!" << std::endl;
    }
    std::ifstream wf(w_path.str());
    json loaded_weights = json::parse(wf, nullptr, true, true);

    // 下面注释掉了还能跑，我感觉是要注释掉的，应该换成训练好的weight但是好像也没有用这个就是了。
    // 但是注释掉不注释掉渲染出的图像是不一样的，所以反正之前是用了权重数据的
    // mNetworkComponents->trainer->deserialize(loaded_weights);

    mIOData->input_mat = new GPUMatrix<float>(NetConfig::n_input_dims, width * height);
    mIOData->output_mat = new GPUMatrix<float>(NetConfig::n_output_dims, width * height);
    mIOData->training_input_mat = new GPUMatrix<float>(NetConfig::n_input_dims, width * height);
    mIOData->training_output_mat = new GPUMatrix<float>(NetConfig::n_output_dims, width * height);

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
    std::cout << "Hello World!" << std::endl;
}

//
//void NRCNetwork::forward(Falcor::RadianceQuery* queries, cudaSurfaceObject_t output)
//{
//    uint32_t n_elements = frame_width * frame_height;
//
//    // 将查询数据（queries）转换为神经网络的输入张量
//    linear_kernel(formatInput<float, NetConfig::n_input_dims>, 0, inference_stream, n_elements, queries, mIOData->input_mat->data());
//
//    mNetworkComponents->network->inference(inference_stream, *mIOData->input_mat, *mIOData->output_mat);
//
//    // 将网络推断的结果映射或渲染到输出表面
//    linear_kernel(
//        mapToOutSurf<float, NetConfig::n_output_dims>, 0, inference_stream, n_elements, frame_width, mIOData->output_mat->data(), output
//    );
//}
//
//// 这里pada一些权重文件会改吗，我吐了
//// 这个数据到底是从trainingstream里读还是queries里读的啊
//// 这个linear_kernel应该是用来转成可以放到网络里的形式吧
//void NRCNetwork::train(Falcor::RadianceQuery* queries, cudaSurfaceObject_t targets, float& loss)
//{
//    uint32_t n_elements = frame_width * frame_height;
//    mNetworkComponents->optimizer->set_learning_rate(learning_rate);
//
//    /// self query,大胆猜测这个和上面的forward前两句一个道理
//    // linear_kernel(
//    //     formatInput<float, NetConfig::n_input_dims>,
//    //     0,
//    //     training_stream,
//    //     self_query_batch_size,
//    //     0,
//    //     self_queries,
//    //     mMemory->training_self_query->data()
//    //);
//    // mNetwork->network->inference(training_stream, *mMemory->training_self_query, *mMemory->training_self_pred);
//    // 上面的mMemory->training_self_query就是*mIOData->input_mat，*mMemory->training_self_pre就是*mIOData->output_mat
//
//    linear_kernel(formatInput<float, NetConfig::n_input_dims>, 0, training_stream, n_elements, queries, mIOData->input_mat->data());
//    mNetworkComponents->network->inference(training_stream, *mIOData->input_mat, *mIOData->output_mat);
//
//    // training
//    // randomly select 4 training batches over all samples
//    /*curandGenerateUniform(rng, mMemory->random_seq->data(), n_train_batch * batch_size);
//    for (uint32_t i = 0; i < n_train_batch; i++)
//    {
//        linear_kernel(
//            generateTrainingDataFromSamples<input_dim, float>,
//            0,
//            training_stream,
//            batch_size,
//            i * batch_size,
//            training_samples,
//            self_queries,
//            mMemory->training_self_pred->data(),
//            mMemory->training_data->data(),
//            mMemory->training_target->data(),
//            training_sample_counter,
//            self_query_counter,
//            mMemory->random_seq->data()
//        );
//        mNetworkComponents->trainer->training_step(training_stream, *mMemory->training_data, *mMemory->training_target, &loss);
//    }*/
//
//    // auto loss_object = reinterpret_cast<tcnn::Loss<tcnn::network_precision_t>*>(&loss);
//    // std::shared_ptr<tcnn::Loss<tcnn::network_precision_t>> loss_ptr(loss_object);
//    // mNetworkComponents->trainer->set_loss(loss_ptr);
//    // 我浅浅当第三个input的target是我pt算出来的结果
//    mNetworkComponents->trainer->training_step(training_stream, *mIOData->input_mat, *mIOData->output_mat);
//    // 下面的运行了，他只是不更新网络权重罢了。上面的最后一个输入参数改成input他就会闪退，所以我感觉是执行了的
//    // std::cout << "Hello World!" << std::endl;
//    //  确保所有CUDA操作都已完成
//    CUDA_CHECK_THROW(cudaStreamSynchronize(training_stream));
//}
