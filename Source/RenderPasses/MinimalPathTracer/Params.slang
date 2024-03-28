#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

struct CommonParams {
    uint2   frameDim = { 0, 0 };
    uint    frameCount = 0;
};

struct RadianceQuery
{
    float3 pos;     // the frequency encoding needs inputs to be mapped into [-1, 1]
    float roughness;// 
    float2 dir;     // the oneblob encoding needs inputs to be mapped in [0, 1]
    float2 normal;  // 
    float3 diffuse; // these parameters are passed through, since they preserve linearity    
    float _pad0;
    float3 specular;
    float _pad1;
};

END_NAMESPACE_FALCOR
