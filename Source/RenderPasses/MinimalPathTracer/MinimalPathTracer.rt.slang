#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Params;
import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;

cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.
    
    float3 gSceneAABBCenter;             
    float3 gSceneAABBExtent;
}

// Inputs
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW; // Optional

// Outputs
RWTexture2D<float4> gOutputColor;
RWStructuredBuffer<RadianceQuery> gRadianceQuries;
RWStructuredBuffer<RadianceTarget> gRadianceTargets;
RWStructuredBuffer<RadianceQuery> gRenderQuries;
RWStructuredBuffer<RadianceTarget> gRenderThps;
RWStructuredBuffer<RadianceTarget> gPtResults;//之前路径追踪的结果
RWStructuredBuffer<uint> TrainCount;//cpp里再累加

static float3 lastThp = float3(1.f);
static float3 lastRadiance = float3(0.f);
static bool isTrain;//说明这个像素对应的路径的所有点都放到网络里
//static RadianceCounter radianceCounter = {};

// Static configuration based on defines set from the host.
#define is_valid(name) (is_valid_##name != 0)
static const uint kMaxBounces = MAX_BOUNCES;
static const bool kComputeDirect = COMPUTE_DIRECT;
static const bool kUseImportanceSampling = USE_IMPORTANCE_SAMPLING;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0, 0, 0);
static const float kRayTMax = FLT_MAX;

static float3 recordR[MAX_BOUNCES + 2];//radiance before throughtout
static float3 recordT[MAX_BOUNCES + 2];//throughout
static float3 origins[MAX_BOUNCES + 2]; // record origins
static float footprints[MAX_BOUNCES + 2]; //area spread approaximation
static RadianceQuery recordQ[MAX_BOUNCES + 2];
static bool myTest = false;
static bool startNN = false; //什么时候开始调用网络的结果
static bool smartEnd = true; //用area spread的公式
static int suffix = 0;


/**
 * Payload for shadow ray.
 */
struct ShadowRayData
{
    bool visible;
};

/**
 * Payload for scatter ray (up to 72B).
 */
struct ScatterRayData
{
    float3 dir;
    bool wow;
    float3 radiance;  ///< Accumulated outgoing radiance from path.
    bool terminated;  ///< Set to true when path is terminated.
    float3 thp;       ///< Current path throughput. This is updated at each path vertex.
    uint pathLength;  ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.). Max 2^31.
    float3 origin;    ///< Next path segment origin.
    float pdf;// 用来做spread area终止
    float3 direction; ///< Next path segment direction.
    bool limited; // 当后面的结果可以开始用网络查询的时候,只需要知道thp和query(碰撞点信息)

    RadianceQuery rq; //其实可以不用传一整个这么大的数组，传一部分就好了，因为有些变量和上面的变量的值是一样的
    SampleGenerator sg; ///< Per-ray state for the sample generator (up to 16B).

    /**
     * Initializes ray payload with default parameters.
     */
    __init(SampleGenerator sg)
    {
        this.dir = float3(0, 0, 0);
        this.wow = false;
        this.terminated = false;
        this.pathLength = 0;
        this.radiance = float3(0, 0, 0);
        this.thp = float3(1, 1, 1);
        this.origin = float3(0, 0, 0);
        this.pdf = 0.f;
        this.direction = float3(0, 0, 0);
        this.limited = false;
        this.sg = sg;
        this.rq = {};
    }
};

/**
 * Setup ShadingData based on loaded vertex/material attributes for a hit point.
 * @param[in] hit Hit information.
 * @param[in] rayOrigin Ray origin.
 * @param[in] rayDir Normalized ray direction.
 * @param[in] lod Method for computing texture level-of-detail.
 * @return ShadingData struct.
 */
ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir, const ITextureSampler lod)
{
    VertexData v = {};
    uint materialID = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        v = gScene.getVertexDataFromCurve(curveHit);
        materialID = gScene.getMaterialID(curveHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    if (hit.getType() == HitType::SDFGrid)
    {
        const SDFGridHit sdfGridHit = hit.getSDFGridHit();
        v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);
        materialID = gScene.getMaterialID(sdfGridHit.instanceID);
    }
#endif

    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);

    return sd;
}

/**
 * Returns the primary ray's direction.
 */
float3 getPrimaryRayDir(uint2 launchIndex, uint2 launchDim, const Camera camera)
{
    if (is_valid(gViewW))
    {
        // If we have the view vector bound as a buffer, just fetch it. No need to compute anything.
        return -gViewW[launchIndex].xyz;
    }
    else
    {
        // Compute the view vector. This must exactly match what the G-buffer pass is doing (jitter etc.).
        // Note that we do not take depth-of-field into account as it would require exactly matching the
        // sample generator between the passes, which is error prone. The host side will issue a warning instead.
        return camera.computeRayPinhole(launchIndex, launchDim).dir;
    }
}

/**
 * Traces a shadow ray towards a light source.
 * @param[in] origin Ray origin for the shadow ray.
 * @param[in] dir Direction from shading point towards the light source (normalized).
 * @param[in] distance Distance to the light source.
 * @return True if light is visible, false otherwise.
 */
bool traceShadowRay(float3 origin, float3 dir, float distance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = distance;

    ShadowRayData rayData;
    rayData.visible = false; // Set to true by miss shader if ray is not terminated before
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff /* instanceInclusionMask */,
        1 /* hitIdx */,
        rayTypeCount,
        1 /* missIdx */,
        ray,
        rayData
    );

    return rayData.visible;
}

/**
 * Traces a scatter ray based on ray parameters stored in the ray payload.
 * @param[in] rayData Describes the ray parameters. The struct is modified based on the result.
 */
void traceScatterRay(inout ScatterRayData rayData)
{
    RayDesc ray;
    ray.Origin = rayData.origin;
    ray.Direction = rayData.direction;
    ray.TMin = 0.f;
    ray.TMax = kRayTMax;

    uint rayFlags = 0; // TODO: Set cull mode from the app
    TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);
}

/**
 * Evaluates the direct illumination from analytic lights.
 * This function samples Falcor's light list uniformly with one shadow ray.
 * @param[in] sd Shading data.
 * @param[in] mi Material instance.
 * @param[in,out] sg SampleGenerator object.
 * @return Outgoing radiance in view direction.
 */
float3 evalDirectAnalytic(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg)
{
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0)
        return float3(0.f);

    // Pick one of the analytic light sources randomly with equal probability.
    const uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);
    float invPdf = lightCount; // Light selection pdf = 1.0 / lightCount.

    // Sample local light source.
    AnalyticLightSample ls;
    if (!sampleLight(sd.posW, gScene.getLight(lightIndex), sg, ls))
        return float3(0.f);

    // Reject sample if not in the hemisphere of a BSDF lobe.
    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool hasReflection = lobeTypes & uint(LobeType::Reflection);
    const bool hasTransmission = lobeTypes & uint(LobeType::Transmission);
    float NdotL = dot(sd.getOrientedFaceNormal(), ls.dir);
    if ((NdotL <= kMinCosTheta && !hasTransmission) || (NdotL >= -kMinCosTheta && !hasReflection))
        return float3(0.f);

    // Get origin with offset applied in direction of the geometry normal to avoid self-intersection.
    const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, ls.dir) >= 0.f ? sd.faceN : -sd.faceN);

    // Test visibility by tracing a shadow ray.
    bool V = traceShadowRay(origin, ls.dir, ls.distance);
    if (!V)
        return float3(0.f);

    // Evaluate contribution.
    return mi.eval(sd, ls.dir, sg) * ls.Li * invPdf;
}

/**
 * Generate a new scatter ray or terminate.
 * @param[in] sd Shading data.
 * @param[in] mi Material instance.
 * @param[in] isCurveHit True if on curve hit.
 * @param[in] rayOrigin Ray origin for the new ray.
 * @param[in,out] rayData Ray payload.
 * @return True if the path continues.
 */
bool generateScatterRay(const ShadingData sd, const IMaterialInstance mi, bool isCurveHit, float3 rayOrigin, inout ScatterRayData rayData)
{
    // Sample material.
    BSDFSample bsdfSample;
    if (mi.sample(sd, rayData.sg, bsdfSample, kUseImportanceSampling))
    {
        rayData.origin = rayOrigin;
        if (!isCurveHit && bsdfSample.isLobe(LobeType::Transmission))
        {
            rayData.origin = sd.computeRayOrigin(false);
        }
        rayData.direction = bsdfSample.wo;
        rayData.thp *= bsdfSample.weight;
        rayData.pdf = bsdfSample.pdf;
        return any(rayData.thp > 0.f);
    }

    return false;
}

/**
 * Process a hit.
 * Loads the shading data, samples analytic lights and samples a new scatter ray.
 * Terminates the path if maximum number of bounces is reached.
 * @param[in] hit Hit info.
 * @param[in,out] rayData Ray payload.
 *
 */
//这里因为是另一个shader调用的，奇奇怪怪的，逻辑少往里面放
void handleHit(const HitInfo hit, inout ScatterRayData rayData)//处理当前的hit并计算下一次hit用到的东西.被shader调用
{
    //myTest = true;  
    //if(0 == rayData.pathLength) TrainCount[0] = 5;//第一此进入的时候rayData.pathLength = 0
    //else if(5 != TrainCount[0]) TrainCount[0] = 7;
    const bool isCurveHit = hit.getType() == HitType::Curve;
    let lod = ExplicitLodTextureSampler(0.f);

    // Load shading data.
    ShadingData sd = loadShadingData(hit, rayData.origin, rayData.direction, lod);
    // Create material instance.
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    rayData.rq = generateQuery(sd);
    //recordT[rayData.pathLength + 1] = rayData.thp;

    if(rayData.limited){
        //只需要知道thp与rq就可以了
        rayData.pathLength++;
        return ;
    }

    // Add emitted light.
    if (kUseEmissiveLights && (kComputeDirect || rayData.pathLength > 0))
    {
        rayData.radiance += rayData.thp * mi.getProperties(sd).emission;
        //recordR[kCurBounces] = mi.getProperties(sd).emission;
    }

    // Check whether to terminate based on max depth.后面应该是这里加一个条件就可以实现那种特殊终止了
    if (rayData.pathLength >= kMaxBounces)
    {
        rayData.terminated = true;
        return;
    }

    // Compute ray origin for !new rays! spawned from the hit.
    float3 rayOrigin;
    if (isCurveHit)
    {
        // For curves, we set the new origin at the sphere center.
        rayOrigin = sd.posW - sd.curveRadius * sd.frame.N;
    }
    else
    {
        //Computes new ray origin based on the hit point to avoid self-intersection.
        rayOrigin = sd.computeRayOrigin();
    }

    // Add contribution of direct light from analytic lights.
    if (kUseAnalyticLights)
    {
        float3 Lr = evalDirectAnalytic(sd, mi, rayData.sg);
        rayData.radiance += rayData.thp * Lr;
        //recordR[kCurBounces] += Lr;
    }

    // Generate scatter ray for the next path segment.
    // The raygen shader will continue the path based on the returned payload.
    if (!generateScatterRay(sd, mi, isCurveHit, rayOrigin, rayData))
    {
        rayData.terminated = true;
        return;
    }
    
    rayData.pathLength++;
}

float3 safe_div(float3 a, float3 b)
{
    float3 res = a / b;
    res.x = isinf(res.x) || isnan(res.x) ? 0 : res.x;
    res.y = isinf(res.y) || isnan(res.y) ? 0 : res.y;
    res.z = isinf(res.z) || isnan(res.z) ? 0 : res.z;
    return res;
}

/**
 * This is the main entry point for the minimal path tracer.
 *
 * One path per pixel is generated, which is traced into the scene.
 * The path tracer is written as a for-loop over path segments.
 *
 * Built-in light sources (point, directional) are sampled explicitly at each
 * path vertex. The contributions from area lights (env map and mesh lights)
 * are explicitly added by the scatter ray hit/miss shaders.
 *
 * @param[in] pixel Pixel to trace a path for.
 * @param[in] frameDim Dimension of the frame in pixels.
 * @return Returns the estimated color (radiance).
 */
float3 tracePath(const uint2 pixel, const uint2 frameDim)
{
    //信息不在hitInfo里，在Gbuffer里和rayData里
    //int cur = gRadianceTargets.IncrementCounter();
    float3 outColor = float3(0.f);

    const float3 primaryRayOrigin = gScene.camera.getPosition();
    origins[0] = primaryRayOrigin;
    const float3 primaryRayDir = getPrimaryRayDir(pixel, frameDim, gScene.camera);

    const HitInfo hit = HitInfo(gVBuffer[pixel]);//这里装着hitinfo又是什么意思呢
    uint32_t idx = pixel.x + pixel.y * frameDim.x;
    // if(idx == frameDim.x * frameDim.y - 2){
    //     TrainCount[0] = gRadianceQuries.IncrementCounter();
    //     if(TrainCount[0] > 2000) return float3(1.f); 
    // }

    if (hit.isValid())
    {
        // Pixel represents a valid primary hit. Compute its contribution.
 
        isTrain = hasFourPercentChance(idx, pixel.y, );//当前路径中了4%的概率，要加suffix拿出来做训练
        //if(isTrain) return float3(1.f);
        //else float3(0.f);
        const bool isCurveHit = hit.getType() == HitType::Curve;
        let lod = ExplicitLodTextureSampler(0.f);

        // Load shading data.应该是根据hitInfo选择不同的shader。 好像不是shader只是shadingData
        ShadingData sd = loadShadingData(hit, primaryRayOrigin, primaryRayDir, lod);
        
        // Create material instance at shading point.
        let mi = gScene.materials.getMaterialInstance(sd, lod);

        // Create sample generator.
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gPRNGDimension; i++)
            sampleNext1D(sg);

        // Compute ray origin for new rays spawned from the G-buffer.
        float3 rayOrigin;
        if (isCurveHit)
        {
            // For curves, we set the new origin at the sphere center.
            rayOrigin = sd.posW - sd.curveRadius * sd.frame.N;
        }
        else
        {
            rayOrigin = sd.computeRayOrigin();
        }

        if (kComputeDirect)
        {
            // Always output directly emitted light, independent of whether emissive materials are treated as light sources or not.
            outColor += mi.getProperties(sd).emission;

            // Add contribution of direct light from analytic lights.
            // Light probe and mesh lights are handled by the scatter ray hit/miss shaders.
            outColor += kUseAnalyticLights ? evalDirectAnalytic(sd, mi, sg) : float3(0.f);
        }
        
        // Prepare ray payload.
        ScatterRayData rayData = ScatterRayData(sg);

        // Generate scatter ray.
        if (!generateScatterRay(sd, mi, isCurveHit, rayOrigin, rayData))
            rayData.terminated = true;

        if(isTrain){  
            int idxTrain = gRadianceQuries.IncrementCounter();
            RadianceQuery query = generateQuery(sd);
            recordQ[0] = query;
            recordR[0] = outColor;
            recordT[0] = float3(1.f);
            lastThp = float3(1.f);
            lastRadiance = float3(0.f);
        }
        else{
            if(smartEnd){
                float3 faceN = sd.frontFacing ? sd.faceN : -sd.faceN; //normal        
                float numerator = distance(rayData.origin, origins[0]) * distance(rayData.origin, origins[0]);
                float denominator = 4 * M_PI * dot(normalize(faceN), normalize(rayData.origin - origins[0]));
                footprints[0] = numerator / denominator;
                origins[1] = rayData.origin;
            }
        }
        
        
        //第一次bounce之后的循环
        // Follow path into the scene and compute its total contribution.
        for (uint depth = 0; depth <= kMaxBounces && !rayData.terminated; depth++)
        {
            // Trace scatter ray. If it hits geometry, the closest hit shader samples
            // direct illumination and generates the next scatter ray.
            traceScatterRay(rayData);
            
            //prepare training data
            if(isTrain){
                recordT[depth + 1] = safe_div(rayData.thp, lastThp);
                recordR[depth + 1] = safe_div(rayData.radiance - lastRadiance, rayData.thp);
                lastThp = rayData.thp;
                lastRadiance = rayData.radiance;

                RadianceQuery query = {};
                query = rayData.rq;
                recordQ[depth + 1] = query;

            }
            else{
                if(smartEnd){
                    //footprint足够大可以提前终止
                    if(rayData.limited || (depth == kMaxBounces)){
                        break;
                    }

                    //计算footprint
                    float3 faceN = latlong_map_to_world(rayData.rq.normal); //normal        
                    float numerator = distance(rayData.origin, origins[depth]) * distance(rayData.origin, origins[depth]);
                    float denominator = 4 * M_PI * dot(normalize(faceN), normalize(rayData.origin - origins[depth]));

                    //因为这里是和的平方不是平方和
                    float lastSqrt = sqrt(footprints[depth]);// 前几次的结果平方之前
                    float curFootprint = sqrt(numerator / rayData.pdf / denominator);
                    footprints[depth + 1] = (lastSqrt + curFootprint) * (lastSqrt + curFootprint);
                    origins[depth + 1] = rayData.origin;
                    if(footprints[depth + 1] > footprints[0]){
                        // 开启inference
                        gPtResults[idx].radiance = outColor + rayData.radiance;
                        rayData.limited = true;
                        startNN = true;
                    }
                }
            } 

            if(smartEnd){
                if(rayData.limited){
                    RadianceQuery query = {};
                    query = rayData.rq;
                    gRenderQuries[idx] = query;
                    gRenderThps[idx].radiance = rayData.thp;
                }
                else{
                    gPtResults[idx].radiance = outColor + rayData.radiance;
                }
            }
            else{
                if(depth == 0){
                    gPtResults[idx].radiance = outColor + rayData.radiance;
                }
                else if(depth == 1){
                    gRenderThps[idx].radiance = rayData.thp;
                    RadianceQuery query = {};
                    query = rayData.rq;
                    gRenderQuries[idx] = query;
                }
                else break;
            }
        }

        
        
        if(isTrain){
            ////prepare tranning data
            float3 tempRR = outColor + rayData.radiance;
            int tempIdx = gRadianceTargets.IncrementCounter();
            int tempIdx2 = gRadianceQuries.IncrementCounter();
            RadianceTarget target = {};
            target.radiance = tempRR;
            gRadianceTargets[tempIdx] = target;
            gRadianceQuries[tempIdx2] = recordQ[0];
            int adx = tempIdx;
            TrainCount[0] = tempIdx;
            for(int i = 1; i <= rayData.pathLength; ++i){
                tempIdx = gRadianceTargets.IncrementCounter();
                tempIdx2 = gRadianceQuries.IncrementCounter();
                tempRR = safe_div(tempRR - recordR[i - 1], recordT[i]);
                RadianceTarget target = {};
                target.radiance = tempRR;
                gRadianceTargets[tempIdx] = target;
                gRadianceQuries[tempIdx2] = recordQ[i];
                TrainCount[0] = tempIdx;
            }
            //gPtResults[idx].radiance = outColor + rayData.radiance;
            //gRenderQuries[idx] = generateEmptyQuery();
            //gRenderThps[idx].radiance = float3(0.);
        }
        //if(kCurBounces >= 2) myTest = true;

        
        //outColor = gRenderTargets[idx].radiance;
        outColor = float3(0.f);
        //if(myTest) outColor = float3(1.f);
        //else outColor = float3(0.f, 1.f, 0.f);
        //int idxTrain = gRadianceQuries.IncrementCounter();
        //if(idxTrain > 1000 ) return float3(0., 1., 0.);
        //else if(0 == idxTrain) return float3(1., 1., 1.);
       
    }
    else
    {
        // Background pixel.
        outColor = kUseEnvBackground ? gScene.envMap.eval(primaryRayDir) : kDefaultBackgroundColor;
        // RadianceQuery query = {}; 
        // RadianceTarget target = {};
 
        // gRadianceQuries[idx] = generateEmptyQuery();
        // gRadianceTargets[idx] = target;
    }

    
    return outColor;
}

//
// Shader entry points for miss shaders.
//

[shader("miss")]
void scatterMiss(inout ScatterRayData rayData)
{
    // Ray missed the scene. Mark the ray as terminated.
    rayData.terminated = true;

    // Add contribution from distant light (env map) in this direction.
    if (kUseEnvLight && (kComputeDirect || rayData.pathLength > 0))
    {
        float3 Le = gScene.envMap.eval(WorldRayDirection());
        rayData.radiance += rayData.thp * Le;
    }
}

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

//
// Shader entry points for TriangleMesh hit groups.
//

[shader("anyhit")]
void scatterTriangleMeshAnyHit(inout ScatterRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void scatterTriangleMeshClosestHit(inout ScatterRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    handleHit(HitInfo(triangleHit), rayData);
}

[shader("anyhit")]
void shadowTriangleMeshAnyHit(inout ShadowRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

//
// Shader entry points for DisplacedTriangleMesh hit groups.
//

[shader("intersection")]
void displacedTriangleMeshIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    DisplacedTriangleMeshIntersector::Attribs attribs;
    float t;
    if (DisplacedTriangleMeshIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterDisplacedTriangleMeshClosestHit(inout ScatterRayData rayData, DisplacedTriangleMeshIntersector::Attribs attribs)
{
    DisplacedTriangleHit displacedTriangleHit;
    displacedTriangleHit.instanceID = getGeometryInstanceID();
    displacedTriangleHit.primitiveIndex = PrimitiveIndex();
    displacedTriangleHit.barycentrics = attribs.barycentrics;
    displacedTriangleHit.displacement = attribs.displacement;
    handleHit(HitInfo(displacedTriangleHit), rayData);
}

//
// Shader entry points for Curve hit groups.
//

[shader("intersection")]
void curveIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    CurveIntersector::Attribs attribs;
    float t;
    if (CurveIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterCurveClosestHit(inout ScatterRayData rayData, CurveIntersector::Attribs attribs)
{
    CurveHit curveHit;
    curveHit.instanceID = getGeometryInstanceID();
    curveHit.primitiveIndex = PrimitiveIndex();
    curveHit.barycentrics = attribs.barycentrics;
    handleHit(HitInfo(curveHit), rayData);
}

//
// Shader entry points for SDFGrid hit groups.
//

[shader("intersection")]
void sdfGridIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    SDFGridHitData sdfGridHitData;
    float t;
    if (SDFGridIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), sdfGridHitData, t))
    {
        ReportHit(t, 0, sdfGridHitData);
    }
}

[shader("closesthit")]
void scatterSdfGridClosestHit(inout ScatterRayData rayData, SDFGridHitData sdfGridHitData)
{
    SDFGridHit sdfGridHit;
    sdfGridHit.instanceID = getGeometryInstanceID();
    sdfGridHit.hitData = sdfGridHitData;
    handleHit(HitInfo(sdfGridHit), rayData);
}

//
// Shader entry point for ray generation shader.
//

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;

    float3 color = tracePath(pixel, frameDim);

    gOutputColor[pixel] = float4(color, 1.f);
}

float3 normalizeCoord(float3 x)
{
    return (x - gSceneAABBCenter) * 2 / gSceneAABBExtent; 
}

RadianceQuery generateQuery(ShadingData sd)
{
    RadianceQuery query = { };
    query.pos = sd.posW;
    query.dir = world_to_latlong_map(sd.V);
    float3 faceN = sd.frontFacing ? sd.faceN : -sd.faceN; 
    query.normal = world_to_latlong_map(faceN);    // reverse the normal if the primitive is seen on the back-face side.
    /*    query.diffuse = float4(sd.diffuse, 1);
        query.specular = float4(sd.specular, 1);*/
    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);
    query.diffuse = mi.getProperties(sd).diffuseReflectionAlbedo;
    query.specular = mi.getProperties(sd).specularReflectionAlbedo;
    query.roughness = mi.getProperties(sd).roughness;
    // query.diffuse = p.diffuseReflectionAlbedo;
    // query.specular = p.specularReflectanceAlbedo;
    // query.roughness = p.roughness;
    return query;
}


RadianceQuery generateEmptyQuery()
{
    RadianceQuery query = { };
    query.pos = float3(0.f);
    query.dir = float2(0.f);
    query.normal = float2(0.f);
    query.diffuse = float3(0.f);
    query.specular = float3(0.f);
    query.roughness = 0.f;
    return query;
}


bool hasFourPercentChance(uint idx, uint frameDimx) {
    // 定义常量
    uint a = 1664525;
    uint c = 1013904223;
    uint m = 1 << 16; // 使用 2^32 作为模数

    // 计算种子
    // 注意：由于我们仅限于使用 uint，需确保计算过程中避免溢出
    uint seed = (a * (idx + frameDimx + gFrameCount) % m + c) % m;

    // 生成一个 [0, m-1] 范围内的随机数，并将其标准化为 [0.0, 1.0]
    float randomValue = float(seed) / m;

    // 比较生成的值是否小于 0.04
    return randomValue < 0.04;
}

