#include "MinimalPathTracer.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, MinimalPathTracer>();
}

namespace
{
const char kShaderFile[] = "RenderPasses/MinimalPathTracer/MinimalPathTracer.rt.slang";
const char kResolveFile[] = "RenderPasses/MinimalPathTracer/ResolvePass.cs.slang";

// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxPayloadSizeBytes = 72u;
const uint32_t kMaxRecursionDepth = 2u;

const char kInputViewDir[] = "viewW";
const char kInputVBuffer[] = "vbuffer";
const char kOutputColor[] = "color";

const ChannelList kInputChannels = {
    // clang-format off
    { kInputVBuffer,        "gVBuffer",     "Visibility buffer in packed format" },
    { kInputViewDir,    "gViewW",       "World-space view direction (xyz float format)", true /* optional */ },
    // clang-format on
};

const ChannelList kOutputChannels = {
    // clang-format off
    { kOutputColor,          "gOutputColor", "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float },
    // clang-format on
};

const char kMaxBounces[] = "maxBounces";
const char kComputeDirect[] = "computeDirect";
const char kUseImportanceSampling[] = "useImportanceSampling";
} // namespace

MinimalPathTracer::MinimalPathTracer(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    parseProperties(props);

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
}

void MinimalPathTracer::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kMaxBounces)
            mMaxBounces = value;
        else if (key == kComputeDirect)
            mComputeDirect = value;
        else if (key == kUseImportanceSampling)
            mUseImportanceSampling = value;
        else
            logWarning("Unknown property '{}' in MinimalPathTracer properties.", key);
    }
}

Properties MinimalPathTracer::getProperties() const
{
    Properties props;
    props[kMaxBounces] = mMaxBounces;
    props[kComputeDirect] = mComputeDirect;
    props[kUseImportanceSampling] = mUseImportanceSampling;
    return props;
}

void MinimalPathTracer::prepareQueryBuffer(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pInputViewDir = renderData.getTexture(kInputViewDir);
    mParams.frameDim = Falcor::uint2(pInputViewDir->getWidth(), pInputViewDir->getHeight());
    const int temp = 3;
    //这里这个buffer的大小有点问题(应该大一点防止溢出？)
    if (!trainQueryBuffer)
    {
        trainQueryBuffer = mpDevice->createStructuredBuffer(
            sizeof(RadianceQuery),
            temp * mParams.frameDim.x * mParams.frameDim.y,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        trainQueryCudaBuffer = createInteropBuffer(mpDevice, temp * sizeof(RadianceQuery) * mParams.frameDim.x * mParams.frameDim.y);
    }
    if (!trainTargetBuffer)
    {
        trainTargetBuffer = mpDevice->createStructuredBuffer(
            sizeof(RadianceTarget),
            temp * mParams.frameDim.x * mParams.frameDim.y,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        trainTargetCudaBuffer = createInteropBuffer(mpDevice, temp * sizeof(RadianceTarget) * mParams.frameDim.x * mParams.frameDim.y);
    }
    // 
    if (!renderQueryBuffer)
    {
        renderQueryBuffer = mpDevice->createStructuredBuffer(
            sizeof(RadianceQuery),
            mParams.frameDim.x * mParams.frameDim.y,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        renderQueryCudaBuffer = createInteropBuffer(mpDevice, sizeof(RadianceQuery) * mParams.frameDim.x * mParams.frameDim.y);
    }

    //其实是只用到第一个元素
    if (!trainCountBuffer)
    {
        trainCountBuffer = mpDevice->createStructuredBuffer(
            sizeof(RadianceCounter),
            mParams.frameDim.x * mParams.frameDim.y,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        trainCountCudaBuffer = createInteropBuffer(mpDevice, sizeof(RadianceCounter) * mParams.frameDim.x * mParams.frameDim.y);
    }

    uint32_t usageFlags = cudaArrayColorAttachment;
    if (!mOutputTex)
    {
        mOutputTex = mpDevice->createTexture2D(mParams.frameDim.x, mParams.frameDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared | ResourceBindFlags::ShaderResource);
        mOutputSurf = cuda_utils::mapTextureToSurface(mOutputTex, usageFlags);
    }
}

RenderPassReflection MinimalPathTracer::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Define our input/output channels.
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

void MinimalPathTracer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Update refresh flag if options that affect the output have changed.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    // If we have no scene, just clear the outputs and return.
    if (!mpScene)
    {
        for (auto it : kOutputChannels)
        {
            Texture* pDst = renderData.getTexture(it.name).get();
            if (pDst)
                pRenderContext->clearTexture(pDst);
        }
        return;
    }

    

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    // Configure depth-of-field.
    const bool useDOF = mpScene->getCamera()->getApertureRadius() > 0.f;
    if (useDOF && renderData[kInputViewDir] == nullptr)
    {
        logWarning("Depth-of-field requires the '{}' input. Expect incorrect shading.", kInputViewDir);
    }

    // Specialize program.
    // These defines should not modify the program vars. Do not trigger program vars re-creation.
    mTracer.pProgram->addDefine("MAX_BOUNCES", std::to_string(mMaxBounces));
    mTracer.pProgram->addDefine("COMPUTE_DIRECT", mComputeDirect ? "1" : "0");
    mTracer.pProgram->addDefine("USE_IMPORTANCE_SAMPLING", mUseImportanceSampling ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    mTracer.pProgram->addDefines(getValidResourceDefines(kInputChannels, renderData));
    mTracer.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars)
        prepareVars();
    FALCOR_ASSERT(mTracer.pVars);

    prepareQueryBuffer(pRenderContext, renderData);
    prepareResolve(renderData);
    // Set constants.
    auto var = mTracer.pVars->getRootVar();
    
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;
    var["gRadianceQuries"] = trainQueryBuffer;
    var["gRadianceTargets"] = trainTargetBuffer;

    var["gRenderQuries"] = renderQueryBuffer;
    var["TrainCount"] = trainCountBuffer;

    
    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            var[desc.texname] = renderData.getTexture(desc.name);
        }
    };
    for (auto channel : kInputChannels)
        bind(channel);
    for (auto channel : kOutputChannels)
        bind(channel);

    // Get dimensions of ray dispatch.
    const Falcor::uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    if (!mNetwork)
    {
        mNetwork = new NRCNetwork(targetDim.x, targetDim.y);
    }

    // Spawn the rays.
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, Falcor::uint3(targetDim, 1));
    //mNetwork->Test();
    NRCTrain(pRenderContext);
    NRCForward(pRenderContext);
    mFrameCount++;

    mpResolvePass->execute(pRenderContext, Falcor::uint3(targetDim, 1));
}

void MinimalPathTracer::NRCForward(RenderContext* pRenderContext)
{
    pRenderContext->copyResource(renderQueryCudaBuffer.buffer.get(), renderQueryBuffer.get());
    mNetwork->forward((RadianceQuery*)renderQueryCudaBuffer.devicePtr, mOutputSurf);
}

void MinimalPathTracer::NRCTrain(RenderContext* pRenderContext)
{
    float loss;
    pRenderContext->copyResource(trainQueryCudaBuffer.buffer.get(), trainQueryBuffer.get());
    pRenderContext->copyResource(trainTargetCudaBuffer.buffer.get(), trainTargetBuffer.get());
    pRenderContext->copyResource(trainCountCudaBuffer.buffer.get(), trainCountBuffer.get());
    //uint32_t* count = (uint32_t*)trainCountCudaBuffer.devicePtr;
    //uint32_t trainCount = count[0];
    mNetwork->train(
        (RadianceQuery*)trainQueryCudaBuffer.devicePtr,
        (RadianceTarget*)trainTargetCudaBuffer.devicePtr,
        loss,
        (RadianceCounter*)trainCountCudaBuffer.devicePtr
    );
}

void MinimalPathTracer::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    dirty |= widget.var("Max bounces", mMaxBounces, 0u, 1u << 16);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    dirty |= widget.checkbox("Evaluate direct illumination", mComputeDirect);
    widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

    dirty |= widget.checkbox("Use importance sampling", mUseImportanceSampling);
    widget.tooltip("Use importance sampling for materials", true);

    // If rendering options that modify the output have changed, set flag to indicate that.
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (dirty)
    {
        mOptionsChanged = true;
    }
}

void MinimalPathTracer::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Clear data for previous scene.
    // After changing scene, the raytracing program should to be recreated.
    mTracer.pProgram = nullptr;
    mTracer.pBindingTable = nullptr;
    mTracer.pVars = nullptr;
    mFrameCount = 0;
    mpResolvePass = nullptr;
    // Set new scene.
    mpScene = pScene;

    if (mpScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("MinimalPathTracer: This render pass does not support custom primitives.");
        }

        // Create ray tracing program.
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("scatterMiss"));
        sbt->setMiss(1, desc.addMiss("shadowMiss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh),
                desc.addHitGroup("scatterTriangleMeshClosestHit", "scatterTriangleMeshAnyHit")
            );
            sbt->setHitGroup(
                1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowTriangleMeshAnyHit")
            );
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("scatterDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection")
            );
            sbt->setHitGroup(
                1,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("", "", "displacedTriangleMeshIntersection")
            );
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::Curve))
        {
            sbt->setHitGroup(
                0, mpScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("scatterCurveClosestHit", "", "curveIntersection")
            );
            sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("", "", "curveIntersection"));
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::SDFGrid))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::SDFGrid),
                desc.addHitGroup("scatterSdfGridClosestHit", "", "sdfGridIntersection")
            );
            sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::SDFGrid), desc.addHitGroup("", "", "sdfGridIntersection"));
        }

        mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());

        ProgramDesc resolveDesc;
        resolveDesc.addShaderLibrary(kResolveFile).csEntry("main");
        resolveDesc.addShaderModules(mpScene->getShaderModules());
        resolveDesc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpSampleGenerator->getDefines());
        defines.add(mpScene->getSceneDefines());
        mpResolvePass = ComputePass::create(mpDevice, resolveDesc, defines);
    }
}

void MinimalPathTracer::prepareResolve(const RenderData& renderData)
{
    auto var = mpResolvePass->getRootVar();
    var["radiance"] = mOutputTex;
    var["gVBuffer"] = renderData.getTexture(kInputVBuffer);
    var["output"] = renderData.getTexture(kOutputColor);
}

void MinimalPathTracer::prepareVars()
{
    FALCOR_ASSERT(mpScene);
    FALCOR_ASSERT(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
    mTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mpDevice, mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    mpSampleGenerator->bindShaderData(var);
}
