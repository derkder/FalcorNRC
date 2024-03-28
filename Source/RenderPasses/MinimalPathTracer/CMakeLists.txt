add_plugin(MinimalPathTracer)

target_sources(MinimalPathTracer PRIVATE
    Config.h
    MinimalPathTracer.cpp
    MinimalPathTracer.h
    MinimalPathTracer.rt.slang
    Network.h
    Network.cu
    Params.slang
)

target_link_libraries(MinimalPathTracer PRIVATE tiny-cuda-nn)

target_copy_shaders(MinimalPathTracer RenderPasses/MinimalPathTracer)

target_source_group(MinimalPathTracer "RenderPasses")
