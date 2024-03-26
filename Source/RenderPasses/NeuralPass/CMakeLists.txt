# step 1: add main.cu in target_sources
# step 2: add tiny-cuda-nn in target_link_libraries
add_plugin(NeuralPass)

target_sources(NeuralPass PRIVATE
    NeuralPass.cpp
    NeuralPass.h
    main.cu
)

target_link_libraries(NeuralPass PRIVATE tiny-cuda-nn)

target_source_group(NeuralPass "RenderPasses")
