## Intro
A plain implementation of [Neural Radiance Cache](https://research.nvidia.com/publication/2021-06_real-time-neural-radiance-caching-path-tracing) with falcor and very low frame rate unfortunately.

## 简介
使用falcor + tiny cuda nn实现 [NRC](https://d1qx31qr3h6wln.cloudfront.net/publications/mueller21realtime.pdf)   
实现是直接叠加在 [miniPathTracer](https://github.com/derkder/FalcorNRC/tree/ptWithNetwork/Source/RenderPasses/MinimalPathTracer)上的（没有新建pass）    
真正的分支就是这里的ptWithNetwork

## 编译
参考https://github.com/yijie21/Falcor-tiny-cuda-nn
（我是完全跟着上面的教程打通falcor + tiny-cuda-nn的流程的）
