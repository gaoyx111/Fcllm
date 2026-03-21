GPU 利用率只有 1-10% 是性能差距的核心线索。让我分析根因并修复能修的部分。

GPU 利用率极低的根本原因：

1. 每个 token 都要同步加载专家权重（CPU→GPU）：24 层 × 2 个专家 × 1024 个 token ≈ 49,000 次 CPU→GPU 数据传输，每次都阻塞等待
2. Python 用 `torch.nn.functional.scaled_dot_product_attention`（Flash Attention 融合内核），Rust 用手动拆开的 matmul/softmax/matmul 多个独立小内核

这两点是 Candle 框架本身的限制（没有异步 CUDA 流、没有 Flash Attention），短期难以改变。