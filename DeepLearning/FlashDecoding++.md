# FlashDecoding++: Faster Large Language Model Inference On GPUs

## Introduction

为了提高 softmax 并行度，之前的方法(FlashAttention、FlashDecoding)将计算过程拆分，各自计算 partial softmax 的结果，最后需要通过同步操作来更新 partial softmax 的结果，而 Flash Decoding 在最后统一更新所有  partial softmax 的结果。