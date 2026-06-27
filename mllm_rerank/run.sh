#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p output/mllm_rerank logs
python -m mllm_rerank.rerank --config mllm_rerank/config.yaml \
    2>&1 | tee logs/mllm_rerank_$(date +%Y%m%d_%H%M%S).log
