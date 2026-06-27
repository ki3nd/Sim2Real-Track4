#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p output/cgcr logs
python -m mllm_rerank.cgcr --config mllm_rerank/cgcr_config.yaml \
    2>&1 | tee logs/cgcr_$(date +%Y%m%d_%H%M%S).log
