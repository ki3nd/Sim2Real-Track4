"""Qwen3-VL-8B vLLM wrapper + batched inference (ported from SSDC vllm_infer_SSDC.py)."""
import os
import gc
from tqdm import tqdm


class MLLMs(object):
    def __init__(self, model_dir, gpu_memory_utilization=0.7, max_model_len=1536):
        import torch                                    # deferred
        from vllm import LLM, SamplingParams           # deferred
        from transformers import AutoProcessor
        self._SamplingParams = SamplingParams
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                              "expandable_segments:True,max_split_size_mb:256")
        self.model_dir = model_dir
        self.llm = LLM(
            model=model_dir, tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=256, max_model_len=max_model_len,
            enforce_eager=True, disable_log_stats=True, trust_remote_code=True,
            limit_mm_per_prompt={"image": 1}, enable_chunked_prefill=True,
            max_num_batched_tokens=4096, dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(model_dir)

    def generate_response_multi_images(self, questions, images=None,
                                       sys="You are a helpful assistant.", t=0.01):
        import torch                                    # deferred
        from qwen_vl_utils import process_vision_info  # deferred
        try:
            messages = [
                [{"role": "system", "content": sys},
                 {"role": "user", "content": [
                     {"type": "image", "image": images[i], "min_pixels": 50176, "max_pixels": 50176},
                     {"type": "text", "text": p}]}]
                for i, p in enumerate(questions)]
            prompts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                       for m in messages]
            image_data = [process_vision_info(m)[0] for m in messages]
            inputs = [{"prompt": p, "multi_modal_data": {"image": image_data[i]}}
                      for i, p in enumerate(prompts)]
            sp = self._SamplingParams(temperature=t, max_tokens=512, skip_special_tokens=True)
            outputs = self.llm.generate(inputs, sampling_params=sp)
            results = [o.outputs[0].text for o in outputs]
            del inputs, outputs, image_data, messages, prompts
            torch.cuda.empty_cache(); gc.collect()
            return results
        except Exception as e:
            print(f"Generation failed: {e}")
            torch.cuda.empty_cache(); gc.collect()
            return [""] * len(questions)

    def generate_response_text(self, questions, sys="You are a helpful assistant.", t=0.01):
        import torch                                    # deferred
        try:
            messages = [
                [{"role": "system", "content": sys},
                 {"role": "user", "content": [{"type": "text", "text": p}]}]
                for p in questions]
            prompts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                       for m in messages]
            inputs = [{"prompt": p} for p in prompts]
            sp = self._SamplingParams(temperature=t, max_tokens=512, skip_special_tokens=True)
            outputs = self.llm.generate(inputs, sampling_params=sp)
            results = [o.outputs[0].text for o in outputs]
            del inputs, outputs, messages, prompts
            torch.cuda.empty_cache(); gc.collect()
            return results
        except Exception as e:
            print(f"Text generation failed: {e}")
            torch.cuda.empty_cache(); gc.collect()
            return [""] * len(questions)


def batch_infer(llm, b_prompts, images, micro_batch=8, t=0.01):
    results = []
    n = len(b_prompts)
    n_batches = (n - 1) // micro_batch + 1 if n else 0
    for i in tqdm(range(n_batches), desc="Batch Inference"):
        start, end = i * micro_batch, min((i + 1) * micro_batch, n)
        try:
            rs = llm.generate_response_multi_images(
                questions=b_prompts[start:end], images=images[start:end], t=t)
            results += rs
        except Exception as e:
            print(f"Batch {i+1} failed: {e}")
            results += [""] * (end - start)
    return results


def batch_infer_txt(llm, b_prompts, micro_batch=16, t=0.01):
    results = []
    n = len(b_prompts)
    n_batches = (n - 1) // micro_batch + 1 if n else 0
    for i in tqdm(range(n_batches), desc="Text Inference"):
        start, end = i * micro_batch, min((i + 1) * micro_batch, n)
        try:
            rs = llm.generate_response_text(questions=b_prompts[start:end], t=t)
            results += rs
        except Exception as e:
            print(f"Text batch {i+1} failed: {e}")
            results += [""] * (end - start)
    return results
