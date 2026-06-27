import ast
from mllm_rerank.mllm import batch_infer, batch_infer_txt


class _FakeLLM:
    def generate_response_multi_images(self, questions, images, t=0.01):
        return [f"img:{q}" for q in questions]

    def generate_response_text(self, questions, t=0.01):
        return [f"txt:{q}" for q in questions]


def test_batch_infer_preserves_order_and_count():
    prompts = [f"p{i}" for i in range(20)]
    images = [None] * 20
    out = batch_infer(_FakeLLM(), prompts, images, micro_batch=8, t=0.01)
    assert out == [f"img:p{i}" for i in range(20)]


def test_batch_infer_txt_preserves_order_and_count():
    prompts = [f"q{i}" for i in range(35)]
    out = batch_infer_txt(_FakeLLM(), prompts, micro_batch=16, t=0.01)
    assert out == [f"txt:q{i}" for i in range(35)]


def test_module_has_no_cross_repo_import():
    with open("mllm_rerank/mllm.py") as f:
        src = f.read()
    assert "open-sources" not in src
    assert "sys.path" not in src
