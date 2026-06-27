import glob
import os


def test_no_module_imports_across_open_sources():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = glob.glob(os.path.join(repo_root, "mllm_rerank", "*.py"))
    assert files, "no mllm_rerank/*.py files found"
    for path in files:
        with open(path) as f:
            src = f.read()
        assert "open-sources" not in src, f"{path} references open-sources"
        assert "sys.path" not in src, f"{path} mutates sys.path"
