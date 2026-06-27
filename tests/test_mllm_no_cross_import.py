import glob


def test_no_module_imports_across_open_sources():
    for path in glob.glob("mllm_rerank/*.py"):
        with open(path) as f:
            src = f.read()
        assert "open-sources" not in src, f"{path} references open-sources"
        assert "sys.path" not in src, f"{path} mutates sys.path"
