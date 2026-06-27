from mllm_rerank.squad import run_squad


class _ScriptedLLM:
    """Detective says Yes for candidate j==0 only; Analyst returns a checklist; nothing else needed."""
    def __init__(self):
        self.calls = {"img": 0, "txt": 0}

    def generate_response_multi_images(self, questions, images, t=0.01):
        self.calls["img"] += 1
        out = []
        for q in questions:
            if "STRICTLY" in q:                 # Detective prompt
                # the test encodes the candidate index in the image placeholder
                out.append("Yes" if images[len(out)] == "cand0" else "No")
            else:                               # Analyst prompt
                out.append("1.The person is male.\n2.Short black hair.\n")
        return out

    def generate_response_text(self, questions, t=0.01):
        self.calls["txt"] += 1
        return ['{"caption": "A man with short black hair."}' for _ in questions]


def test_run_squad_keeps_only_detective_yes():
    llm = _ScriptedLLM()
    queries = ["a man in red", "a woman walking"]
    cand_paths = [["cand0", "candX"], ["candX", "candX"]]  # only query0/cand0 -> "Yes"
    gate_mask = [True, True]
    out = run_squad(llm, queries, cand_paths, gate_mask)
    assert (0, 0) in out
    assert out[(0, 0)] == "A man with short black hair."
    # query0 cand1 ("candX") => "No"; query1 both "candX" => "No"
    assert (0, 1) not in out
    assert (1, 0) not in out and (1, 1) not in out


def test_run_squad_skips_gated_out_queries():
    llm = _ScriptedLLM()
    queries = ["a man in red"]
    cand_paths = [["cand0", "candX"]]
    gate_mask = [False]                          # gated out => no MLLM calls
    out = run_squad(llm, queries, cand_paths, gate_mask)
    assert out == {}
    assert llm.calls["img"] == 0 and llm.calls["txt"] == 0
