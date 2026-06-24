# Sim2Real — Text-based Person Anomaly Search (AI City Challenge 2026, Track 4 / ECCV 2026)

This repository is our team's solution for the **[AI City Challenge 2026 — Track 4: Text-Based Person Anomaly Search (Sim2Real)](https://www.aicitychallenge.org/2026-track4/)** (ECCV 2026).

It builds on the official **CMP** framework (see References) and adds an **LHP** module in [`lhp/`](lhp/): a BeiT-3 dual-encoder retriever trained with Local-global Hybrid Perspective views (local / global / masked-attention). LHP serves as the stage-1 retriever and can be reranked by CMP's cross-encoder. See [`lhp/README.md`](lhp/README.md) for the LHP training, evaluation, and LHP→CMP rerank pipelines.

---

## Reference repos

- BeiT-3 (Microsoft UniLM): https://github.com/microsoft/unilm/tree/master/beit3
- CMP (Shuyu-XJTU): https://github.com/Shuyu-XJTU/CMP
