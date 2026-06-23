# LHP Eval on PAB Test — Design Spec

**Date:** 2026-06-23
**Scope:** A single-GPU evaluation script `lhp/eval.py` that scores the LHP/BeiT-3 **dual-encoder** retriever on the PAB **test** set and reports R@1/R@5/R@10, mAP, mINP — to compare the zero-shot BeiT-3 COCO baseline vs the LHP-fine-tuned checkpoint.
**Out of scope:** cross-encoder rerank (BeiT-3 is a dual encoder), distractor galleries, the challenge name-masked test, `answer.txt` submission.

## 1. Goal
Measure how much LHP fine-tuning improves retrieval on PAB test, using **the same scoring as CMP/the paper** (reuse `eval.mAP`). Run once per checkpoint and compare.

## 2. Scoring (reused from CMP, confirmed)
- Test set format (`search_test_dataset`): each JSON record = `{image, image_id, caption: [list ≥1]}`. Gallery = images (`g_pids` = each image's `image_id`); queries = all captions flattened (`q_pids` = owning image's id). Match = same `image_id`.
- `eval.mAP(scores_t2i, g_pids, q_pids)` takes a `[num_query, num_gallery]` matrix → R@1/R@5/R@10, mAP, mINP. **Imported and reused as-is** (single source of truth, identical to CMP).
- BeiT-3/LHP is a dual encoder → score matrix is **stage-1 only**: `txt_feats @ img_feats.T` (no ITM rerank).

## 3. Data flow
```
args: --config lhp/config.yaml  --checkpoint <ckpt>  --kind {beit3|lhp}
build LHPRetriever + load ckpt (see §5) → .eval().cuda()
test set (CMP format): test_file → images[] + g_pids[], captions(flattened) + q_pids[]
  test transform = LHPTransform(local_prob=0.0)  # global-only: resize 384 + Inception norm, deterministic
encode images (batched, no_grad) → img_feats [N_img, D]   (model heads already L2-normalize)
encode captions (batched)        → txt_feats [N_txt, D]
scores_t2i = (txt_feats @ img_feats.T).cpu().numpy()      # [N_query, N_gallery]
from eval import mAP; mAP(scores_t2i, g_pids, q_pids)      # prints table
```
Tokenization: `XLMRobertaTokenizer(cfg["spm_model"])` + `lhp.tokenization.tokenize_caption` (max_tokens 64), same as train.

## 4. Config additions (`lhp/config.yaml`)
```yaml
test_file: 'annotation/test/attr.json'   # relative to data_root (like train_file)
batch_size_eval: 64
```
Test images resolved as `os.path.join(data_root, ann["image"])` (same data_root as train).

## 5. Checkpoint loading — TWO kinds (load-bearing)
- `--kind beit3` (baseline, zero-shot COCO): the file is a `BEiT3ForRetrieval` state_dict → load via `LHPRetriever(ckpt_path=<path>)` (goes through `load_model_and_may_interpolate`, interpolates pos-embed).
- `--kind lhp` (trained): the file is `{"model": <LHPRetriever state_dict>, "config": ...}` (keys prefixed with the wrapper's `beit3.`) → build `LHPRetriever(ckpt_path=None)`, then `model.load_state_dict(torch.load(path, map_location="cpu")["model"])`.

## 6. CLI / usage
```bash
# baseline (zero-shot COCO)
python -m lhp.eval --config lhp/config.yaml --kind beit3 \
    --checkpoint checkpoint/beit3_base_patch16_384_coco_retrieval.pth
# after LHP
python -m lhp.eval --config lhp/config.yaml --kind lhp \
    --checkpoint output/lhp/lhp_epoch2.pth
```
The R@1/mAP delta between the two runs = LHP fine-tuning improvement.

## 7. Components / files
- Create `lhp/eval.py` (single file): arg parsing, checkpoint loading (§5), a small test-set reader (CMP format), encode loops, score, `mAP` call.
- Reuse: `lhp.model.LHPRetriever`, `lhp.transform.LHPTransform`, `lhp.tokenization.tokenize_caption`, `dataset.utils.{read_json_to_list, pre_caption}`, `eval.mAP`.
- No DDP (test set ~1978 images + ~1978 captions → trivial on 1 GPU).

## 8. Error handling
- Missing `test_file` / images → clear error.
- Wrong `--kind` (e.g., loading an lhp ckpt as beit3) → key-mismatch; log missing/unexpected keys to surface it.

## 9. Testing
- Unit (no weights): test-set reader returns `(images, g_pids, captions, q_pids)` with correct lengths/ids from a tiny temp annotation; score-matrix orientation `[N_query, N_gallery]`; `mAP` on a hand-built perfect-ranking matrix returns R@1=100.
- Smoke (needs weights+GPU+PAB test): run both `--kind` once, confirm a metrics table prints.
```
