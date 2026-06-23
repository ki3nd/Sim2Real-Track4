# LHP→CMP Two-Stage Rerank Eval — Design Spec

**Date:** 2026-06-23
**Scope:** A single-GPU eval script `lhp/rerank_eval.py` that uses the **LHP dual-encoder as stage-1** (top-k candidate selection) and the **trained CMP cross-encoder as stage-2** (rerank only those top-k) on the PAB test set, reporting R@1/R@5/R@10, mAP, mINP. Measures how much CMP reranking adds on top of LHP retrieval.
**Out of scope:** training CMP (assumed already trained), distractor galleries, challenge submission.

## 1. Goal
Reproduce CMP's 2-stage retrieval, but with stage-1 swapped to LHP: LHP picks top-k per query, CMP's cross-encoder + ITM head reranks those candidates. Compare `lhp.eval` (stage-1 only) vs `lhp.rerank_eval` (stage-1 + stage-2) → the R@1/mAP delta is the rerank gain.

## 2. Reranker (confirmed)
CMP checkpoint is **pose-aware** (`be_pose_img=True`) and **no-hard-sampling** (`be_hard=False`, challenge-compliant). Pose-aware ⇒ stage-2 requires **test pose images** at `data_root/pose/<ann["image"]>` (present at `pose/test/...`, matching CMP's `search_test_dataset` convention `image_root + 'pose/' + ann['image']`).

## 3. Approach (A — reuse CMP eval functions)
Reuse CMP's `eval.evaluation_itc`, `eval.evaluation_itm`, `eval.mAP` **verbatim** so the rerank math (top-k via `k_test`, cross-encoder + ITM head, `+0.002*sim` blend, min-fill, normalize) is identical to CMP/paper. Run single-GPU with an `args` object whose `.distributed = False` (so `evaluation_itm`'s `all_reduce`/`barrier` block is skipped; `utils.get_world_size()==1`, rank 0 processes all queries).

## 4. Data flow
```
cmp_cfg = configs/cmp.yaml ; lhp_cfg = lhp/config.yaml
# Single source of test order (gallery images, queries, ids) — pose-aware CMP dataset:
test_ds = search_test_dataset(cmp_cfg, cmp_test_transform)   # .image[], .text[], g_pids[], q_pids[]
cmp_loader = create_loader([test_ds], ...)

# STAGE-2 embeds (CMP, pose-fused):
cmp_model = Search(cmp_cfg); load CMP checkpoint; .eval().cuda()
_, image_embeds, text_embeds, text_atts = evaluation_itc(cmp_model, cmp_loader, bert_tok, device, cmp_cfg)
    # discard CMP's own sims; keep CMP image/text embeds (image_embeds already pose-fused)

# STAGE-1 sims (LHP), SAME order as test_ds:
lhp_model = LHPRetriever(ckpt); .eval().cuda()
img_feats = encode each test_ds.image[i] (LHP transform 384, global) -> [N_img, D]
txt_feats = encode each test_ds.text[i]  (spm + tokenize_caption)     -> [N_txt, D]
lhp_sims  = similarity(img_feats, txt_feats)   # [N_query, N_gallery] (txt @ img.T)

# STAGE-2 rerank (reuse CMP), fed LHP sims for candidate selection:
score = evaluation_itm(cmp_model, device, cmp_cfg, args(distributed=False),
                       sims_matrix=lhp_sims, image_embeds, text_embeds, text_atts)
mAP(score, test_ds.g_pids, test_ds.q_pids)   # R@1/5/10, mAP, mINP
```

## 5. Alignment (index-only; feature dims are independent)
Stage-1 and stage-2 share **only the integer index** (which image/query position) — never feature vectors. LHP feats (768-d) live in LHP space; CMP embeds (1024-d) live in CMP space; `lhp_sims` is computed entirely in LHP space, the rerank entirely in CMP space. So the differing input/embedding dims are irrelevant to correctness.

The only real requirement is **identical ordering**: `lhp_sims[i][j]` index `j` must be the same image as `image_embeds[j]`/`g_pids[j]`, and `i` the same caption as `text_embeds[i]`/`q_pids[i]`. This holds automatically because:
1. **No shuffle** — verified: `create_loader` sets `shuffle=False` for test; `evaluation_itc` builds `image_embeds` (loader order) and `text_embeds` (`dataset.text` order) in dataset order. LHP's encode loaders must likewise use `shuffle=False`.
2. **Same caption-flatten** — both iterate per image then per caption (`search_test_dataset` and `lhp.eval.build_test_index` are identical here).

Simplest way to guarantee both: drive LHP's encode from the **same** `search_test_dataset` instance's `test_ds.image[]`/`test_ds.text[]`. (Two separate datasets over the same `test_file` also align as long as 1+2 hold.) `lhp_sims` shape = `[N_query, N_gallery]` = `[len(test_ds.text), len(test_ds.image)]` (rows=queries, `.topk` over gallery dim).

- `lhp_sims` shape = `[N_query, N_gallery]` = `[len(test_ds.text), len(test_ds.image)]` (what `evaluation_itm` expects: rows=queries, `.topk` over gallery dim).
- LHP tokenizes `test_ds.text[i]` (already `pre_caption`'d by CMP) with spm — harmless.
- LHP loads images as `os.path.join(lhp_cfg["data_root"], test_ds.image[i])` with the global-only transform (`LHPTransform(local_prob=0.0)`).

## 6. Coupling note
This is an **intentional bridge file**: unlike the LHP training module (which must not import CMP), `lhp/rerank_eval.py` deliberately imports CMP (`Search`, `eval.*`, `search_test_dataset`, `BertTokenizer`, `configs/cmp.yaml`) — that is the whole point (CMP is the reranker). Kept in a separate file from `lhp/eval.py` so the pure-LHP eval stays CMP-free.

## 7. CLI / usage
```bash
python -m lhp.rerank_eval \
    --lhp-config lhp/config.yaml --lhp-checkpoint output/lhp/lhp_epoch2.pth --lhp-kind lhp \
    --cmp-config configs/cmp.yaml --cmp-checkpoint <cmp_checkpoint.pth>
```
`--lhp-kind {beit3|lhp}` (reuse the two-kind loader from `lhp.eval`). `k_test` comes from `configs/cmp.yaml` (default 128).

## 8. Prerequisites (user-managed)
- Trained CMP checkpoint (pose-aware, no-hard) + `configs/cmp.yaml` + BERT tokenizer dir (`bert-base-uncased`) + swin/x-vlm config files CMP build needs.
- Test pose images at `data_root/pose/test/...`.
- LHP checkpoint + `beit3.spm` (already used by `lhp.eval`).

## 9. Components / files
- Create `lhp/rerank_eval.py`: arg parsing; load CMP (`Search` + checkpoint) and LHP; build `search_test_dataset` (single order source); LHP encode loops over `test_ds.image`/`test_ds.text` (reuse `lhp.eval.load_retriever`, `LHPTransform`, `tokenize_caption`, `infer.similarity`); call `evaluation_itc`/`evaluation_itm`/`mAP`.
- Reuse: `eval.{evaluation_itc, evaluation_itm, mAP}`, `models.model_search.Search`, `dataset.search_dataset.search_test_dataset`, `dataset.create_loader`, `lhp.eval.load_retriever`, `lhp.transform.LHPTransform`, `lhp.tokenization.tokenize_caption`, `lhp.infer.similarity`.
- No DDP (`args.distributed=False`).

## 10. Error handling
- Missing CMP checkpoint / pose images → clear error (CMP dataset will fail loading pose; surface the path).
- Shape/order mismatch guard: assert `lhp_sims.shape == (len(test_ds.text), len(test_ds.image))` before `evaluation_itm`.

## 11. Testing
- Unit (no weights): an `_args` helper exposes `.distributed=False`; a shape/orientation test that `lhp_sims` built from dummy feats has shape `[N_query, N_gallery]` and the alignment assert passes. (Full rerank needs CMP+LHP weights+pose → smoke.)
- Smoke (needs both checkpoints + pose + GPU): run end-to-end, confirm a metrics table prints and mAP ≥ the stage-1-only `lhp.eval` number.
