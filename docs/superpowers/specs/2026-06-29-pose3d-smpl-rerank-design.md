# 3D Body-Geometry Pose Branch (SMPL → Normal-Map) — Design Spec

**Date:** 2026-06-29
**Short name:** `pose3d`
**Scope:** Replace CMP's current 2D pose-skeleton image branch with a **3D
body-geometry** representation built from SMPL: render a camera-aligned
**surface normal map** (224²) into the existing `pose_conv` branch, plus a few
**explicit orientation scalars** (optional fused token). The 3D body extractor is
a **public, frozen** model run offline; CMP is retrained on synthetic train as
before. Eval on PAB test. Dual goal: (a) close the Sim2Real gap with a
texture-invariant geometric signal, (b) capture static-frame anomalous behavior
(body orientation relative to gravity) better than a 2D skeleton.
**Out of scope:** large cross-modal architecture changes, training the 3D
extractor, any use of test images/text for training, hard-negative mining,
LoRA/SFT.

## 1. Goal

Lift Sim2Real robustness and behavior discrimination over the current
pose-skeleton branch by feeding CMP a **render-invariant 3D body geometry**
instead of a 2D skeleton image. The crux of the task — inferring anomalous
behavior from a single still — is carried mostly by **body orientation relative
to gravity** (standing vs lying/falling/climbing), which a 3D body captures
directly and a 2D skeleton only partially. The same geometric signal is far less
synthetic-biased than RGB texture, attacking the synthetic→real gap at the same
time.

## 2. Challenge compliance — no test data in training

Separate **representation extraction** from **training**:
- The 3D body extractor (Multi-HMR / HMR2.0 / SMPLer-X) is **public, pretrained,
  frozen** — it is only ever run in inference to extract a representation; its
  weights are never updated.
- **Training:** the extractor runs on **synthetic train** images; CMP learns only
  on synthetic train.
- **Test:** the extractor runs on **real test** images at **inference** time —
  the same status as CMP reading test pixels to rank. No test image/text enters
  any gradient.
- No LoRA/SFT of the extractor. CMP remains the legitimate trainable retriever,
  trained exactly as today. No hard sampling introduced.

## 3. 3D extractor model choice

| Role | Model | Rationale |
|---|---|---|
| **Primary** | **Multi-HMR (2024)** | One-shot **multi-person** whole-body SMPL-X, **detector-free**, fast → fits both PAB's multi-person frames and 1M-image throughput |
| Proven alternative | **4DHumans / HMR2.0** (ViT) | Very robust single-person SMPL; needs a person detector (ViTDet/Detectron2) for bboxes. Use if Multi-HMR quality on real test disappoints |
| Quality ceiling | **SMPLer-X** | SOTA cross-dataset generalization (good for Sim2Real), heavier — reserve for pushing accuracy if needed |

**Renderer:** PyTorch3D, rendering a **surface normal map** in **camera view**
(preserves spatial correspondence with the RGB image so the conv branch fuses at
matching locations). An optional **depth** channel may be added later; normal map
is the v1 default.

All extractor/renderer dependencies are **deferred imports** inside the offline
preprocessing script — never imported at module top level, never on the
train/eval/dataset path.

## 4. Representation & orientation scalars

From the selected person's SMPL:
- **Image:** camera-view normal map, 224², 3-channel → drop-in for `pose_conv`.
- **Orientation scalars** (derived from `global_orient` + joints, essentially
  free once SMPL is recovered):
  - `torso_tilt`: angle of the torso axis (pelvis→neck) from world vertical —
    upright ≈ 0°, supine ≈ 90°.
  - `head_below_hip`: continuous flag for head lower than hips (falling / lying /
    handstand).
  - `body_bend`: hip/knee flexion magnitude.

## 5. Architecture — two staged versions (isolate variables)

**v1 — core, minimal change.** Swap only the *content* of the pose branch image:
`pose/<split>/*.webp` (skeleton) → `pose3d/<split>/*.webp` (normal map).
`pose_conv` + `be_pose_img` stay identical; CMP is retrained unchanged. This
yields a **clean A/B**: "does 3D geometry beat the 2D skeleton?" — one variable.

**v2 — orientation token.** Orientation scalars → small MLP → one token fused on
the image side of the cross-modal encoder, guarded by a `be_orient_token` flag.
Ablate its marginal gain on top of v1.

## 6. Offline preprocessing pipeline (run once)

```
for each image (synthetic train + real test):
  Multi-HMR(image) -> [SMPL-X params, camera] for every detected person
  select dominant person (largest bbox area; tie -> closest to center)
  render normal map (camera view, 224²) -> pose3d/<split>/<name>.webp
  compute orientation scalars -> write to an index (parquet/json keyed by image name)
  on failure (no person / low confidence / render error):
     fallback -> reuse the existing 2D skeleton image + neutral orientation (0)
     record the image in a fallback manifest (to measure 3D coverage %)
```

Caching mirrors the current `pose/` mechanism, so train/eval only switch the
source directory (`pose_dir`).

## 7. Files

- **`tools/extract_3d_pose.py`** — offline extractor + renderer + orientation
  writer. Heavy libs (Multi-HMR, pytorch3d) **deferred-imported** inside
  functions, not at top level.
- **`pose3d/orientation.py`** — pure orientation-scalar math + dominant-person
  selection (unit-testable, no heavy deps).
- **`dataset/search_dataset.py`** — add `pose_dir` config key (default `pose`,
  set to `pose3d`) + optional orientation-vector loading.
- **`models/model_search.py`** — fuse the orientation token (v2), guarded by
  `be_orient_token`.
- **`configs/cmp.yaml`** — new keys: `pose_dir: pose3d`, `be_orient_token: false`.
- **`tests/`** — see Section 9.

## 8. Error handling

- No person detected → fallback to 2D skeleton image + zero orientation; count in
  manifest.
- Multiple people → dominant person by bbox area (+ centrality tiebreak),
  deterministic.
- Render failure → fallback. Report the **percentage of images with valid 3D** —
  a reliability indicator for the whole approach.

## 9. Testing (CPU, no GPU / no heavy models — repo discipline)

- **`orientation.py`:** small synthetic SMPL tensors → `torso_tilt` correct
  (upright ≈ 0, supine ≈ 90), `head_below_hip` correct sign; pure math.
- **Dominant-person selection:** a list of bboxes → picks largest / most central;
  ties broken deterministically.
- **Fallback logic:** empty extractor output → uses skeleton path + zero
  orientation; manifest records the fallback.
- **Render contract:** produces a 224² 3-channel image file (mock renderer; test
  I/O + shape via the save/normalize wrapper).
- **Config wiring:** `pose_dir` switches the source to `pose3d`;
  `be_orient_token` toggles the token path.
- **Dataset integration:** loads the correct `pose3d` image shape + orientation
  vector shape.
- **No-heavy-import-at-top guard:** assert the train/eval/dataset path does not
  import Multi-HMR / pytorch3d at module top level (mirrors the `mllm_rerank`
  cross-import guard).
- **Smoke (DEFERRED — needs GPU + HMR weights + PAB):** extract a few images →
  render → train CMP a few steps → eval; compare R@K vs the skeleton baseline;
  report 3D coverage % and extractor throughput.

## 10. Comparison with the current pose branch

| | 2D pose skeleton (current) | `pose3d` (SMPL normal-map) |
|---|---|---|
| Nature | rendered 2D skeleton image | normal map from a 3D body |
| Sim2Real | good (no texture) | **better** (3D geometry, more view-invariant) |
| Static behavior | partial (2D joints) | **stronger** (3D torso orientation + orientation token) |
| Architecture change | — | **none** (v1 drop-in) |
| Cost | already available | + offline preprocessing of 1M images (once) |

## 11. Strengths

- **Double win on both chosen axes** (Sim2Real + behavior) from one signal.
- **Drop-in, clean A/B** against the skeleton baseline — one variable changed.
- **Challenge-compliant** (public frozen extractor, no test data in training).
- **Orientation scalars** give a strong static-frame anomaly cue (gravity-relative
  posture) that neither a 2D skeleton nor DensePose exposes.
- **Auditable coverage** via the fallback manifest (% valid 3D).

## 12. Weaknesses / risks

- **SMPL recovery fails** on small / occluded / multi-person subjects; mitigated by
  the skeleton fallback and the detector-free Multi-HMR, but coverage on real test
  must be measured.
- **Preprocessing cost** on 1M images (offline, once) — measure throughput before
  committing to the primary extractor.
- **The extractor has its own Sim2Real gap** (trained on real images, applied to
  synthetic train); a normal map is less affected than RGB but not immune.
- **v2 adds hyperparameters** (token MLP size, fusion point); v1 stays clean.
- **Latency of preprocessing** does not affect online retrieval (cache is
  precomputed), but is a one-time pipeline dependency.

## 13. Prerequisites (user-managed)

Multi-HMR (or HMR2.0 + ViTDet detector, or SMPLer-X) public weights; `pytorch3d`;
a GPU for offline extraction; the existing CMP Stage-1 training setup + PAB train
images and the current `pose/` skeleton cache (used as the fallback source); PAB
test images for inference-time extraction.
