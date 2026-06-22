# LHP Retriever Module — Design Spec

**Date:** 2026-06-22
**Scope:** Train-only LHP retriever (BeiT-3 + local/global crop), as a self-contained module in the Sim2Real-Track4 (CMP) project.
**Out of scope:** 2-stage eval / mAP, feature-selection wiring, UIT reranker, iterative ensemble. (Future steps.)

---

## 1. Mục tiêu & bối cảnh

Triển khai module **LHP** (Local-global Hybrid Perspective) theo paper *"Hybrid, Unified and Iterative"* (WWW'25), dùng làm **stage-1 retriever** cho bài Text-based Person Anomaly Search (ECCV 2026 Track 4).

- LHP = augmentation chọn view local/global lúc train một dual-encoder **BeiT-3**, tối ưu bằng contrastive (ITC).
- Module này **train riêng**, xuất ra checkpoint retriever. Bước sau (ngoài scope) sẽ dùng nó lọc top-k rồi đưa qua reranker (CMP cross-encoder).
- **Không tích hợp vào CMP `Search.py`** — BeiT-3 khác tokenizer (SentencePiece/XLM-R vs BERT), khác API, dùng ClipLoss (1 loss) thay vì 3-loss của CMP.

### Ràng buộc challenge (BẮT BUỘC)
- **Không hard sampling** (IHNM bị cấm). ClipLoss chỉ dùng in-batch negatives → tự nhiên tuân thủ.
- Reproducibility: winner phải nộp code verify → **vendor BeiT-3 vào repo**, không trỏ path ngoài.

---

## 2. Kiến trúc & layout

Package tự chứa `lhp/`, độc lập với code CMP; chỉ tái dùng I/O thuần của CMP.

```
lhp/
├── beit3/              # VENDOR: copy file BeiT-3 cần từ /home/pc1175/Code/open-sources/unilm/beit3
│   ├── modeling_finetune.py   (BEiT3ForRetrieval, register_model)
│   ├── modeling_utils.py      (BEiT3Wrapper, _get_base_config)
│   ├── utils.py               (ClipLoss, get_rank/world_size — hoặc trích phần cần)
│   └── (randaug.py nếu cần)
├── model.py            # LHPRetriever: wrap beit3_base_patch16_384_retrieval + load ckpt
├── transform.py        # LHPTransform: switch ngẫu nhiên local(RandomResizedCrop)/global(resize)
├── dataset.py          # LHPDataset: đọc annotation JSON → (image, caption, image_path)
├── train.py            # vòng train DDP: spm tokenize → ClipLoss → AdamW → cosine → save ckpt
├── infer.py            # API: encode_image/encode_text → similarity (sanity + cho rerank sau)
└── config.yaml

# Tái dùng từ CMP (import trực tiếp, KHÔNG copy — đã decoupled):
dataset/utils.py  →  pre_caption(), read_json_to_list()
dataset/eda.py    →  eda()  (chỉ khi cfg.eda = true)
```

**Nguyên tắc cô lập:** `lhp/` KHÔNG import từ `Search.py`, `models/cmp.py`, `models/model_search.py`. Mỗi file một trách nhiệm để test độc lập.

---

## 3. Components

### 3.1 `transform.py` — LHPTransform
- Mỗi sample tung đồng xu theo `local_prob` (mặc định 0.5; paper Normal(0.5,1/6) thực chất Bernoulli):
  - **local**: `RandomResizedCrop(resolution, scale=crop_scale)` — mặc định `scale=(0.5, 0.8)`.
  - **global**: `Resize((resolution, resolution))`.
- Sau đó dùng chung: `ToTensor` + `Normalize` theo **đúng mean/std của BeiT-3** (lấy từ `beit3/datasets.py` lúc implement, KHÔNG tự chế).
- Trả về một image tensor (local *hoặc* global) + flag view để debug.
- `resolution`, `crop_scale`, `local_prob` đều từ config.

### 3.2 `dataset.py` — LHPDataset(Dataset)
- Dùng `read_json_to_list()` đọc các file trong `train_file`; ảnh tại `image_root` (layout PAB như config CMP).
- `__getitem__` → `(image_tensor, caption_str, image_path)`.
  - caption = `pre_caption(cap, max_words, is_eda=cfg.eda, eda_p=cfg.eda_p)`.
  - **KHÔNG** trả pose, hard_i/hard_c, idx.
- Tokenize KHÔNG làm ở dataset (trả raw string); train loop tokenize bằng SentencePiece.

### 3.3 `model.py` — LHPRetriever
- Dựng `beit3_base_patch16_384_retrieval`, load checkpoint COCO-retrieval (`strict=False`, **log missing/unexpected keys**).
- `BEiT3ForRetrieval.forward(image, text_ids, padding_mask)` **tự tính ClipLoss** (in-batch negatives + logit_scale) → trả `(loss, vision_cls, language_cls)`.
- **Contrastive chuẩn theo paper (eq.1-3) = ClipLoss diagonal.** Bỏ `idx`-grouping của CMP: trên PAB mỗi `image_id` là duy nhất theo dòng nên idx ≈ diagonal, không sinh false-negative; các ảnh cùng `source_id` là ảnh/caption khác nhau → đúng là negative.

### 3.4 `train.py` — DDP training loop
```
spm = XLMRobertaTokenizer(cfg.spm_model)
for epoch in range(cfg.epochs):
    sampler.set_epoch(epoch)                       # shuffle đúng giữa các rank
    for (image, captions, _) in loader:
        text_ids, padding_mask = spm(captions, max_len=cfg.max_tokens)
        loss, _, _ = model(image, text_ids, padding_mask)   # ClipLoss
        loss.backward() (AMP bf16) → AdamW → cosine.step()
    save checkpoint (mỗi epoch + cuối)
```

### 3.5 `infer.py` — inference API
- `encode_image(images) -> feats`, `encode_text(texts) -> feats` (gọi model với `only_infer=True`).
- `similarity(img_feats, txt_feats) -> [N_txt, N_img]`, `topk(...)`.
- Mục đích: sanity-check + interface cho bước rerank sau cắm vào.

---

## 4. Data flow (train)

```
JSON annotation ──read_json_to_list──▶ list[ann]
ann ──pre_caption(+eda?)──▶ caption_str
ann.image ──PIL──▶ LHPTransform ──▶ image_tensor (local|global, 384)
        │
DataLoader + DistributedSampler (DDP)
        │
train loop: spm tokenize → model(image, text_ids, padding_mask) → ClipLoss → backward/step
        │
checkpoint → output_dir/
```

---

## 5. Config (`lhp/config.yaml`)

```yaml
# data (theo layout PAB của CMP)
image_root: 'data/PAB/'
train_file: [ 'data/PAB/annotation/train/attr_0.json', ... ]   # như configs/cmp.yaml

# LHP transform
resolution: 384
crop_scale: [0.5, 0.8]
local_prob: 0.5

# text aug (tùy chọn, off theo paper)
eda: false
eda_p: 0.5
max_tokens: 64                # theo BeiT-3 retrieval

# training (theo paper mục 3.1)
batch_size: 184               # effective; xem mục 7 (VRAM)
lr: 1.0e-5
epochs: 3
optimizer: adamW
scheduler: cosine

# checkpoints (user-managed, xem mục 6)
beit3_ckpt: 'checkpoint/beit3_base_patch16_384_coco_retrieval.pth'
spm_model:  'checkpoint/beit3.spm'
output_dir: 'output/lhp'
```

---

## 6. Setup — prerequisites (USER-MANAGED)

Người dùng tự chuẩn bị (đã cài torchscale + sentencepiece; timm 0.4.12 = đúng bản BeiT-3, không cần vá):

```sh
# tokenizer
wget https://github.com/addf400/files/releases/download/beit3/beit3.spm -P checkpoint/
# checkpoint base/384 retrieval-finetuned (khuyến nghị COCO)
wget https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth -P checkpoint/
```
- Config mặc định trỏ tới 2 file trên trong `checkpoint/`.
- Alt: `beit3_base_patch16_384_f30k_retrieval.pth` (Flickr30k) nếu muốn.

---

## 7. Batch / VRAM + DDP correctness (QUAN TRỌNG)

- **Contrastive cần batch lớn**: ClipLoss allgather feature toàn world → số negative = **tổng batch toàn cục**. Batch nhỏ = ít negative = yếu hơn → không giảm tùy tiện.
- Batch 184 @ base/384 trên 4×3090 24G (~46/GPU) → **dễ OOM**. Khi test thật: giảm per-GPU batch cho vừa VRAM + **gradient accumulation** giữ effective ~184, hoặc chấp nhận batch nhỏ hơn (yếu hơn paper). *(User sẽ cân nhắc lúc test.)*
- **DDP — 2 điểm phải đảm bảo loss đúng:**
  1. **Gather feature phải differentiable** ở phần local rank (custom autograd kiểu `AllGather` trong CMP `cmp.py`). Phải **verify `ClipLoss` của BeiT-3** (`beit3/utils.py`) làm đúng việc này + `rank`/`world_size` set chuẩn. Đây là **rủi ro cao nhất** của module.
  2. **DistributedSampler + `set_epoch(epoch)`** mỗi epoch.

---

## 8. Error handling / robustness

- Ảnh lỗi (webp/jpg hỏng) → try/except trong `__getitem__`, resample index khác.
- `pre_caption` ném `ValueError` khi caption rỗng → bắt và skip.
- Load checkpoint `strict=False` + log rõ missing/unexpected keys (wrap `BEiT3ForRetrieval` có thể lệch tên head).

---

## 9. Testing / verification (scope train-only)

1. **transform**: output `(3,384,384)`; cả 2 nhánh local/global reachable; tỉ lệ local ≈ `local_prob`.
2. **dataset**: trả `(tensor, str, path)`; spm tokenize ok; không có field pose/hard.
3. **smoke model**: build + load ckpt (missing keys hợp lý); forward 1 batch nhỏ → `loss` scalar; `loss.backward()` chạy.
4. **DDP smoke (2 proc)**: vài iter, loss giảm/không treo; xác nhận gather differentiable (loss khác bản single-process diagonal → bằng chứng allgather hoạt động). ← kiểm tra rủi ro mục 7.1.
5. **infer.py**: encode vài ảnh+text → similarity đúng shape; top-k định tính hợp lý trên mẫu nhỏ.

---

## 10. Future (ngoài scope spec này)
- Feature-selection: dùng LHP retriever lọc top-k.
- Rerank: đưa top-k qua cross-encoder CMP (thay vai trò UIT).
- (Có thể) iterative ensemble.
