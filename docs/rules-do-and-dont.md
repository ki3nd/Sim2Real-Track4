# ECCV 2026 — Track 4 (Text-based Person Anomaly Search / Sim2Real)
## Tổng hợp: ĐƯỢC PHÉP vs KHÔNG ĐƯỢC PHÉP

> Tóm tắt từ `challenge-track4.md`, dùng để tra cứu nhanh khi làm bài.
> Hai nhóm rule phải thỏa **đồng thời**: (A) *Additional Datasets* và (B) *Important note về test*.

---

## 1. Dữ liệu được cung cấp

| Tập | Nội dung | Ghi chú |
|---|---|---|
| **Train** | 1,013,605 ảnh **synthetic** (normal + abnormal) + text (appearance/action/scene) + nhãn normal/abnormal + scene | Cùng tập train với ICCV25 |
| **Test (name-masked)** | 1,978 query text (normal:abnormal = 1:1); gallery = 1,978 ảnh GT real + **34,795 distractor** | Ảnh **real** (đề không nêu nguồn); **giữ kín** |

- **Metric leaderboard: mAP** (trung bình diện tích dưới đường precision-recall trên mọi query).
- **Submission**: `answer.txt`, mỗi dòng = **top-10 tên ảnh** cho 1 query (Rank-1 → Rank-10), thứ tự theo `query_index.txt`.

---

## 2. ✅ ĐƯỢC PHÉP

- ✅ Dùng **tập train synthetic** được cung cấp để train tự do.
- ✅ Dùng **thêm dataset PUBLIC bất kỳ** cho **train / validation / test** (rule liệt kê rõ cả 3).
  - VD train/pretrain: CUHK-PEDES, ICFG-PEDES, RSTPReid, MALS, dữ liệu dựng từ UCF-Crime (public)...
- ✅ Tự **cắt validation từ tập train synthetic** để chọn model / tune / ablation.
  - Nên split **theo identity / source-caption** (không split ngẫu nhiên theo ảnh) → tránh leakage.
  - Nên dựng đúng format test: query 1:1 normal:abnormal + chèn **distractor** mô phỏng tỷ lệ ~1:17.
- ✅ Dùng **dataset public KHÁC phân phối test** làm proxy eval real-domain.
  - Tốt nhất cho khía cạnh anomaly: **UCC** (dựng từ UCF-Crime public + caption).
- ✅ **Augmentation** mạnh (giảm sáng, blur, noise, weather...) để thu hẹp sim2real gap.
- ✅ Inference trên official test **đúng 1 lần** để tạo `answer.txt` và nộp.

---

## 3. ❌ KHÔNG ĐƯỢC PHÉP

- ❌ Dùng **official test set (query/gallery)** để train — dưới mọi hình thức.
- ❌ Dùng **official test làm validation**, **kể cả khi KHÔNG có nhãn**.
- ❌ Dùng **output trên test** cho: model selection, threshold tuning, ensemble selection, pseudo-labeling, post-processing. → Test chỉ để **inference cuối + chấm leaderboard**.
- ❌ Dùng **test data VÀ PHÂN PHỐI của nó** trong quá trình train.
  - ⚠️ Hệ quả: **PAB-test gốc (ICCV25)** tuy *public* nhưng **rất có thể cùng phân phối** với test challenge → **KHÔNG dùng** (kể cả để validation). Rule cấm phân phối test thắng rule cho phép dataset public.
- ❌ Dùng **dataset non-public** (tự thu thập riêng / nội bộ / không công khai) ở bất kỳ khâu nào (train/val/test).

---

## 4. ⚠️ Điểm tinh tế — quy tắc giao nhau

Khi một dataset vừa *public* vừa *trùng phân phối test*, **rule cấm phân phối test (B) thắng** rule cho phép public (A).

| Dataset | Public? | Cùng phân phối test? | Train? | Eval/Val? |
|---|---|---|---|---|
| Train synthetic (cung cấp) | ✅ | ❌ | ✅ | ✅ (held-out) |
| Official test (name-masked) | — | ✅ | ❌ | ❌ |
| PAB-test gốc (ICCV25) | ✅ | ✅ | ❌ | ❌ |
| UCC (UCF-Crime-based) | ✅ | ❌ | ✅ | ✅ |
| CUHK-PEDES / ICFG / RSTPReid | ✅ | ❌ | ✅ | ⚠️ chỉ sanity (thiếu anomaly) |
| Dataset non-public bất kỳ | ❌ | — | ❌ | ❌ |

---

## 5. Verify sau deadline

- Winner & runner-up phải **nộp code train + test** để ban tổ chức verify.
- Mục tiêu verify: **không dùng non-public data**, đảm bảo "algorithm-driven, not human-performed".
- → Mọi thứ phải **tái lập được** và **không lách** bằng dữ liệu trùng phân phối test.

---

## 6. Quy trình đề xuất (đúng rule)

1. **Phát triển/tune/ablation** → trên **synthetic held-out** (identity-disjoint + distractor).
2. **Xác nhận sim2real generalization** → trên **UCC public** (proxy real-domain hợp lệ).
3. **Chọn model cuối** → dựa trên (1) + (2), **không bao giờ peek official test**.
4. **Nộp** → inference official test 1 lần → `answer.txt`.
