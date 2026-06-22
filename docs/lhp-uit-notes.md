# Ghi chú: LHP & UIT (paper Hybrid, WWW'25) — cho ý tưởng triển khai

## Kiến trúc: 2 MODEL RIÊNG, train tách biệt
- **LHP** = BEiT-3 (baseline) + cơ chế crop local/global. Fine-tune 3 epoch, ảnh **384×384**.
- **UIT** = Swin-B + BERT + cross-encoder + decoder. Train 22 epoch, ảnh **224×224**.
- Khác backbone / epoch / resolution → độc lập hoàn toàn.

## Pipeline inference (2 tầng)
```
BEiT-3 (LHP)  →  TẦNG 1: cosine sim → chọn top-k candidate   (selector/retrieval)
top-k image   →  encode LẠI bằng Swin của UIT
UIT cross-enc →  TẦNG 2: ITM head → điểm matching CUỐI → rank  (reranker = ra kết quả)
```
- **UIT cho kết quả cuối.** BEiT-3 chỉ **lọc top-k** đứng trước, KHÔNG phải reranker.
- LHP "tốt hơn UIT" = lọc top-k sạch hơn; nếu ảnh đúng rớt khỏi top-k thì UIT không cứu được.
- Map sang `eval.py` của CMP: `evaluation_itc` (cosine, top-128) ↔ LHP; `evaluation_itm` (cross-encoder+ITM) ↔ UIT.

## LHP — bản chất
- KHÔNG phải module có trọng số. Là **augmentation chọn view** lúc train: mỗi sample random local-crop hoặc global-full (paper dùng Normal(0.5,1/6) → thực chất Bernoulli 50/50).
- Local = zoom chi tiết tư thế/hành động; Global = giữ context/scene.
- Lợi ích đến từ **contrastive objective + encoder**, không từ crop. Crop chỉ cấp view đa-scale.
- Gain một mình **nhỏ** (~+0.15% R@1 @0.1M). Cú nhảy lớn ở UIT(FS) + iterative ensemble.

## Crop region-of-interest — các vấn đề (data thực tế)
- Pose là **ảnh render RGB nền đen, KHÔNG có toạ độ keypoint** → bbox phải lấy bằng threshold pixel khác đen (sạch, rẻ).
- **Multi-person**: nhiều ảnh có 2-3 skeleton → min/max toàn bộ = box trùm cả đám → mất ý nghĩa "zoom".
  - Xử lý: connected-components tách cụm → chọn **cụm lớn nhất** (data person-centric nên thường trúng target). Skeleton dính nhau → fallback global.
- Không có coords → ý tưởng "crop theo chi đang hoạt động" khó làm sạch → thực tế chỉ làm được **bbox-bao-người**.
- Vì chỉ bbox-bao-người, lợi thế pose so với YOLO **chỉ còn**: rẻ + không cần model phụ + không domain-gap (ảnh PAB là synthetic).

## Đánh đổi quan trọng: crop làm LỆCH ảnh↔text
- Caption mô tả appearance+action+**scene**; crop bỏ background → đưa **nhiễu nhãn** vào contrastive.
- Khác DINO/SwAV (match image↔image, không lệch); LHP match image↔TEXT nên **dính** lỗi này. Paper không bàn.
- Sống được vì: (1) stochastic — global view giữ background; (2) contrastive chỉ cần đúng **tương đối**, phần discriminative (người+hành động) sống sót qua crop; background thường là phần dùng-chung ít phân biệt.
- **Giảm nhẹ khi triển khai**: crop **margin rộng** (giữ một phần scene) + **hạ trọng số loss local** (kiểu CMP để 0.8 cho nhánh EDA).

## Baseline
- Bài Hybrid: baseline = **BEiT-3** (fine-tuned, 384×384). 0.1M: R@1 85.24. +LHP@1M: 87.11. Full (LHP+UIT+IE): 89.23.
- Lưu ý: chọn backbone BEiT-3 mạnh đã là phần lớn lợi thế (CMP tái lập trong bài chỉ 72.80 R@1 @0.1M).
- Số liệu trên đo ở test PAB **sạch (không distractor)** → leaderboard challenge có 34,795 distractor sẽ **thấp hơn nhiều**.
