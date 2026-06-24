# Hướng đề xuất: VLM rerank để nâng R@1

## Bối cảnh
- Stage-1 (BEiT-3 + LHP) cho R@1=83.87, R@5=99.24, R@10=99.75 (đo trên test PAB sạch, gallery = đúng các ảnh đáp án, không distractor).
- R@5/R@10 gần như tuyệt đối → ảnh đúng hầu như luôn nằm trong top-5. Headroom R@1 còn ~15 điểm, đáp án đã sẵn trong top-k → **rerank top-k là chỗ phát huy tối đa**.
- Ý tưởng: sau khi stage-1 select top-k, dùng VLM (Qwen3-VL-8B) rerank theo kiểu RankGPT (sliding window). Server 4×4090 24GB, host bằng vLLM/SGLang.

## Nguyên tắc xuyên suốt
**Đo trước khi thêm độ phức tạp.** Không build sliding window + thinking + tensor-parallel ngay từ đầu.

---

## Phase 0 — Dump candidates ra file JSON
Chạy stage-1 **một lần**, xuất file JSON gồm `text` + top-k `image_id` (kèm score) cho mỗi query. Rerank đọc file tĩnh này, không phải chạy lại model nặng.
- Tận dụng code BEiT-3+LHP đã có để sinh ranking, chỉ thêm phần trích top-k + ghi JSON.
- Tính ngay **recall ceiling**: với k=5/10/20, bao nhiêu % query có ảnh đúng trong top-k → đây là trần R@1 sau rerank, dùng để chọn k.

## Phase 1 — Host VLM (đơn giản hóa hạ tầng)
- Qwen3-VL-8B (bf16 ≈ 16GB) **fit gọn 1 card 4090** → khởi đầu 1 instance vLLM, OpenAI-compatible API; sau nhân 4 instance data-parallel để tăng tốc.
- Dùng bản **Instruct trước** (nhanh) để validate pipeline; so Thinking sau.
- Chưa cần tensor-parallel weights; chỉ cân nhắc khi KV cache (image tokens) tràn VRAM.

## Phase 2 — Reranker MVP: listwise top-5, một cửa sổ
- Prompt: caption + 5 ảnh nhãn `[1]..[5]`, yêu cầu VLM xuất permutation kiểu RankGPT (`[3] > [1] > ...`).
- Parse → reorder → tính lại R@1/R@5/R@10 bằng đúng `eval.mAP` của CMP (tái dùng).
- **Dừng lại đo.** Đây là 80% giá trị với 20% công sức. Top-5 nhét gọn 1 lần gọi, chưa cần sliding window.

## Phase 3 — Chống nhiễu (chỉ thêm khi MVP có tín hiệu tốt)
- **Position bias:** chạy 2 lần đảo/shuffle thứ tự ảnh, gộp kết quả (rank averaging / Borda).
- **Ablation pointwise:** VLM chấm match-score từng ảnh 0–10 độc lập, so với listwise.
- Fallback: parse-fail / VLM từ chối / sai format → giữ nguyên thứ tự stage-1.

## Phase 4 — Mở rộng k + sliding window (chỉ khi cần)
- Khi sang môi trường có distractor (leaderboard có 34,795 distractor), top-5 ceiling sẽ tụt → cần k lớn hơn (20/30).
- Lúc này mới port RankGPT sliding window (`window_size`, `step_size`, bubble từ dưới lên), mỗi window multi-image; cân nhắc Thinking + nhiều instance để chịu tải.

---
