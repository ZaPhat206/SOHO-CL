# Báo cáo ngắn: SRQ-FLY

## 1. Ý tưởng và nguồn gốc

FLY gốc dùng frozen ViT, sparse random projection và sample-dependent WTA để
tạo code \(Z_t\in\mathbb{R}^{n_t\times m}\), sau đó cập nhật

\[
G_t=G_{t-1}+Z_t^\top Z_t,\qquad
Q_t=\operatorname{expand}(Q_{t-1})+Z_t^\top Y_t,
\]

và giải classifier toàn cục

\[
W_t=(G_t+\lambda I)^{-1}Q_t.
\]

Với \(m=10{,}000\), riêng dense float32 Gram \(G_t\) đã chiếm 400 MB. SRQ-FLY
giữ nguyên backbone, projection, WTA, \(Q_t\), classifier và inference của FLY,
nhưng thay cách biểu diễn hệ Ridge:

\[
A_t=G_t+\lambda I=R_t^\top R_t.
\]

Ở task đầu, P2B tạo factor bằng Gram--Cholesky. Từ task sau, code giải mã
\(\widetilde R_{t-1}\), xếp code mới bên dưới factor cũ và cập nhật bằng blocked
QR thay vì Cholesky lại một dense Gram. Đường chéo được giữ float32; phần
strict-upper được lượng tử hóa đối xứng groupwise int8 theo block. Classifier
được tính bằng hai triangular solve, không tạo nghịch đảo tường minh. Vì
\(\widetilde A_t=\widetilde R_t^\top\widetilde R_t\), hệ giải giữ positive
definite theo cấu trúc nếu đường chéo factor dương.

Nguồn cảm hứng trực tiếp là Jingyang Li, Kuangyu Ding, Kim-Chuan Toh và Pan
Zhou, *Memory-Efficient 4-bit Preconditioned Stochastic Optimization*, ICCV
2025. Bài báo lượng tử hóa Cholesky factor của Shampoo thay vì lượng tử hóa
trực tiếp preconditioner, giữ đường chéo ở FP32 và dùng error feedback.
SRQ-FLY chỉ chuyển **mẫu thiết kế factor-space quantization** sang streaming
Ridge; đây không phải Shampoo áp dụng trực tiếp cho FLY.

Các khác biệt bắt buộc phải ghi rõ:

- bài ICCV dùng 4-bit Cholesky quantization cho optimizer Shampoo;
- code SRQ-FLY hiện dùng groupwise int8 cho sufficient state của Ridge;
- SRQ-FLY hiện không có error feedback, momentum optimizer hoặc inverse
  fourth-root;
- định lý hội tụ Shampoo của bài ICCV không chuyển sang SRQ-FLY. Code SRQ hiện
  chỉ hỗ trợ các kết luận về positive definiteness, exact square-root streaming
  khi không lượng tử hóa và perturbation của nghiệm Ridge.

## 2. Bằng chứng từ artifact ba dataset

Nguồn chính cho same-width comparison là
`srq_fly_p2b_final_confirmation.zip`, SHA-256
`14826488b8d82bc306a07e6d4f229cc389a8447150833aefc1de664961a9e85d`.
Artifact dùng implementation P2B, chọn hyperparameter chỉ trên
train-validation, sau đó chạy sáu replicate ghép cặp về class-order và
projection seed (`3031`-`3036`). Exact FLY và SRQ dùng cùng ViT-B/16 frozen,
preprocessing, \(m=10{,}000\), projection, WTA và Ridge \(\lambda\) trong mỗi
dataset.

Các số dưới đây là mean ± sample standard deviation qua sáu replicate. Chênh
lệch accuracy là SRQ trừ Exact FLY; thời gian update là tổng analytic update,
không gồm feature extraction dùng chung.

| Dataset | Final: Exact / SRQ | Δ final (pp) | AIA: Exact / SRQ | Δ AIA (pp) | State: Exact / SRQ | Giảm state | Update SRQ/Exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | 88.632±0.138 / 88.580±0.106 | -0.052 | 92.249±0.447 / 92.231±0.420 | -0.018 | 444.01 / 97.17 MB | 78.1% | 2.09× |
| CUB-200-2011 | 88.297±0.115 / 88.126±0.088 | -0.170 | 92.766±0.534 / 92.683±0.534 | -0.083 | 452.01 / 105.17 MB | 76.7% | 1.60× |
| ImageNet-R | 71.948±0.256 / 71.869±0.247 | -0.079 | 78.215±0.472 / 78.153±0.481 | -0.062 | 452.01 / 105.17 MB | 76.7% | 1.86× |

Paired 95% CI của chênh lệch AIA SRQ - Exact FLY là
`[-0.076,+0.040]` trên CIFAR-100, `[-0.149,-0.017]` trên CUB và
`[-0.094,-0.030]` trên ImageNet-R. Vì vậy kết luận đúng là SRQ bám rất sát
Exact FLY với state nhỏ hơn nhiều; không được kết luận SRQ tăng accuracy hoặc
hoàn toàn tương đương thống kê trên cả ba dataset. Inference time gần như giữ
nguyên (tỷ lệ SRQ/Exact từ 0.977 đến 1.027).

Raw-feature Ridge dùng state nhỏ hơn nhiều (5.95-7.18 MB) nhưng final accuracy
thấp hơn SRQ khoảng 1.47 pp trên CIFAR-100, 2.33 pp trên CUB và 2.73 pp trên
ImageNet-R. Điều này cho thấy SRQ nằm ở một điểm trade-off khác: giữ phần lớn
lợi ích accuracy của FLY, không nhằm đạt state tối thiểu tuyệt đối.

Artifact P2B có trạng thái `CONFIRMATION_REPORTED_WITHOUT_ACCURACY_GATE` và
thực sự đã dùng test set. ImageNet-R chỉ là **legacy processed-split**: audit
phát hiện 19 nội dung
trùng qua train/test, trong đó 18 trường hợp nằm dưới nhãn xung đột. Kết quả này
không được gọi là content-disjoint ImageNet-R held-out result.

### 2.1. FLY được giảm width tới cùng state budget

Artifact mới `srq_fly_state_matched_final.zip`, SHA-256
`a5adc883089f6108a01f33d57f0737894af843262a18a50f5309d82a54f323f9`,
kiểm tra một baseline chặt hơn: thay vì giữ FLY ở width 10,000, chọn width lớn
nhất sao cho persistent tensor bytes không vượt state của P2B. Width được suy
ra chỉ từ công thức byte, trước khi nhìn accuracy:

| Dataset | Width FLY state-matched | Ridge chọn trên train-only | State FLY / P2B | Sai lệch state |
|---|---:|---:|---:|---:|
| CIFAR-100 | 4,409 | \(10^6\) | 97,163,276 / 97,166,236 B | 0.0030% |
| CUB-200-2011 | 4,518 | \(10^5\) | 105,149,848 / 105,166,636 B | 0.0160% |
| ImageNet-R | 4,518 | \(10^6\) | 105,149,848 / 105,166,636 B | 0.0160% |

Kết quả test sáu replicate:

| Dataset | Final: FLY-matched / P2B | AIA: FLY-matched / P2B | Δ AIA P2B-FLY (pp), paired 95% CI |
|---|---:|---:|---:|
| CIFAR-100 | 87.915±0.112 / 88.580±0.106 | 91.767±0.396 / 92.231±0.420 | +0.464 [+0.340,+0.589] |
| CUB-200-2011 | 87.856±0.137 / 88.126±0.088 | 92.552±0.566 / 92.683±0.534 | +0.132 [+0.001,+0.262] |
| ImageNet-R | 70.675±0.264 / 71.869±0.247 | 77.297±0.514 / 78.153±0.481 | +0.856 [+0.698,+1.014] |

Đây là bằng chứng quan trọng nhất cho cơ chế của SRQ: tại gần như cùng state
budget, giữ width 10,000 rồi nén factor tốt hơn giảm width của FLY xuống khoảng
4,400--4,500. Nó **không** chứng minh lượng tử hóa làm tăng accuracy so với
Exact FLY cùng width 10,000; same-width result phía trên vẫn cho thấy P2B giảm
nhẹ 0.018--0.083 điểm AIA.

Train-only checkpoint `srq_state_matched_train_only_checkpoint.zip`, SHA-256
`9c42d3f51581443b642b8b79e793d44f412a73936fc8e45cf9cd7238dcb22801`,
khớp byte-for-byte với ba `selection.json` được đóng trong final ZIP.
Final ZIP có trạng thái
`STATE_MATCHED_CONFIRMATION_REPORTED_WITHOUT_ACCURACY_GATE`, `uses_test_set=true`
và `test_tuning_allowed=false`.

## 3. Ưu điểm và hạn chế so với FLY gốc

### Ưu điểm

- Giảm 76.7-78.1% persistent learner tensor bytes tại cùng expansion width
  10,000 và gần như giữ nguyên accuracy.
- Không làm yếu representation bằng cách giảm projection dimension; phần thay
  đổi nằm ở cách lưu hệ analytic.
- Hệ giải tái dựng có dạng \(R^\top R\), nên positive definite theo cấu trúc;
  an toàn hơn lượng tử hóa trực tiếp dense Gram.
- Classifier vẫn global, analytic, deterministic và không cần Task-ID.
- Exemplar-free **ở learner-state level**: checkpoint không chứa raw image,
  historical feature, WTA code hay tensor theo từng sample.
- Inference time thực nghiệm gần Exact FLY.

### Hạn chế hiện tại

- SRQ không vượt Exact FLY về accuracy trong artifact cuối; AIA giảm nhỏ nhưng
  nhất quán trên CUB và ImageNet-R.
- Update P2B chậm hơn Exact FLY 1.60-2.09 lần do giải mã factor, blocked QR,
  lượng tử hóa lại và triangular solve sau mỗi task.
- State 97-105 MB vẫn lớn hơn raw Ridge nhiều lần; đây là memory-accuracy
  trade-off, không phải phương pháp nhỏ nhất.
- Code là int8, chưa phải 4-bit như bài ICCV, và chưa dùng error feedback.
- Isolated T4 benchmark đã đo peak PyTorch CUDA allocation: P2B giảm 23.8% so
  với Exact FLY. Đây chưa phải whole-process/NVML peak memory.
- Frozen feature/WTA caches trên disk chứa dữ liệu theo sample và có thể rất
  lớn. Chúng là hạ tầng thí nghiệm, không phải learner state, và không được đóng
  gói vào checkpoint khi tuyên bố exemplar-free.
- Bằng chứng chỉ dùng một frozen ViT-B/16. Chưa có train-from-scratch,
  representation adaptation hay backbone khác.
- Sáu replicate dùng lại cùng tập test; confidence interval phản ánh biến thiên
  class-order/projection seed, không phải uncertainty do lấy mẫu dataset mới.
- State-matched final là secondary control trên test đã dùng trước đó, không
  phải một held-out benchmark mới.
- Do lỗi runner duyệt key của dictionary loader, test-feature extraction của
  state-matched ZIP đã dùng runtime compatibility adapter chỉ để chuyển
  `{task_id: DataLoader}` thành danh sách theo task ID. Adapter không đổi mẫu,
  model hoặc hyperparameter, nhưng không nằm trong source identity ban đầu;
  vì vậy ZIP hiện là recovery evidence, chưa phải artifact source-locked cuối.

## 4. Việc cần làm tiếp theo

1. **Chạy direct-quantization control đã khóa.** Notebook Priority 3 so sánh
   Exact FLY, direct INT8 không sửa, direct INT8 với Weyl-certified diagonal
   loading, FP16 square-root và SRQ P2B trên cùng CIFAR train-validation stream.
   Hiện mới có implementation và correctness gate; chưa có số thực nghiệm nên
   chưa được kết luận lợi ích đến từ square-root structure.
2. **Đóng lại provenance của state-matched control.** Rerun extraction/final
   evaluation trên commit đã sửa dictionary-loader; không thay selection,
   width, lambda, seed hoặc test-time decision.
3. **Đo whole-process peak memory.** Bổ sung NVML peak trên cùng GPU/software
   stack và giữ riêng persistent state, allocator peak và disk cache.
4. **Thử error feedback như một method mới.** Chỉ triển khai sau khi có công
   thức state và bound rõ ràng; error state phải được tính vào persistent bytes.
   So sánh no-EF/EF trên train-validation trước, không tune bằng test.
5. **Thử true int4 có packing thực.** Nếu chỉ lưu int4 trong tensor int8 thì
   không được tuyên bố giảm byte. Cần pack hai giá trị mỗi byte, kiểm tra kernel,
   tốc độ giải mã và accuracy-memory Pareto.
6. **Củng cố lý thuyết.** Bổ sung bound tích lũy lỗi factor qua task, bound
   perturbation nghiệm/logit và điều kiện margin bảo toàn prediction. Không tái
   sử dụng định lý hội tụ Shampoo ngoài phạm vi của nó.
7. **Hoàn thiện bằng chứng paper.** Giữ CIFAR và CUB như kết quả đã tiêu thụ;
   thay ImageNet-R legacy bằng split sạch hoặc thêm một dataset chưa mở test.
   Báo Exact FLY là baseline chính, raw Ridge là lower-memory baseline và SOHO
   replay ở bảng riêng với toàn bộ sample-level state bytes.

## 5. Kết luận ngắn

SRQ-FLY hiện là một hướng **khả thi và có tín hiệu paper rõ về
memory-accuracy trade-off**: giảm khoảng bốn phần năm persistent state của
Exact FLY trong khi chỉ mất 0.02-0.08 pp AIA, đồng thời hơn FLY giảm-width tại
cùng state budget từ 0.132 đến 0.856 pp AIA. Tuy nhiên, nó chưa phải phương pháp
tăng accuracy so với FLY cùng width và update vẫn chậm hơn 1.60-2.09 lần. Định
vị trung thực nhất hiện tại là “structure-preserving compression that preserves
representation width”, không phải “better FLY in every metric”.
