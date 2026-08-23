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

Sau mỗi task, code hiện tại giải mã \(\widetilde R_{t-1}\), cộng thống kê mới
\(Z_t^\top Z_t\), Cholesky lại và lưu factor tam giác trên. Đường chéo được
giữ float32; phần strict-upper được lượng tử hóa đối xứng groupwise int8 theo
block. Classifier được tính bằng hai triangular solve, không tạo nghịch đảo
tường minh. Vì \(\widetilde A_t=\widetilde R_t^\top\widetilde R_t\), hệ giải
giữ positive definite theo cấu trúc nếu đường chéo factor dương.

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

Nguồn số liệu là `srq_fly_selfcontained_three_dataset_results.zip`, SHA-256
`e4b630781ff6f69deaecb63dda9926d256cd6b654ef4b51a682bf3ef94e6490b`.
Artifact khóa implementation tại commit
`0c9b2b67c6a5fcc41f89ff72fef3b8e6931edced`, chọn hyperparameter chỉ trên
train-validation, sau đó chạy sáu replicate test độc lập về class-order và
projection seed (`3031`-`3036`). Exact FLY và SRQ dùng cùng ViT-B/16 frozen,
preprocessing, \(m=10{,}000\), projection, WTA và Ridge \(\lambda\) trong mỗi
dataset.

Các số dưới đây là mean ± sample standard deviation qua sáu replicate. Chênh
lệch accuracy là SRQ trừ Exact FLY; thời gian update là tổng analytic update,
không gồm feature extraction dùng chung.

| Dataset | Final: Exact / SRQ | Δ final (pp) | AIA: Exact / SRQ | Δ AIA (pp) | State: Exact / SRQ | Giảm state | Update SRQ/Exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | 88.633±0.137 / 88.587±0.079 | -0.047 | 92.249±0.447 / 92.232±0.416 | -0.017 | 444.01 / 97.17 MB | 78.1% | 3.10× |
| CUB-200-2011 | 88.297±0.115 / 88.089±0.128 | -0.207 | 92.766±0.534 / 92.688±0.540 | -0.079 | 452.01 / 105.17 MB | 76.7% | 7.02× |
| ImageNet-R | 71.948±0.256 / 71.848±0.262 | -0.100 | 78.215±0.472 / 78.144±0.485 | -0.071 | 452.01 / 105.17 MB | 76.7% | 5.96× |

Paired 95% CI của chênh lệch AIA SRQ - Exact FLY là
`[-0.078,+0.044]` trên CIFAR-100, `[-0.148,-0.009]` trên CUB và
`[-0.104,-0.038]` trên ImageNet-R. Vì vậy kết luận đúng là SRQ bám rất sát
Exact FLY với state nhỏ hơn nhiều; không được kết luận SRQ tăng accuracy hoặc
hoàn toàn tương đương thống kê trên cả ba dataset. Inference time gần như giữ
nguyên (tỷ lệ SRQ/Exact từ 0.977 đến 1.027).

Raw-feature Ridge dùng state nhỏ hơn nhiều (5.95-7.18 MB) nhưng final accuracy
thấp hơn SRQ khoảng 1.48 pp trên CIFAR-100, 2.29 pp trên CUB và 2.71 pp trên
ImageNet-R. Điều này cho thấy SRQ nằm ở một điểm trade-off khác: giữ phần lớn
lợi ích accuracy của FLY, không nhằm đạt state tối thiểu tuyệt đối.

Artifact có trạng thái `REPORTED_WITHOUT_ACCURACY_GATE` và thực sự đã dùng test
set. ImageNet-R chỉ là **legacy processed-split**: audit phát hiện 19 nội dung
trùng qua train/test, trong đó 18 trường hợp nằm dưới nhãn xung đột. Kết quả này
không được gọi là content-disjoint ImageNet-R held-out result.

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
- Update hiện chậm hơn 3.1-7.0 lần do giải mã factor, tạo hệ dense, Cholesky và
  lượng tử hóa lại sau mỗi task.
- State 97-105 MB vẫn lớn hơn raw Ridge nhiều lần; đây là memory-accuracy
  trade-off, không phải phương pháp nhỏ nhất.
- Code là int8, chưa phải 4-bit như bài ICCV, và chưa dùng error feedback.
- Artifact chưa báo peak runtime GPU memory. Persistent learner bytes không
  được thay thế cho highest peak runtime memory.
- Frozen feature/WTA caches trên disk chứa dữ liệu theo sample và có thể rất
  lớn. Chúng là hạ tầng thí nghiệm, không phải learner state, và không được đóng
  gói vào checkpoint khi tuyên bố exemplar-free.
- Bằng chứng chỉ dùng một frozen ViT-B/16. Chưa có train-from-scratch,
  representation adaptation hay backbone khác.
- Sáu replicate dùng lại cùng tập test; confidence interval phản ánh biến thiên
  class-order/projection seed, không phải uncertainty do lấy mẫu dataset mới.
- Phần `methods/srq_fly_optimized/` đang phát triển chưa được dùng để tạo ZIP
  này; không được gán kết quả accuracy hoặc runtime trong bảng cho code tối ưu.

## 4. Việc cần làm tiếp theo

1. **Khóa tối ưu update mà không đổi predictor.** Hoàn thiện namespace tối ưu,
   chứng minh checkpoint compatibility và output equivalence với implementation
   đã khóa; chạy benchmark CUDA ở (m=10{,}000). Chỉ dùng train/synthetic stream
   cho timing, không cần mở test set lại.
2. **Đo đúng chi phí hệ thống.** Trên cùng GPU và software stack, báo peak
   allocated/reserved GPU memory, update time theo từng stage, inference time,
   serialized checkpoint bytes, persistent tensor bytes và disk-cache bytes
   thành các đại lượng riêng biệt.
3. **Ablation nguồn lợi ích.** So sánh Exact FLY-10000, SRQ-int8-10000, direct
   int8 Gram, float16 square-root, state-matched lower-dimensional Exact FLY và
   raw Ridge dưới cùng protocol.
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
Exact FLY trong khi chỉ mất 0.02-0.08 pp AIA trung bình. Tuy nhiên, nó chưa phải
một phương pháp tăng accuracy và chưa có bằng chứng giảm peak runtime memory;
nút thắt cần giải quyết ngay là update time. Định vị trung thực nhất hiện tại là
“structure-preserving compression of analytic continual-learning state”, không
phải “better FLY in every metric”.
