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

Các ký hiệu trong các bảng accuracy:

- **AIA (Average Incremental Accuracy)** là trung bình accuracy sau từng bước
  học liên tục. Nếu \(a_t\) là accuracy trung bình trên tất cả lớp đã thấy sau
  task \(t\), thì
  \[
  \mathrm{AIA}=\frac{1}{T}\sum_{t=1}^{T}a_t.
  \]
  AIA phản ánh chất lượng của toàn bộ quá trình học, không chỉ trạng thái cuối.
- **\(A_{\mathrm{final}}\)** là accuracy trung bình trên tất cả lớp sau task
  cuối, tức \(a_T\). Đây là kết quả cuối cùng sau khi mô hình đã thấy toàn bộ
  class stream.
- **Update SRQ/Exact** là
  \(\text{tổng thời gian analytic update của SRQ}/\text{tổng thời gian analytic
  update của Exact FLY}\). Tỷ lệ lớn hơn 1 nghĩa là SRQ cập nhật chậm hơn. Ví
  dụ `2.09×` nghĩa là phần update của SRQ mất khoảng 2.09 lần thời gian của
  Exact FLY; con số này không bao gồm feature extraction dùng chung.

| Dataset | \(A_{\mathrm{final}}\): Exact / SRQ | Δ \(A_{\mathrm{final}}\) (pp) | AIA: Exact / SRQ | Δ AIA (pp) | State: Exact / SRQ | Giảm state | Update SRQ/Exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | 88.632±0.138 / 88.580±0.106 | -0.052 | 92.249±0.447 / 92.231±0.420 | -0.018 | 444.01 / 97.17 MB | 78.1% | 2.09× |
| CUB-200-2011 | 88.297±0.115 / 88.126±0.088 | -0.170 | 92.766±0.534 / 92.683±0.534 | -0.083 | 452.01 / 105.17 MB | 76.7% | 1.60× |
| ImageNet-R | 71.948±0.256 / 71.869±0.247 | -0.079 | 78.215±0.472 / 78.153±0.481 | -0.062 | 452.01 / 105.17 MB | 76.7% | 1.86× |

Kết luận đúng là SRQ bám rất sát Exact FLY với state nhỏ hơn nhiều; không được
kết luận SRQ tăng accuracy. Inference time gần như giữ nguyên (tỷ lệ SRQ/Exact
từ 0.977 đến 1.027).

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

| Dataset | \(A_{\mathrm{final}}\): FLY-matched / P2B | AIA: FLY-matched / P2B | Δ AIA P2B-FLY (pp) |
|---|---:|---:|---:|
| CIFAR-100 | 87.915±0.112 / 88.580±0.106 | 91.767±0.396 / 92.231±0.420 | +0.464 |
| CUB-200-2011 | 87.856±0.137 / 88.126±0.088 | 92.552±0.566 / 92.683±0.534 | +0.132 |
| ImageNet-R | 70.675±0.264 / 71.869±0.247 | 77.297±0.514 / 78.153±0.481 | +0.856 |

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

### 2.2. Vì sao phải nén trong không gian square-root?

Artifact train-only `srq_fly_priority3_direct_control_train_only.zip`, SHA-256
`9c5f8c9c0d945393cea48d23204ed585e42e656359bbb12386691f4ba452988e`,
đã chạy đủ direct-quantization control trên cùng một CIFAR development stream:

Năm phương pháp được đặt cạnh nhau để tách riêng hai câu hỏi: lợi ích có đến
từ việc dùng ít bit hay từ việc biểu diễn hệ Ridge trong không gian
square-root, và mức giảm state có phải đánh đổi bằng mất accuracy hay không.

- **Exact FLY-10000** là mốc tham chiếu không nén. Nó giữ dense Gram
  \(G\) ở FP32 với width 10,000 và giải hệ Ridge gốc. Phương pháp này cho biết
  accuracy tham chiếu cần cố giữ và chi phí state trước khi nén.
- **Direct INT8 Gram, không sửa** lượng tử hóa trực tiếp từng phần tử của
  \(G\) sang INT8 rồi giải hệ. Control này kiểm tra phương án đơn giản nhất:
  chỉ giảm bit mà không khai thác cấu trúc. Sai số lượng tử có thể làm ma trận
  đối xứng không còn dương xác định, nên Cholesky có thể thất bại.
- **Direct INT8 Gram + Weyl repair** cũng lượng tử hóa trực tiếp \(G\), nhưng
  cộng một diagonal load được tính từ biên trị riêng để ép hệ trở lại SPD.
  Control này trả lời liệu lỗi của direct INT8 chỉ là lỗi số có thể sửa rẻ hay
  không. Repair không dùng label hoặc validation accuracy.
- **FP16 square-root** lưu Cholesky factor \(R\) ở FP16 thay vì lưu Gram.
  Nó giữ cấu trúc \(R^\top R\) nhưng nén ít hơn SRQ, giúp phân biệt lợi ích của
  factorization với lợi ích riêng của INT8.
- **SRQ mixed INT8/FP32** là phương pháp đề xuất: giữ đường chéo của \(R\) ở
  FP32 và lượng tử hóa groupwise INT8 phần strict-upper. Khi giải mã, hệ luôn
  được dựng dưới dạng \(\widehat R^\top\widehat R\), nên giữ SPD theo cấu trúc
  nếu đường chéo dương.

| Phương pháp | Validation AIA | \(A_{\mathrm{final}}\) | Persistent state | Kết quả |
|---|---:|---:|---:|---|
| Exact FLY-10000 | 92.257 | 88.140 | 444,006,540 B | Hoàn thành |
| Direct INT8 Gram, không sửa | -- | -- | -- | Hệ mất positive definiteness ngay task 1 |
| Direct INT8 Gram + Weyl repair | 87.342 | 80.960 | 97,166,244 B | Hoàn thành, ổn định số |
| FP16 square-root | 92.260 | 88.130 | 144,036,540 B | Hoàn thành |
| SRQ mixed INT8/FP32 | 92.267 | 88.050 | 97,166,228 B | Hoàn thành |

Direct Gram đã sửa và SRQ gần như cùng state (chỉ lệch 16 byte), nhưng SRQ hơn
4.925 pp validation AIA. Weyl repair không dùng label/accuracy và không retry,
nhưng diagonal load tăng từ 112.95 lên 5,148.32 lần base Ridge qua 10 task;
regularization quá lớn làm mất tín hiệu. Kết quả này hỗ trợ cơ chế cốt lõi:
giữ hệ dưới dạng \(\widehat R^\top\widehat R\) có lợi hơn chỉ lượng tử hóa từng
phần tử Gram rồi sửa SPD.

Vì vậy, bảng này không nhằm chứng minh SRQ luôn tốt hơn mọi kỹ thuật INT8. Nó
cho thấy trong cùng implementation và cùng stream: (i) direct INT8 ngây thơ
không ổn định; (ii) sửa SPD bằng diagonal loading có thể làm hệ quá
regularized; và (iii) nén factor vừa đạt state gần direct INT8 đã sửa, vừa giữ
accuracy gần Exact FLY. FP16 square-root là cầu nối quan trọng để chỉ ra rằng
cấu trúc factor đã hữu ích trước cả khi giảm xuống 8 bit.

Đây là ablation train-only một development seed. Nó loại được hai direct-INT8
control cụ thể đã kiểm tra, không chứng minh mọi direct quantizer hoặc mọi SPD
repair đều kém. Chênh lệch SRQ--Exact +0.010 pp trên stream này cũng không phải
bằng chứng SRQ tăng accuracy.

### 2.3. Độ bền khi tăng từ 10 lên 20 task

Artifact `srq_fly_priority4_task_frequency_train_only.zip`, SHA-256
`2891b3ca7ed53c62bd63aa1cb5b3dabb374f4de392de6fa5ccf14fb7bb6690c4`,
chạy năm replicate CIFAR train-only. Trong mỗi replicate, 10-task và 20-task
dùng cùng mẫu train/validation theo lớp, class order, projection, WTA code,
Ridge và implementation; chỉ cách gom lớp thành task thay đổi.

| Schedule | Exact aligned AIA | SRQ aligned AIA | Δ AIA Exact-SRQ (pp) | \(A_{\mathrm{final}}\): Exact / SRQ |
|---|---:|---:|---:|---:|
| 10 task | 92.154 | 92.093 | 0.061 | 88.300 / 88.206 |
| 20 task | 92.154 | 92.069 | 0.085 | 88.300 / 88.142 |

Tăng gấp đôi số lần lượng tử hóa chỉ làm tăng aligned-AIA loss trung bình
0.023 pp; final-loss tăng 0.064 pp. Mức thay đổi quan sát được là nhỏ, nên thí
nghiệm chưa cho thấy lỗi tích lũy đáng kể từ 10 lên 20 task. Tuy nhiên SRQ vẫn
thấp hơn Exact một lượng nhỏ; kết quả này không chứng minh hai phương pháp hoàn
toàn tương đương.

Exact FLY cho final prediction giống hệt giữa hai schedule, residual lớn nhất
là `3.13e-6`, prediction agreement nhỏ nhất giữa SRQ và Exact là 98.41%, và SRQ
giảm 78.116% persistent state. Artifact có `uses_test_set=false` và đủ 10 unit.

### 2.4. Peak GPU memory của toàn pipeline

Artifact `srq_fly_priority5_whole_process_memory.zip`, SHA-256
`7f111e80ec3e4d12fafae39a868795fc36c967d223c99f8ade98107b5b180403`,
đo Exact FLY và P2B trong hai worker tách biệt trên một Tesla T4. Luồng đo gồm
load frozen ViT-B/16, trích xuất đủ 50.000 feature train CIFAR-100, giải phóng
backbone, chạy 10 analytic update và probe cố định 512 mẫu. Hai phương pháp dùng
cùng dữ liệu, projection, WTA, class order và Ridge; test set không được tạo.

| Đại lượng | Exact FLY | P2B | Thay đổi |
|---|---:|---:|---:|
| Persistent state | 423.44 MiB | 92.66 MiB | giảm 78.1% |
| Analytic PyTorch peak allocated | 1,802.76 MiB | 1,405.30 MiB | giảm 22.0% |
| Analytic PyTorch peak reserved | 2,422 MiB | 2,062 MiB | giảm 14.9% |
| Whole-process NVML worker peak | 2,588 MiB | 2,228 MiB | giảm 13.9% |
| Analytic-stage time | 12.01 s | 22.46 s | P2B chậm 1.87 lần |
| Tổng thời gian các stage | 542.50 s | 557.25 s | P2B chậm 2.7% |

Feature-extraction peak là 1,696 MiB ở cả hai worker. Vì backbone và workspace
chung không được nén, giảm peak toàn pipeline nhỏ hơn nhiều so với giảm
persistent state. P2B đồng ý 510/512 prediction với Exact FLY (99.609%), có
solver residual `1.61e-6`, nhưng relative logit drift vẫn là 0.167. Tất cả gate
đều pass với trạng thái `PASS_PRIORITY5_MEMORY`. Đây là bằng chứng train-only
trên một CIFAR/T4 run, không phải accuracy result hay bảo đảm cho mọi GPU.

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
- Whole-process audit xác nhận lợi ích không chỉ nằm trên giấy: process-attributed
  NVML peak giảm 13.9% và analytic PyTorch allocation peak giảm 22.0% trên T4.

### Hạn chế hiện tại

- SRQ không vượt Exact FLY về accuracy trong artifact cuối; AIA giảm nhỏ nhưng
  nhất quán trên CUB và ImageNet-R.
- Update P2B chậm hơn Exact FLY 1.60-2.09 lần do giải mã factor, blocked QR,
  lượng tử hóa lại và triangular solve sau mỗi task.
- State 97-105 MB vẫn lớn hơn raw Ridge nhiều lần; đây là memory-accuracy
  trade-off, không phải phương pháp nhỏ nhất.
- Code là int8, chưa phải 4-bit như bài ICCV, và chưa dùng error feedback.
- Whole-process memory hiện mới được đo một lần trên CIFAR-100/T4; chưa có lặp
  lại để lập khoảng tin cậy và chưa chứng minh mức giảm tương tự trên GPU khác.
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

1. **Đóng lại provenance của state-matched control.** Rerun extraction/final
   evaluation trên commit đã sửa dictionary-loader; không thay selection,
   width, lambda, seed hoặc test-time decision.
2. **Lặp systems measurement nếu claim rộng hơn.** Chạy lại Priority 5 trên
   nhiều replicate hoặc GPU khác nếu muốn tuyên bố mức giảm peak tổng quát;
   luôn tách NVML process, NVML device, PyTorch allocated và reserved.
3. **Thử error feedback như một method mới.** Chỉ triển khai sau khi có công
   thức state và bound rõ ràng; error state phải được tính vào persistent bytes.
   So sánh no-EF/EF trên train-validation trước, không tune bằng test.
4. **Thử true int4 có packing thực.** Nếu chỉ lưu int4 trong tensor int8 thì
   không được tuyên bố giảm byte. Cần pack hai giá trị mỗi byte, kiểm tra kernel,
   tốc độ giải mã và accuracy-memory Pareto.
5. **Củng cố lý thuyết.** Bổ sung bound tích lũy lỗi factor qua task, bound
   perturbation nghiệm/logit và điều kiện margin bảo toàn prediction. Không tái
   sử dụng định lý hội tụ Shampoo ngoài phạm vi của nó.
6. **Hoàn thiện bằng chứng paper.** Giữ CIFAR và CUB như kết quả đã tiêu thụ;
   thay ImageNet-R legacy bằng split sạch hoặc thêm một dataset chưa mở test.
   Báo Exact FLY là baseline chính, raw Ridge là lower-memory baseline và SOHO
   replay ở bảng riêng với toàn bộ sample-level state bytes.

## 5. Kết luận ngắn

SRQ-FLY hiện là một hướng **khả thi và có tín hiệu paper rõ về
memory-accuracy trade-off**: giảm khoảng bốn phần năm persistent state của
Exact FLY trong khi chỉ mất 0.02-0.08 pp AIA, đồng thời hơn FLY giảm-width tại
cùng state budget từ 0.132 đến 0.856 pp AIA. Whole-process audit còn xác nhận
NVML worker peak giảm 13.9% trên CIFAR/T4. Tuy nhiên, nó chưa phải phương pháp
tăng accuracy so với FLY cùng width và update vẫn chậm hơn 1.60-2.09 lần. Định
vị trung thực nhất hiện tại là “structure-preserving compression that preserves
representation width”, không phải “better FLY in every metric”.
