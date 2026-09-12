# M13-N — kiểm toán số học cho đối chứng LoRanPAC

## 1. Artifact và phạm vi

- Artifact: `srq_generalization_m13n_numerical_audit.zip`
- SHA-256 toàn ZIP:
  `726853486664cbf26ec109a061585a1effcd90569ce056108e9ff594f94e031d`
- SHA-256 của `m13n_results.json`:
  `8bb9725856022bb8f770764fa55a7619562570aebebcec2a0d98b356fd710737`
- Source commit:
  `e12662a37816e031ec60da1e5680b6427ab24a35`
- Trạng thái: `PASS_M13N_NUMERICAL_AUDIT`
- `uses_test_set=false`; `computes_predictive_metrics=false`.

ZIP hợp lệ, toàn bộ member khớp manifest và tất cả gate của M13-N đều pass.
M13-N không đọc nội dung khoa học của ZIP M13 nguồn; nó chỉ xác minh SHA-256
của container. Train cache, checkpoint, tập chỉ số train/validation, số chiều
đặc trưng và toàn bộ rank contract đều khớp.

Lần chạy đầu của M13-N bị dừng trước phép SVD do hai SHA của tập chỉ số được
đăng ký sai. Recovery chỉ sửa hai identity này sau khi đối chiếu dữ liệu nguồn;
không metric số học hoặc accuracy nào đã được quan sát, và không thay rank,
threshold, phương pháp hay seed.

## 2. Kết quả chính

`Raw` là basis FP32 trả về ở task đầu. `QR diagnostic` trực chuẩn hóa lại basis
để xác định phần sai số đến từ mất trực giao. QR này chỉ là chẩn đoán; nó không
được dùng để thay thế LoRanPAC trong M13.

| Width | Budget | Rank task 1 | Raw Frobenius | Raw Frobenius / sqrt(rank) | Raw spectral | Raw solver residual | QR Frobenius | QR solver residual |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10.000 | P2B | 1.328 | 0,041057 | 0,001127 | 0,002435 | 6,34e-4 | 5,11e-5 | 5,32e-7 |
| 10.000 | Adaptive | 1.621 | 0,043089 | 0,001070 | 0,002435 | 6,32e-4 | 2,21e-5 | 5,46e-7 |
| 20.000 | P2B | 2.656 | 0,054541 | 0,001058 | 0,002492 | 6,07e-4 | 3,69e-5 | 8,31e-7 |
| 20.000 | Adaptive | 3.000 | 0,057439 | 0,001049 | 0,002491 | 6,09e-4 | 6,50e-5 | 5,03e-7 |

Giá trị M13-N tái hiện metric raw của M13 gần như chính xác. Sau QR chẩn đoán,
sai số trực giao Frobenius giảm ít nhất 803,65 lần và solver residual giảm ít
nhất 730,74 lần. Trong oracle FP64 kích thước nhỏ, Frobenius chuẩn hóa là
`1,92e-15` và solver residual ở cỡ `2,4e-15`, tức gần giới hạn số học máy.

## 3. Diễn giải khoa học

M13 thất bại vì hai gate tuyệt đối đã khóa trước không phù hợp với basis SVD
FP32 rank lớn ở task đầu:

1. Raw Frobenius cộng sai số trên toàn bộ số cột nên tăng theo rank, trong khi
   Frobenius chuẩn hóa và spectral norm gần như ổn định giữa bốn cấu hình.
2. Mất trực giao nhỏ của basis đi vào công thức projected solve, tạo residual
   khoảng `6e-4`; trực chuẩn hóa chẩn đoán đưa residual về cỡ `1e-7`.
3. Oracle FP64 loại trừ khả năng đây là sai sót đại số cơ bản của adapter.

Kết luận được phép là: M13-N xác nhận lỗi formal của M13 là hiệu ứng
metric/precision ở task đầu, không phải bằng chứng adapter LoRanPAC sai. Kết
luận không được phép là: QR đã làm M13 pass, hoặc M13 đã trở thành bằng chứng
đa seed. `FAIL_M13_LORANPAC_TRAIN_ONLY` vẫn được giữ nguyên.

## 4. Quyết định tiếp theo

M14 sẽ là đánh giá train-only đa seed được đăng ký mới. Nó giữ nguyên đường
LoRanPAC chính thức, luôn báo raw Frobenius nhưng dùng Frobenius chuẩn hóa,
spectral norm và solver residual có ngưỡng scale-aware làm gate số học. Không
được chọn rank, budget, Ridge hay kết luận từ accuracy sau khi chạy.

