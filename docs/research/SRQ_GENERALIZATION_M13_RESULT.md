# M13 — LoRanPAC equal-budget challenger: kết quả train-only

## 1. Artifact và trạng thái

- Artifact: `srq_generalization_m13_loranpac_train_only.zip`
- SHA-256: `b7cc3e1993b150d829806ac8062b10a2e31ad9c533ef729ce7a806647496d28c`
- Source commit: `120a84cd0340ef880d34680e07b6225b356f0928`
- Trạng thái khóa: `FAIL_M13_LORANPAC_TRAIN_ONLY`
- `uses_test_set=false`; không có accuracy gate và không chọn rank từ accuracy.

Toàn bộ chín member được liệt kê trong manifest khớp SHA-256. Các identity
check của train cache, class order, train/validation indices, full projection
và hai projection prefix đều pass. Bốn đơn vị width/budget đều hoàn thành;
rank được suy ra đúng từ byte và final state không vượt target.

## 2. So sánh accuracy tại cùng persistent-state budget

LoRanPAC dưới đây dùng nghiệm Ridge 0 theo config CIFAR của implementation
chính thức. `Delta` được tính bằng LoRanPAC trừ backend SRQ tương ứng. Đây là
một development seed nên các chênh lệch chỉ có giá trị mô tả.

| Width | Ngân sách | Rank LoRanPAC | AIA Lo / SRQ | Delta AIA (pp) | Final Lo / SRQ | Delta Final (pp) |
|---:|---|---:|---:|---:|---:|---:|
| 10,000 | P2B INT8 | 1,328 | 91.817 / 92.355 | -0.538 | 87.05 / 88.23 | -1.18 |
| 10,000 | Adaptive INT8/FP16 | 1,621 | 91.955 / 92.434 | -0.480 | 87.25 / 88.44 | -1.19 |
| 20,000 | P2B INT8 | 2,656 | 92.391 / 92.387 | +0.004 | 88.08 / 88.52 | -0.44 |
| 20,000 | Adaptive INT8/FP16 | 3,242 | 92.429 / 92.647 | -0.218 | 88.27 / 88.94 | -0.67 |

Byte underfill chỉ từ 29,256 đến 78,752 byte, luôn nhỏ hơn byte của một rank
bổ sung. Vì vậy khác biệt không đến từ việc cấp cho LoRanPAC một budget nhỏ
hơn đáng kể.

Nghiệm dùng Ridge M6 đã khóa thấp hơn nghiệm Ridge 0 từ 0.034 đến 0.061 điểm
AIA và từ 0.03 đến 0.13 điểm final. Do đó kết quả yếu hơn của TSVD không thể
được giải thích bằng việc M13 chọn Ridge 0 thay vì hệ số Ridge của SRQ.

Kết luận mô tả đúng là: LoRanPAC không thống trị SRQ trên stream này. SRQ cao
hơn rõ tại width 10k và tại adaptive budget. Ở width 20k/P2B budget, LoRanPAC
và P2B gần như hòa về AIA nhưng P2B cao hơn 0.44 điểm final. Không được kết
luận SRQ luôn tốt hơn low-rank trên mọi seed, dataset hoặc protocol.

## 3. Vì sao artifact vẫn FAIL?

Hai gate thất bại đều đạt cực đại ở task đầu:

| Unit | Task-1 `||U^T U-I||_F` | Chia `sqrt(rank)` | Task-1 solver residual | Max residual task 2--10 |
|---|---:|---:|---:|---:|
| 10k / P2B budget | 0.04107 | 0.001127 | 6.34e-4 | 3.27e-6 |
| 10k / adaptive budget | 0.04309 | 0.001070 | 6.32e-4 | 2.65e-6 |
| 20k / P2B budget | 0.05453 | 0.001058 | 6.07e-4 | 3.38e-6 |
| 20k / adaptive budget | 0.05739 | 0.001048 | 6.09e-4 | 3.75e-6 |

Gate đã khóa yêu cầu orthogonality residual không quá `0.01` và solver
residual không quá `2e-5`. Implementation LoRanPAC công khai lấy trực tiếp
left singular vectors ở lần SVD đầu, còn các update sau mới QR lại basis. Trên
CUDA/FP32 và rank 1,328--3,242, raw Frobenius norm ở task 1 tăng theo số cột.
Sau QR của task 2, orthogonality residual giảm xuống khoảng `2e-5--7e-5` và
solver residual còn dưới `3.8e-6`.

Quan sát này gợi ý gate raw-Frobenius không scale-aware, nhưng không cho phép
đổi trạng thái M13 thành PASS sau khi đã xem kết quả. Accuracy trong M13 chỉ
được giữ như descriptive development evidence.

## 4. Cổng tiếp theo: M13-N numerical audit

Không chạy ngay M14 hoặc mở test. M13-N phải là audit không dùng accuracy:

1. Đối chiếu task-1 factor và classifier với source LoRanPAC đã pin.
2. Báo đồng thời raw Frobenius, Frobenius chia `sqrt(rank)` và spectral norm của
   `U^T U-I`, tránh dùng một đại lượng phụ thuộc rank mà không chuẩn hóa.
3. Tách residual của công thức projected solve khỏi sai số trực chuẩn của
   basis, và so sánh CUDA FP32 với một oracle FP64 trên bài toán đủ nhỏ.
4. Không đổi `truncate_percent`, rank cap, Ridge, seed hoặc accuracy decision.
5. M13 vẫn giữ FAIL. Nếu M13-N xác nhận đây là metric/precision effect chứ
   không phải sai adapter, M14 phải đăng ký gate scale-aware mới trước khi chạy.

Chỉ sau M13-N mới hợp lệ để chạy multi-seed và lịch nhiều task nhằm kiểm tra
liệu kết luận accuracy--state có bền hay không.
