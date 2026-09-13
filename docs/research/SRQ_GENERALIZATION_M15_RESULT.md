# M15 — khép kín số học cho residual task 1 của LoRanPAC

## 1. Artifact và tính toàn vẹn

- Artifact: `srq_generalization_m15_loranpac_task1_closure.zip`
- SHA-256 toàn ZIP:
  `942cd777674d3e1089ef60b7c1835de70b283b00d948b77c43ab673497a6f9c6`
- SHA-256 của `m15_results.json`:
  `71b365bdecd2337c4774dfbf444eded2c4d0bea43af4c5bdfced504c6293b9b7`
- Source commit: `13b0bd063c471ebddfd7e054d5f1a02a50eeeed4`
- Trạng thái: `PASS_M15_LORANPAC_TASK1_CLOSURE`
- Phạm vi: train-only, không prediction, không accuracy và không đọc test set.

ZIP có 10 member duy nhất. Cả 9 file được khai báo trong manifest đều khớp
SHA-256. Artifact chứa đủ 8/8 record: hai seed `4101/4105`, hai width
`10.000/20.000` và hai ngân sách P2B/adaptive. Source, backend, train cache,
split, projection và rank đều khớp M14.

## 2. Câu hỏi M15 kiểm tra

M14 thất bại đúng một gate vì residual của projected solver ở task 1 đạt
`1,9148e-3`, vượt ngưỡng khóa trước `1e-3`. M15 không sửa solver chính thức và
không thay gate. Nó tách ba khả năng:

1. residual chỉ do thực hiện phép tính ở FP32;
2. factor SVD FP32 đã hơi mất trực chuẩn, nhưng công thức projected solver vẫn
   xem `U^T U=I` một cách chính xác;
3. adapter hoặc phương trình Ridge bị sai về đại số.

Với cùng factor `U` đã lưu, M15 tính lại công thức chính thức ở FP32 và FP64.
Sau đó nó dùng QR `U=QT`, chuyển core thành

\[
C=T\,\operatorname{diag}(s^2)T^\top,
\]

để bảo toàn đúng hệ truncated
`U diag(s^2) U^T = Q C Q^T`, rồi giải core tổng quát thay vì giả định `T=I`.
QR ở đây chỉ là oracle chẩn đoán, không thay thế đường LoRanPAC trong M14.

## 3. Kết quả

| Đại lượng | Kết quả | Ý nghĩa |
|---|---:|---|
| Sai khác tương đối lớn nhất khi tái lập residual M14 | `3,2123e-4` | M15 tái lập hiện tượng M14; khác biệt chỉ khoảng 0,0322% |
| Tỷ lệ residual FP64/FP32 trên cùng factor | `0,99966--1,00025` | Chỉ đổi arithmetic sang FP64 không sửa được lỗi |
| Sai số tái dựng factor của đường QR bảo toàn hệ | `<=7,5299e-7` | QR/core vẫn biểu diễn cùng truncated system ở độ chính xác FP32 |
| Backward error lớn nhất của core FP64 | `5,0595e-17` | Hệ bảo toàn được giải gần machine precision |
| Hệ số cải thiện backward error nhỏ nhất | `2,5360e11` lần | Lỗi không nằm ở phương trình Ridge tổng quát |
| Sai khác trọng số tương đối lớn nhất, official so với core bảo toàn hệ | `1,3706e-4` | Ảnh hưởng lên nghiệm nhỏ nhưng khác không tuyệt đối bằng 0 |

Kết quả then chốt là residual gần như không đổi khi chỉ chuyển phép tính chính
thức sang FP64. Nghĩa là sai số đã nằm trong cơ sở `U` được tạo bởi SVD FP32;
nó không phải chỉ là sai số làm tròn ở bước solve cuối. Khi QR được dùng cùng
phép biến đổi core bảo toàn hệ, backward error giảm xuống cỡ `1e-17`. Vì vậy,
M15 không tìm thấy lỗi đại số trong adapter Ridge. Cơ chế phù hợp với quan sát
là cơ sở SVD FP32 hơi mất trực chuẩn, trong khi projected formula chéo hóa core
dưới giả định trực chuẩn chính xác.

## 4. Kết luận khoa học và giới hạn tuyên bố

- M15 **PASS** với vai trò kiểm toán số học.
- M14 vẫn là `FAIL_M14_LORANPAC_MULTISEED_TRAIN_ONLY`; không gate nào được nới
  và không seed nào bị loại.
- Có thể báo cáo kết quả accuracy sáu seed của M14 như bằng chứng mô tả cùng
  byte, kèm chú thích rõ formal FAIL và nguyên nhân residual task 1.
- Không được nói M15 chứng minh hai đường cho prediction giống nhau: M15 cố ý
  không tính prediction/accuracy. Sai khác trọng số tối đa `1,37e-4` chưa tự nó
  là một bound bảo toàn nhãn dự đoán.
- Nếu thay projected solver chính thức bằng QR/core tổng quát, đó là một
  comparator mới và phải đăng ký/chạy lại; không được dùng kết quả oracle để
  sửa ngược M14.

M15 khép lại nhu cầu chẩn đoán task 1. Bước có giá trị khoa học tiếp theo không
phải lặp thêm cùng audit, mà là xác nhận end-to-end trên frontend/dataset chưa
được dùng để chọn biến thể, với protocol và solver được khóa trước.
