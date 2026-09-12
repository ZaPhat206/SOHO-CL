# M15 — đóng kiểm toán số học task 1 của LoRanPAC

## 1. Câu hỏi và ranh giới

M15 trả lời đúng một câu hỏi sau formal FAIL của M14: residual task-1
`1,9148e-3` ở seed 4105 là hệ quả của basis SVD FP32 chưa trực chuẩn hoàn hảo,
hay là bằng chứng công thức solver/adapter không giải đúng hệ truncated?

M15 là audit train-only và không predictive. Nó không đọc hoặc tạo `test.pt`,
không tính accuracy/AIA/logit/prediction, không lựa chọn rank, Ridge, seed hoặc
phương pháp từ chất lượng dự đoán. M14 vẫn mang trạng thái
`FAIL_M14_LORANPAC_MULTISEED_TRAIN_ONLY` bất kể kết quả M15.

## 2. Thiết kế khóa trước

- Seed chẩn đoán: `4105`, unit tạo cực đại task-1 residual của M14.
- Seed sentinel: `4101`, một unit cùng thiết kế đã đạt gate M14.
- Width: `10.000` và `20.000`.
- Budget/rank: đúng hai prefix rank lấy từ các unit LoRanPAC P2B/adaptive trong
  artifact M14 đã khóa; không suy ra lại từ accuracy.
- Tổng cộng tám record: hai seed, hai width và hai budget. Mỗi seed/width chỉ
  chạy một SVD task 1; hai budget dùng hai prefix của cùng factor.
- Cùng train cache, checkpoint, split, class order và projection seed như M14.

M15 mở artifact M14 chỉ để xác minh status, gate, unit identity, rank contract,
split hash và residual tham chiếu. Artifact không chứa sample hoặc factor.

## 3. Các phép đo

Với basis (U\in\mathbb{R}^{d\times r}), singular values \(s\), cross-statistic
\(C\) và \(D=\operatorname{diag}(s^2+\lambda)\), đường chính của LoRanPAC dùng

\[
B=U^\top C,\qquad W=UD^{-1}B.
\]

M15 báo residual cũ

\[
r_{\mathrm{legacy}}=
\frac{\|D(U^\top W)-B\|_F}{\max(\|B\|_F,1)}
\]

và normwise backward error

\[
\eta=
\frac{\|D(U^\top W)-B\|_F}
{\|D\|_2\|U^\top W\|_F+\|B\|_F}.
\]

Hai đại lượng được tính trong FP32 và được tính lại trong FP64 trên chính factor
FP32 đã lưu. Phép tính FP64 này tách rounding của solver khỏi sai số đã nằm sẵn
trong (U); nó không phải một FP64-SVD oracle mới.

### QR bảo toàn hệ

QR chẩn đoán M13-N thay (U) bằng basis trực chuẩn nhưng giữ nguyên (s), nên
làm thay đổi nhẹ hệ truncated. M15 dùng biến đổi đúng hơn. Nếu

\[
U=\bar U T,
\]

thì hệ truncated được giữ nguyên qua

\[
U\operatorname{diag}(s^2)U^\top
=\bar U\left(T\operatorname{diag}(s^2)T^\top\right)\bar U^\top.
\]

M15 giải core dày tương ứng trong FP32 và FP64, báo reconstruction error,
core-solve backward error và độ lệch trọng số so với công thức chính. Đây chỉ
là oracle chẩn đoán; không thay factor, classifier hoặc prediction của M14.

## 4. Gate và diễn giải

M15 pass khi source/cache/split/rank đều đúng, đủ tám record, mọi metric hữu
hạn, QR tái dựng factor với relative error không quá `1e-5`, FP64 core solve có
backward error không quá `1e-10`, và cải thiện ít nhất `100x` so với backward
error FP64 của công thức raw. Các ngưỡng này được khóa trước khi chạy M15 và
không thay thế gate `1e-3` đã thất bại của M14.

- Nếu các gate đạt, bằng chứng ủng hộ kết luận lỗi M14 đến từ trực giao FP32 ở
  basis task 1 chứ không phải sai đại số cơ bản của adapter. M14 vẫn là FAIL.
- Nếu identity/rank không khớp, M15 dừng mà không đưa ra kết luận số học.
- Nếu core bảo toàn hệ vẫn có backward error cao, không được dùng M14 cho claim
  mạnh về LoRanPAC; phải audit adapter hoặc sửa dưới một protocol mới.
