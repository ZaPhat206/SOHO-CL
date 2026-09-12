# M14 — kế hoạch xác nhận đa seed cho đối chứng LoRanPAC cùng byte

## 1. Câu hỏi nghiên cứu

Tại cùng tổng persistent-state budget, kết luận mô tả của M13 có bền qua nhiều
class-order/projection seed hay không: giữ full-width additive Ridge bằng SRQ
so với nén representation bằng truncated-SVD LoRanPAC?

M14 là train-only confirmation. Nó không mở, tạo hoặc đọc test cache. Mục tiêu
là đo độ bền của so sánh cùng byte, không phải tìm thêm phương pháp hoặc tune
LoRanPAC sau khi thấy accuracy.

## 2. Thiết kế khóa trước

- Dataset: CIFAR-100 train split và train/validation identity đã khóa.
- Frontend: controlled RanPAC random-linear--ReLU.
- Width: 10.000 và 20.000.
- Schedule chính: 10 task.
- Sáu cặp seed mới: `4101` đến `4106`; trong mỗi cặp, mọi backend dùng cùng
  class order, projection prefix, mẫu train và validation.
- Hai budget: final total state thực đo của P2B và adaptive INT8/FP16 trong
  chính paired seed đó. Vì adaptive mask phụ thuộc giá trị factor, cách này
  tránh dùng byte của một seed cũ làm đại diện không chính xác cho seed mới.
- Rank LoRanPAC là rank nguyên lớn nhất không vượt final byte của backend được
  ghép cặp; rank chỉ được suy ra sau khi state byte của backend tương ứng đã
  được kiểm tra và không được chọn từ accuracy.
- Ridge và adaptive policy giữ nguyên các lựa chọn train-only đã khóa ở
  M6/M11. LoRanPAC dùng nghiệm Ridge đã khóa cho control chính và báo thêm
  nghiệm zero-Ridge chính thức chỉ như sensitivity field nếu không làm thay
  đổi rank hoặc backend.

Mỗi seed/width chạy một Exact sentinel, P2B, adaptive và hai LoRanPAC tương ứng
với hai budget. Tổng cộng có 60 unit có thể resume độc lập. Kết quả chính báo
mean ± sample SD, paired difference theo seed, số seed cùng dấu, final
accuracy, AIA, state byte và update time. Accuracy không phải completion gate.

## 3. Gate số học mới

Các gate dưới đây được đăng ký từ M13-N trước khi chạy M14:

- luôn báo raw `||U^T U-I||_F`, nhưng không dùng nó làm gate vì phụ thuộc rank;
- `||U^T U-I||_F / sqrt(rank) <= 1.5e-3` ở mọi task;
- task-1 `||U^T U-I||_2 <= 3.5e-3`; spectral norm được tính chính xác bằng
  symmetric eigendecomposition ở task đầu, không lặp lại phép toán bậc ba này
  sau mỗi update khi raw/normalized Frobenius và solver residual đã được báo;
- task-1 solver relative residual `<= 1.0e-3`;
- solver relative residual ở task 2--10 `<= 2.0e-5`;
- mọi metric hữu hạn, rank/state contract đúng và final state không vượt
  target byte.

Ngưỡng được đặt với khoảng đệm hữu hạn so với cực đại M13-N lần lượt là
`1.127e-3`, `2.492e-3` và `6.34e-4`. Không được nới ngưỡng sau khi quan sát M14.
QR reorthogonalization chỉ được phép xuất hiện trong trường chẩn đoán riêng;
nó không thay basis, singular value hoặc prediction của đường LoRanPAC chính.

## 4. Quy tắc diễn giải

- M14 pass nghĩa là toàn bộ unit hoàn thành và đạt gate nguồn/số học; không có
  nghĩa SRQ thắng về accuracy.
- Nếu SRQ có mean AIA cao hơn, chỉ kết luận lợi thế quan sát được trên sáu seed
  và stream này; phải công bố độ lệch chuẩn và từng paired difference.
- Nếu LoRanPAC cao hơn tại một budget, phải giữ kết quả và thu hẹp Pareto claim;
  không được đổi rank, Ridge hoặc seed để đảo kết luận.
- Không gọi đây là reproduction end-to-end của LoRanPAC hoặc RanPAC; đây là
  đối chứng backend cùng frontend, dữ liệu, width và byte accounting.
- Lịch 20 task và dataset khác là thí nghiệm kế tiếp riêng, không được thêm vào
  M14 sau khi xem kết quả 10 task.
