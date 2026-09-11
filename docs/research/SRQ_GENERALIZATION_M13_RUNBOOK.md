# M13 — LoRanPAC equal-budget challenger screen (train-only)

## Câu hỏi nghiên cứu

M13 kiểm tra rủi ro phản biện lớn nhất còn lại của SRQ: nếu thay dense
square-root bằng một trạng thái hạng thấp kiểu LoRanPAC, ta có đạt accuracy tốt
hơn ở cùng số byte hay không? Đây là thí nghiệm sàng lọc đối thủ trực tiếp,
không phải một lần mở test mới.

## Thiết kế đã khóa

- Dùng đúng CIFAR-100 train/validation stream, ViT-B/16 cache, random-ReLU
  projection, class order và projection seed của M6/M11.
- Chỉ chạy width 10,000 và 20,000.
- Với mỗi width có hai ngân sách: final state của P2B và final state của
  adaptive INT8/FP16. Rank cap là số nguyên lớn nhất vừa ngân sách đó, tính từ
  shape/dtype trước khi xem accuracy.
- Cập nhật TSVD theo repository LoRanPAC tại commit
  `32782f9d260e5d722de5702675ed66cca8234883`.
- `truncate_percent=25`. Quy tắc rank chính tái lập code công khai bằng Python
  `round`; paper viết `ceil`, nên artifact phải công bố khác biệt này.
- Báo hai nghiệm trên cùng factor: Ridge 0 theo config CIFAR chính thức của
  LoRanPAC và Ridge đã khóa từ M6. Nghiệm thứ hai chỉ là diagnostic để tách ảnh
  hưởng của TSVD khỏi khác biệt regularization; hai accuracy không được dùng để
  chọn lại rank hoặc phương pháp.
- Không tạo, đọc hoặc đánh giá `test.pt`. Không có accuracy gate.

## Persistent state được tính

M13 tính projection random-ReLU, `Q=H^T Y`, class counts, classifier weights,
left singular vectors `U` và singular values. Current-task codes, residual QR
và core-SVD workspace không phải persistent state; chúng phải được đo riêng
nếu LoRanPAC vượt qua màn sàng lọc accuracy--state.

Với width `E`, số lớp `c` và rank `r`, backend FP32 sau task cuối có

```text
backend bytes = 8*E*c + 4*c + 4*E*r + 4*r.
```

Do đó rank được suy ra duy nhất từ target bytes. Sai lệch ngân sách phải không
âm và nhỏ hơn số byte của đúng một rank bổ sung.

## Ý nghĩa kết quả

- Nếu LoRanPAC cao hơn SRQ tại cùng byte: không được tiếp tục định vị SRQ như
  Pareto tốt nhất. Paper nên chuyển sang so sánh cơ chế full-rank
  structure-preserving với low-rank truncation, hoặc phải cải tiến SRQ trước.
- Nếu SRQ cao hơn ở cả hai width/budget: đây là bằng chứng rằng giữ đủ các
  hướng trong factor có lợi hơn cắt hạng trong regime này. Khi đó mới đáng chạy
  multi-seed và task-frequency lớn hơn.
- Nếu kết quả chia đôi theo width/budget: báo frontier, không chọn riêng điểm có
  lợi sau khi xem validation.

M13 chỉ là controlled analytic-head comparison trên cached ViT features. Nó
không phải tái lập LoRanPAC end-to-end: chưa tái tạo PETL, toàn bộ augmentation,
backbone và lịch Inc-1 của paper chính thức.

## Trình tự sau M13

1. Chạy toàn bộ gate cục bộ và runner train-only; giữ artifact dù PASS hay FAIL
   về accuracy vì PASS chỉ phản ánh integrity/numerical gates.
2. Nếu challenger cho kết quả khoa học có ích, M14 lặp nhiều seed và thêm lịch
   100 task/Inc-1 ở một width đã định trước.
3. Sau M14 mới quyết định có đủ cơ sở cho một confirmation khóa test hay không.
4. Đo peak memory/workspace và runtime trong process tách biệt cho phương pháp
   nằm trên Pareto frontier; persistent byte một mình chưa đủ.
