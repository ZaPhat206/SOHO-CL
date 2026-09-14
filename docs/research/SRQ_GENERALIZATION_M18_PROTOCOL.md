# M18 — FLY-CL 20k Adaptive locked-test closure

## Câu hỏi nghiên cứu

Ở frontend chính FLY-CL với chiều rộng 20.000, SRQ-Adaptive có giảm sai số
so với P2B INT8 thuần hay không, và phải trả thêm bao nhiêu persistent state?

M18 là thí nghiệm đóng khoảng trống phạm vi áp dụng của Adaptive. Đây không
phải một lần chọn lại phương pháp hoặc siêu tham số.

## Thiết kế đã khóa trước khi chạy test

- Dataset: CIFAR-100, 10 task class-incremental.
- Backbone: ViT-B/16 đóng băng, feature dimension 768.
- Frontend: FLY WTA, width 20.000, synaptic degree 300, coding level 0,3.
- Ba backend dùng đúng cùng projection và đúng cùng WTA codes trong mỗi lần lặp:
  1. `exact_fly_20000`: Gram FP32;
  2. `srq_fly_p2b_20000`: đường chéo FP32, strict-upper INT8;
  3. `srq_fly_adaptive_20000`: đường chéo FP32, strict-upper INT8/FP16 với
     ngân sách cố định ở 25% khoảng byte từ all-INT8 đến all-FP16.
- Sáu cặp seed kế thừa nguyên vẹn từ thí nghiệm chính:
  `(3031,5031)`, ..., `(3036,5036)` cho class order và projection.
- Ridge `lambda=10^6` kế thừa từ lựa chọn train-only đã khóa của FLY family.
  Không retune ở width 20k. Vì thế M18 là so sánh backend có kiểm soát, không
  phải tuyên bố FLY-20k đã được tối ưu riêng.
- Adaptive chọn block chỉ từ giá trị factor hiện tại, theo mức giảm MSE trên
  mỗi byte tăng thêm; không dùng label, logit, prediction hay accuracy.

## Biên test và tính trung thực báo cáo

Trước khi tạo `test.pt`, runner bắt buộc xác minh:

1. checkout Git sạch và đã commit;
2. hash của source/config;
3. artifact lựa chọn train-only gốc và file lựa chọn CIFAR nằm trong artifact;
4. `train.pt` đúng checkpoint/dataset và `test.pt` chưa tồn tại;
5. toàn bộ seed, method, Ridge và chính sách Adaptive đã cố định.

Test split CIFAR-100 đã được dùng trong nghiên cứu SRQ-FLY chính. Vì vậy M18
phải được mô tả là *prespecified frontend-closure comparison*, không phải
fresh first-use held-out evaluation.

Không có accuracy gate, không retry theo accuracy và không loại seed. Kết quả
chỉ được tổng hợp khi đủ cả sáu replicate. Nếu một structural/numerical gate
thất bại, artifact vẫn phải được giữ nguyên để chẩn đoán.

## Đại lượng phải báo cáo

- AIA, final accuracy và forgetting: mean, sample standard deviation và paired
  difference so với Exact;
- accuracy curve theo task;
- persistent state tổng và theo task;
- update/inference time;
- solver relative residual;
- prediction agreement và relative logit Frobenius error so với Exact;
- số block/giá trị FP16 được Adaptive chọn, byte sử dụng và byte ceiling.

## Gate hợp lệ

- đủ sáu paired replicate và không bỏ seed;
- ba phương pháp dùng cùng WTA cache trong từng replicate;
- `Exact state > Adaptive state > INT8 state` ở mọi task;
- factor Adaptive nằm trong ngân sách đã khóa;
- maximum solver relative residual không vượt `2e-5`;
- accuracy gate là `null`.

## Resume

Mỗi replicate hoàn thành được ghi ngay vào `output/units/` và cập nhật
`m18_progress.json`. Chạy lại cùng lệnh trong cùng source/data/authorization
context sẽ bỏ qua những replicate đã hoàn thành. WTA cache là hạ tầng thí
nghiệm chứa dữ liệu mức mẫu; nó không phải learner state và không được đưa vào
checkpoint triển khai.

## Quy tắc cập nhật paper

Không sửa claim trước khi có `m18_results.json`. Nếu toàn bộ gate cấu trúc qua,
paper báo cáo cả sáu replicate bất kể dấu của chênh lệch accuracy. Nếu Adaptive
không cải thiện INT8 trên FLY-20k, kết luận đúng là lợi ích Adaptive phụ thuộc
frontend/geometry; không được loại kết quả hoặc chọn riêng seed đẹp.
