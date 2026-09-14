# M19 — benchmark hệ thống panel-size của SRQ

## Câu hỏi

M19 trả lời hai câu hỏi kỹ thuật hẹp, không phải một thí nghiệm độ chính xác:

1. Với kích thước cập nhật thật của bài báo, thời gian SRQ nằm chủ yếu ở giải
   mã factor, blocked QR, lượng tử hóa lại hay hai phép giải tam giác?
2. Trong năm panel size đã định trước, panel nào giảm tổng thời gian bốn stage
   trên cả width 10.000 và 20.000 mà không đổi dung lượng trạng thái hay dự
   đoán tổng hợp?

Benchmark hoàn toàn dùng dữ liệu tổng hợp, không tải dataset, feature cache,
validation split hoặc test split. Kết quả không được dùng để đưa ra tuyên bố
accuracy.

## Thiết lập khóa trước khi chạy

- GPU báo cáo chính: Tesla T4;
- seed: 2025;
- width: 10.000 và 20.000;
- một cập nhật gồm 5.000 hàng;
- số vế phải/lớp: 100;
- panel: 64, 128, 256, 512 và 1.024;
- P2B version 1: block 256, group 64, INT8 strict-upper, FP32 diagonal;
- `trailing_chunk_size=null`, đúng cấu hình backend hiện tại;
- public non-consuming QR contract: update rows được bảo toàn;
- một warm-up và ba lần đo cho mỗi cặp width/panel;
- TF32 tắt và float32 matmul precision đặt ở `highest`.

Mỗi stage được đồng bộ CUDA ngay trước và sau phép đo. Vì vậy, số đo là thời
gian wall-clock đã chờ kernel hoàn thành, không chỉ là thời gian enqueue kernel.
Median của ba lần đo là thống kê chính; minimum, maximum và số đo thô vẫn được
lưu trong artifact.

## Bốn stage

1. `factor_dequantization`: tái dựng factor FP32 từ đúng codec P2B đang dùng;
2. `blocked_qr`: cập nhật factor bằng implementation
   `methods.analytic_ridge.qr.blocked_qr_rank_update`;
3. `factor_requantization`: codec streaming P2B lượng tử hóa và giải mã tại chỗ;
4. `triangular_solve`: hai phép `torch.linalg.solve_triangular` để thu hồi W.

Sinh dữ liệu, tạo thống kê đích, residual và logits kiểm tra nằm ngoài bốn
stage. Chúng không được nhập vào thời gian update.

## Quy tắc chọn panel

Panel 128 là tham chiếu hiện hành. Với mỗi panel, lấy tỷ số median tổng thời
gian so với panel 128 riêng tại mỗi width, rồi lấy geometric mean của hai tỷ
số. Chọn panel có geometric mean nhỏ nhất; nếu bằng nhau chọn panel nhỏ hơn.
Quy tắc này không đọc accuracy, label thật hoặc test set.

Panel chỉ đủ điều kiện nếu mọi unit hoàn thành, residual không vượt `2e-5`,
state bytes giống nhau giữa các panel cùng width, timing hữu hạn, và prediction
trên probe tổng hợp không đổi. Nếu một kiểm tra không đạt, runner vẫn xuất toàn
bộ artifact với trạng thái `COMPLETE_..._WITH_WARNINGS`; không xóa unit, không
nới gate và không chọn lại grid sau khi xem kết quả.

## Diễn giải được phép

- Nếu QR chiếm tỷ trọng lớn nhất, bài báo có thể báo bottleneck là blocked QR
  và ghi panel được benchmark chọn trong phụ lục hệ thống.
- Nếu codec chiếm tỷ trọng lớn nhất, hướng tối ưu phù hợp là fused quantization
  kernel hoặc giải trực tiếp từ factor nén.
- M19 không chứng minh tăng accuracy, không so sánh phần cứng khác T4 và không
  thay thế đo end-to-end trên dữ liệu thật.

## Artifact

Notebook Colab xuất
`srq_generalization_m19_panel_system_benchmark_t4.zip`, gồm config, protocol,
result tổng hợp và 10 unit JSON. Mỗi unit là checkpoint nguyên tử nên có thể
chạy lại cell dài trong cùng runtime mà không lặp các unit đã hoàn thành.
