# M16 — xác nhận SRQ trên Stanford Cars với RanPAC Phase-2/ResNet-50

## 1. Câu hỏi khoa học

M16 kiểm tra liệu backend SRQ đã khóa từ các milestone trước có còn tạo một
điểm đánh đổi bộ nhớ--độ chính xác hữu ích khi đồng thời thay cả dataset và
backbone. Frontend là đường Phase-2 không PETL của RanPAC:

\[
h(x)=\operatorname{ReLU}(f_{\mathrm{R50}}(x)W_{\mathrm{rand}}),\qquad
(H^\top H+\lambda I)W=H^\top Y.
\]

Stanford Cars chưa được dùng để chọn P2B hay luật adaptive. ResNet-50
ImageNet-1K V2 có 2.048 chiều, khác backbone ViT-B/16 768 chiều của các thí
nghiệm trước. Vì vậy M16 là phép kiểm tra out-of-development có giá trị hơn
việc tiếp tục tìm biến thể trên CIFAR-100.

## 2. Mức độ tương thích với RanPAC chính thức

Repository RanPAC được khóa tại commit
`cf4b301d18b0c27db030f4371b72b768005ae58a`. Cấu hình dùng đúng hàng số 10
trong `args/cars_publish.csv`: Cars, ResNet-50 pretrained, `init_cls=16`,
`increment=20`, chuẩn hóa ImageNet, random projection và `M=10000`. Phần
analytic dùng đúng ánh xạ standard-normal + ReLU và thống kê cộng dồn
`G += H^T H`, `Q += H^T Y`.

M16 **không phải** tái lập toàn bộ RanPAC end-to-end và không dùng PETL. Một
khác biệt được công bố rõ là Ridge: code chính thức chọn lại lambda trên mỗi
task, còn square-root recurrence chứa `sqrt(lambda) I` trong factor. M16 do đó
chọn lambda một lần trên task đầu bằng lưới và tiêu chí MSE 80/20 chính thức,
rồi khóa lambda cho phần còn lại của stream. Exact, P2B và Adaptive luôn dùng
cùng class order, projection và lambda trong từng replicate. Đây là so sánh
backend công bằng; không được dùng kết quả để tuyên bố đã tái lập con số của
paper RanPAC.

## 3. Protocol khóa trước

- Dataset: official Stanford Cars, 196 lớp, 8.144 train và 8.041 test.
- Nguồn tải: Kaggle mirror `jutrera/stanford-car-dataset-by-classes-folder`,
  version 2, đã được sắp theo `train/<class>` và `test/<class>`. Handle có version,
  tổng số byte và mô tả giấy phép được khóa trong config; mirror chỉ là phương tiện
  phân phối lại official split, không phải một split thực nghiệm mới.
- Schedule: 16 lớp đầu, sau đó 9 task x 20 lớp.
- Backbone: ResNet-50 IMAGENET1K_V2; checkpoint URL, kích thước và SHA-256
  đầy đủ được khóa trong config.
- Biểu diễn: random projection standard-normal 2.048 x 10.000 và ReLU.
- Replicate: sáu bộ seed 1993--1998; seed đầu trùng cấu hình công bố.
- Phương pháp: Exact Gram FP32, P2B INT8/FP32 đã khóa, Adaptive INT8/FP16 với
  ngân sách 25% đã khóa.
- Ridge: trên 16 lớp của task đầu, shuffle xác định bởi `ridge_split_seed`,
  dùng 80% fit và 20% validation; chọn MSE thấp nhất trên
  `10^{-8},...,10^8`, hòa thì chọn lambda nhỏ nhất. Nghiệm dual được dùng vì
  tương đương toán học với primal nhưng không cần 17 phân rã 10.000 x 10.000.
  Sau lựa chọn, cả task đầu được dùng để cập nhật learner.
- Chỉ sau khi sáu lambda, config, commit, train cache và checkpoint identity
  đã được khóa vào authorization thì mới được tạo `test.pt`.
- Test không có accuracy gate, không chọn lại phương pháp và không retry theo
  kết quả.

## 4. Báo cáo bắt buộc

Cho mỗi phương pháp, báo cáo mean và sample standard deviation trên sáu
replicate của AIA, final accuracy, persistent state và thời gian analytic
update. Báo cáo paired difference so với Exact. Bảng chính không dùng paired
95% CI. Danh sách lambda đã chọn và toàn bộ accuracy theo task phải có trong
artifact.

Gate chỉ kiểm tra tính toàn vẹn, hoàn thành, residual và việc hai backend nén
nhỏ hơn Exact; không gate nào dựa vào accuracy. Nếu accuracy kém, kết quả vẫn
phải được giữ và báo cáo.

## 5. Tuyên bố được phép nếu M16 hoàn thành

Nếu các gate hệ thống qua, có thể nói SRQ là backend trạng thái đã được kiểm
tra trên hai backbone và trên một cấu hình Phase-2 của RanPAC ngoài dataset
phát triển. Không được gọi là “universal plug-in”, không được nói đây là full
RanPAC/PETL, và không được suy rộng sang mọi analytic learner.

## 6. Executable

Notebook Colab khóa source là
`notebooks/srq_generalization_m16_cars_phase2_colab.ipynb`. Runner lưu từng
unit riêng và tạo handoff sau mỗi sáu unit; vì vậy hai checkpoint 6/18 và
12/18 có thể được tải về máy, kiểm tra hash, rồi nhập lại nếu phải đổi runtime.
Artifact cuối dự kiến là
`srq_generalization_m16_cars_phase2_locked.zip`.
