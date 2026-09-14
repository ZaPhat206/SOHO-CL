# M20 — độ nhạy ngân sách và tiêu chí chọn block của SRQ-Adaptive (train-only)

## Câu hỏi

M20 trả lời ba câu hỏi, đều trên dữ liệu huấn luyện; không mở tập test:

1. **Độ nhạy ngân sách.** Với tiêu chí chọn block hiện hành (giảm sai số tái
   dựng bình phương của thừa số trên mỗi byte), độ chính xác và bộ nhớ của
   SRQ-Adaptive thay đổi thế nào khi ngân sách đi từ 0% (toàn INT8) đến 100%
   (toàn FP16)? Ngân sách 25% dùng trong bài báo có cần thiết không?
2. **Tiêu chí theo sai số hệ.** Nếu trọng số hóa sai số của từng hàng bằng bình
   phương chuẩn của hàng tương ứng trong thừa số chưa lượng tử hóa, việc chọn
   block có giảm sai số bộ phân loại ở cùng ngân sách không?
3. **Chấm điểm theo lô.** Tính lợi ích của mọi block theo lô trên GPU nhanh hơn
   vòng lặp từng block bao nhiêu, và có cho đúng cùng quyết định chọn không?

M20 không thay đổi bất kỳ kết quả test nào đã báo cáo. Ngân sách 25% trong bài
báo đã được khóa trước khi đánh giá test và vẫn giữ nguyên.

## Thiết lập khóa trước khi chạy

- CIFAR-100, 10 tác vụ, backbone ViT-B/16 đóng băng (checkpoint
  `32aa17d6…`), cache `train.pt` có SHA-256 `ba53b821…`; `test.pt` bị cấm.
- Head RanPAC phase-2 (phép chiếu Gauss + ReLU, không PETL), chiều rộng 20.000,
  `lambda = 10^6` (giá trị M6 đã chọn cho chiều rộng này, dùng cho mọi luồng).
- Tập validation: 20% của từng lớp trong tập huấn luyện.
- Ba luồng phát triển:
  - `s2025`: seed thứ tự lớp, tách validation và phép chiếu đều là 2025. Luồng này
    bị khóa danh tính theo M6 (thứ tự lớp, chỉ số train/validation, phép chiếu).
  - `s2026`, `s2027`: seed tương ứng 2026 và 2027; không có artifact tham chiếu.
- P2B: block 256, nhóm 64, panel QR 128, bước đầu Gram-Cholesky, lượng tử hóa
  streaming lô 64. TF32 tắt, `float32_matmul_precision=highest`.
- Phương pháp trong mỗi luồng:
  - `exact` (Gram FP32) và `p2b_int8`;
  - tiêu chí `factor_mse` (backend khóa `SquareRootBackend`) với ngân sách
    {0; 0,05; 0,10; 0,25; 0,50; 1,00};
  - tiêu chí `row_weighted_system` (`CriterionAdaptiveSquareRootBackend`) với
    ngân sách {0,05; 0,10; 0,25; 0,50}.
- Một unit benchmark ở luồng `s2025`: `CriterionAdaptiveSquareRootBackend` với
  tiêu chí `factor_mse`, ngân sách 0,25, kèm hook chỉ đọc đo thời gian chấm
  điểm từng block và theo lô trên đúng thừa số thật trước khi lượng tử hóa.

## Định nghĩa tiêu chí

Với thừa số chưa lượng tử hóa `R`, sai số lượng tử hóa `E` và block `b`, gọi
`e8_ij`, `e16_ij` là bình phương sai số của phần tử `(i, j)` khi mã hóa INT8 và
FP16; `c_b` là số byte tăng thêm khi nâng block lên FP16.

- `factor_mse`: lợi ích `g_b = Σ_{(i,j)∈b} (e8_ij − e16_ij)`.
- `row_weighted_system`: lợi ích `g_b = Σ_{(i,j)∈b} w_i (e8_ij − e16_ij)` với
  `w_i = ||R_{i·}||²`. Nếu các hàng của `E` độc lập và có kỳ vọng 0 thì
  `E||RᵀE||_F² = Σ_i ||R_{i·}||² E||E_{i·}||²`, nên `g_b` là mức giảm kỳ vọng
  của số hạng bậc nhất trong sai số hệ theo tác vụ `D_t = RᵀE + EᵀR + EᵀE`.

Cả hai tiêu chí dùng cùng quy tắc tham lam đã khóa: sắp xếp theo `g_b / c_b`
giảm dần, hòa thì theo chỉ số block, bỏ qua block có `g_b ≤ 0` hoặc `c_b ≤ 0`,
chọn nếu còn vừa ngân sách byte `floor(β · (byte toàn FP16 − byte toàn INT8))`.
Không tiêu chí nào dùng nhãn, logit, dự đoán hay độ chính xác.

## Đại lượng ghi lại

Theo từng tác vụ và từng unit: độ chính xác validation trên các lớp đã xuất
hiện, trạng thái lưu bền vững (byte), byte của thừa số, trần ngân sách, byte bổ
sung đã dùng, số block FP16, sai số thừa số tương đối, sai số bộ phân loại tương
đối `||Ŵ_t − W_t||_F / ||W_t||_F` so với Exact cùng luồng, residual của bộ giải,
thời gian cập nhật và SHA-256 của mask độ chính xác và trọng số.

Unit benchmark còn ghi, theo tác vụ: thời gian chấm điểm từng block, thời gian
chấm điểm theo lô, quyết định chọn có trùng không, số block khác biệt và chênh
lệch tương đối lớn nhất của lợi ích.

## Gate cấu trúc

Không có gate độ chính xác (`accuracy_gate = null`). Trạng thái là
`PASS_M20_ADAPTIVE_BUDGET_CRITERION_TRAIN_ONLY` nếu mọi gate sau đạt, ngược lại
là `COMPLETE_M20_ADAPTIVE_BUDGET_CRITERION_WITH_WARNINGS`; artifact luôn được
xuất đầy đủ.

1. đủ mọi unit (37 unit);
2. luồng `s2025` khớp danh tính M6;
3. Exact và P2B trên `s2025` tái lập M6: chênh lệch AIA và độ chính xác cuối
   không quá `10^-6` điểm, byte trạng thái bằng nhau;
4. mọi unit adaptive nằm trong trần ngân sách byte;
5. trạng thái cuối không giảm khi ngân sách tăng, với từng tiêu chí và từng luồng;
6. unit benchmark (tiêu chí `factor_mse` của backend mới) cho cùng mask độ chính
   xác, cùng độ chính xác và cùng byte ở mọi tác vụ với backend khóa ngân sách
   0,25 trên cùng luồng;
7. chấm điểm theo lô cho đúng cùng quyết định chọn ở mọi tác vụ;
8. residual của bộ giải không quá `2·10^-5`;
9. mọi unit chạy trên Tesla T4.

## Quy tắc diễn giải đã khóa

- **Đường cong ngân sách** (tiêu chí `factor_mse`): báo cáo trung bình, độ lệch
  chuẩn, nhỏ nhất và lớn nhất trên ba luồng cho mức giảm AIA so với Exact, sai số
  bộ phân loại cuối và bộ nhớ. "Ngân sách đủ nhỏ nhất" là ngân sách nhỏ nhất có
  mức giảm AIA không quá 0,01 điểm trên cả ba luồng; đây chỉ là đại lượng mô tả
  trên validation. Bài báo có thể thêm đường cong này vào phụ lục bất kể hình
  dạng, kèm ghi chú rằng 25% được khóa trước và M20 chạy sau, trên validation.
  Không kết quả nào của M20 được dùng để đổi ngân sách của các kết quả test.
- **Tiêu chí theo sai số hệ** được xem là *được ủng hộ* chỉ khi, ở cả hai ngân
  sách 0,05 và 0,10, sai số bộ phân loại cuối của nó thấp hơn tiêu chí
  `factor_mse` trên cả ba luồng, và trung bình mức giảm AIA không cao hơn. Kết
  quả nào cũng phải báo cáo. Tiêu chí này là biến thể phương pháp mới; nó không
  được đưa vào các tuyên bố chính của bản nộp hiện tại, chỉ dùng cho rebuttal,
  camera-ready hoặc công trình tiếp theo.
- **Chấm điểm theo lô**: chỉ được mô tả là "cho cùng quyết định" nếu gate 7 đạt;
  tốc độ báo cáo bằng trung vị tỷ số thời gian trên các tác vụ.

## Chạy tiếp và artifact

Mỗi unit là một checkpoint nguyên tử trong `output/units/`; chạy lại cùng lệnh sẽ
bỏ qua unit đã xong. Unit Exact của mỗi luồng ghi trọng số bộ phân loại theo tác
vụ vào `output/cache/` để các unit khác cùng luồng tính sai số bộ phân loại. Thư
mục `cache/` là trạng thái học (không chứa dữ liệu mức mẫu), không được đưa vào
artifact xuất. Artifact `srq_generalization_m20_adaptive_budget_criterion_t4.zip`
gồm các unit JSON, `m20_results.json`, `m20_budget_curve.csv`, config và
protocol này.

Thời gian ước tính trên T4: khoảng 10 phút trích đặc trưng và 45–60 phút cho 37
unit.
