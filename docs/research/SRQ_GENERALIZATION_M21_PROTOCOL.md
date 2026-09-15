# M21 — đối chứng tích lũy sai số và tải đường chéo tối thiểu cho INT8 Gram (train-only)

## Câu hỏi

M21 kiểm tra hai luận điểm của bài báo mà một reviewer có thể chất vấn. Cả hai
đều chạy trên dữ liệu huấn luyện; không mở tập test.

1. **Tải tối thiểu cho INT8 Gram (Mục 5.3, Bảng 2).** Bài báo sửa ma trận Gram
   INT8 bằng tải đường chéo nhỏ nhất mà bất đẳng thức Weyl chứng nhận được. Cận
   `||N||_∞` rất bảo thủ, nên tải tăng lên 113–5.148 lần `λ` và AIA giảm 4,9
   điểm. Nếu chỉ thêm tải nhỏ nhất đủ để phân rã Cholesky FP32 thành công (không
   chứng nhận), INT8 Gram có theo kịp SRQ-INT8 không?
2. **Tích lũy sai số (Mục 4 và 5.4).** Bài báo nói sai số lượng tử hóa được mang
   qua các tác vụ và điều này giới hạn INT8 ở chiều rộng lớn. Nhưng sai số logit
   cũng có thể tăng chỉ vì hệ ngày càng xấu điều kiện, dù không có tích lũy. Nếu
   ở mỗi tác vụ ta lượng tử hóa **một lần** thừa số chính xác của hệ hiện tại
   (bằng cùng codec), sai số logit cuối nhỏ hơn bản đệ quy của SRQ bao nhiêu?

M21 không thay đổi bất kỳ kết quả test nào đã báo cáo, và không kết quả nào của
M21 được dùng để chọn phương pháp, tải, seed hay siêu tham số.

## Thiết lập chung khóa trước khi chạy

- CIFAR-100, 10 tác vụ, backbone ViT-B/16 đóng băng (checkpoint `32aa17d6…`),
  cache `train.pt` có SHA-256 `ba53b821…`; `test.pt` bị cấm.
- FP32 cho thống kê và bộ giải; TF32 tắt, `float32_matmul_precision=highest`.
- Mọi unit chạy trên Tesla T4.

## Phần 1 — tải tối thiểu cho INT8 Gram (FLY-CL)

**Luồng.** Đúng luồng của Priority 3 (config
`configs/srq_fly_priority3_direct_control_cifar100_train_only.json`, SHA-256
`909fbd4d…`): seed 2025, 20% validation mỗi lớp, FLY-CL chiều rộng 10.000, bậc
synapse 300, WTA ρ = 0,3, `λ = 10^6`, block 256, nhóm 64.

**Các nhánh (mỗi nhánh một unit).**

1. `exact_fly_10000`, `srq_int8_p2b`, `direct_int8_gram_weyl_repair`: chạy lại
   bằng đúng các vòng lặp của Priority 3 và phải tái lập AIA của artifact
   `srq_fly_priority3_direct_control_train_only.zip` (`9c5f8c9c…`).
2. `direct_int8_gram_minimal_load` (mới, `MinimalLoadDirectInt8GramLearner`): ma
   trận Gram, cách lượng tử hóa INT8 và phép giải giống hệt nhánh Weyl; chỉ tải
   đường chéo khác.

**Quy tắc chọn tải (khóa).** Ở tác vụ `t`, gọi `S̃_t` là ma trận Gram INT8 đã
giải mã và `μ_t = 8 ε m max{||diag(S̃_t)||_∞, λ, 1}` là biên số học cố định của
nhánh Weyl (`ε` là epsilon FP32). Tải `δ_t` là phần tử nhỏ nhất của lưới
`{0} ∪ {μ_t · 2^{j/4} : j = 0, 1, 2, …}` sao cho phân rã Cholesky FP32 của
`(S̃_t + (λ + δ_t) I)` (sau khi đối xứng hóa, như bộ giải) thành công. Thuật toán:
thử `δ = 0`; nếu thất bại thì nhân đôi tải cho đến khi thành công, rồi chia đôi
chỉ số lưới giữa lần thất bại cuối và lần thành công đầu. Quy tắc giả định rằng
khả năng phân rã thành công tăng đơn điệu theo tải; mọi lần thử đều được đếm và
ghi lại. Tải không lưu dồn qua tác vụ, không dùng nhãn, logit hay độ chính xác,
và không có chứng nhận. Nếu tải vượt `10^6 λ`, unit thất bại. Trạng thái lưu gồm
Gram INT8 và một số vô hướng `δ_t` (8 byte).

**Đại lượng ghi lại theo tác vụ.** Độ chính xác validation (trung bình theo tác
vụ đã thấy, như Priority 3), `δ_t / λ`, chỉ số lưới, số lần thử Cholesky,
`||N_t||_∞`, sai số lưu trữ tương đối, residual của bộ giải, byte trạng thái,
thời gian cập nhật.

**Quy tắc diễn giải (khóa).** Gọi `gap = AIA(SRQ-INT8) − AIA(tải tối thiểu)`.

| Điều kiện | Kết luận | Hệ quả cho bài báo |
|---|---|---|
| `gap ≥ 0,5` | `SQUARE_ROOT_ADVANTAGE_MATERIAL_AGAINST_MINIMAL_LOAD` | Thêm một câu vào Mục 5.3: ngay cả tải nhỏ nhất đủ cho Cholesky vẫn làm mất X điểm. |
| `0,1 < gap < 0,5` | `SQUARE_ROOT_ADVANTAGE_MODEST_AGAINST_MINIMAL_LOAD` | Nêu cả hai: SRQ vẫn hơn nhưng khoảng cách nhỏ hơn nhiều so với 4,9 điểm. |
| `|gap| ≤ 0,1` | `MINIMAL_LOAD_PRACTICALLY_EQUIVALENT_TO_SQUARE_ROOT` | Viết lại Mục 5.3: lợi thế của SRQ là tính xác định dương được bảo đảm và không cần dò tải, không phải độ chính xác. |
| `gap < −0,1` | `MINIMAL_LOAD_OUTPERFORMS_SQUARE_ROOT` | Như trên, và nêu rõ tải tối thiểu tốt hơn trên luồng này. |

Residual của nhánh tải tối thiểu được ghi lại nhưng không có gate, vì theo cách
chọn, hệ nằm sát ngưỡng xác định dương về mặt số học; residual lớn là một phần
của cách làm này.

## Phần 2 — tích lũy sai số (RanPAC)

**Luồng.** Head RanPAC phase-2 (phép chiếu Gauss + ReLU, không PETL), `λ = 10^6`,
20% validation, P2B: block 256, nhóm 64, panel QR 128, bước đầu Gram-Cholesky,
lượng tử hóa streaming lô 64. Bốn unit:

| Unit | Chiều rộng | Luồng | Tham chiếu tái lập |
|---|---|---|---|
| `acc_w20000_s2025` | 20.000 | seed 2025 (khóa danh tính M6) | M6, M20, M7 |
| `acc_w20000_s2026` | 20.000 | seed 2026 | M20 |
| `acc_w20000_s2027` | 20.000 | seed 2027 | M20 |
| `acc_w10000_s2025` | 10.000 | seed 2025 (khóa danh tính M6) | M6, M7 |

**Ba nhánh chạy song song trong mỗi unit, cùng một luồng dữ liệu.**

1. `exact`: `ExactGramBackend`.
2. `recursive_int8`: `SquareRootBackend` INT8 đã khóa (chính là SRQ-INT8). Thừa
   số lưu trữ mang sai số lượng tử hóa của mọi tác vụ trước:
   `Â_t = Â_{t−1} + Φ_tᵀΦ_t + D_t`.
3. `one_shot_int8` (đối chứng giả định): sau khi cập nhật Exact, tính thừa số
   Cholesky chính xác `R_t` của `A_t = λI + Σ_{k≤t} Φ_kᵀΦ_k` (đúng các phép tính
   của Exact), lượng tử hóa **một lần** bằng đúng codec của SRQ, rồi giải hai hệ
   tam giác với `B_t` chính xác. Sai số của nhánh này chỉ gồm sai số lượng tử hóa
   của tác vụ hiện tại. Nhánh cần thống kê chính xác nên chỉ là đại lượng chẩn
   đoán, không phải phương pháp.

Ở tác vụ 1, nhánh 2 và 3 lượng tử hóa cùng một ma trận bằng cùng phép tính, nên
trọng số phải trùng nhau; khác biệt chỉ bắt đầu từ tác vụ 2.

**Đại lượng ghi lại theo tác vụ** (so với Exact cùng tác vụ, trên validation của
các lớp đã thấy). Mọi sai số tương đối dùng chuẩn Frobenius và **không** chặn
dưới mẫu số (khác chỉ số `relative_weight_error` của M7):

- sai số logit tương đối `||L̂_t − L_t|| / ||L_t||`;
- sai số hệ tương đối `||(Â_t − A_t) Z|| / ||A_t Z||` với 16 vector dấu cố định
  của M7 (seed 7071);
- sai số bộ phân loại tương đối `||Ŵ_t − W_t|| / ||W_t||`;
- sai số thừa số cục bộ tương đối, tỷ lệ dự đoán trùng Exact, tỷ lệ mẫu thỏa
  điều kiện biên `2||ℓ̂ − ℓ||_∞ < γ(x)`, độ chính xác, residual;
- byte trạng thái và thời gian cập nhật của nhánh đệ quy.

**Quy tắc diễn giải (khóa).** Chỉ tiêu chính: tỷ số
`ρ = sai số logit cuối (đệ quy) / sai số logit cuối (một lần)` ở chiều rộng
20.000, trên cả ba luồng.

| Điều kiện | Kết luận về sai số logit |
|---|---|
| `min ρ ≥ 1,5` | `ACCUMULATION_DOMINATES_FINAL_LOGIT_ERROR` |
| `1,1 ≤ min ρ < 1,5` | `ACCUMULATION_CONTRIBUTES_TO_FINAL_LOGIT_ERROR` |
| `min ρ < 1,1` | `ACCUMULATION_NOT_SUPPORTED_BY_FINAL_LOGIT_ERROR` |

Kết luận phụ về độ chính xác, với `g = AIA(một lần) − AIA(đệ quy)`:
`ONE_SHOT_HIGHER_AIA_ON_EVERY_STREAM` nếu `g > 0` trên cả ba luồng;
`ONE_SHOT_HIGHER_MEAN_AIA_NOT_EVERY_STREAM` nếu chỉ trung bình dương;
`ACCUMULATION_ACCURACY_COST_NOT_SUPPORTED` nếu trung bình không dương. Unit chiều
rộng 10.000 chỉ để mô tả, không vào quy tắc.

**Hệ quả cho bài báo (khóa trước).**

- *Dominates* hoặc *contributes*: giữ luận điểm tích lũy, thêm một câu định
  lượng vào Mục 5.4 (ví dụ: lượng tử hóa thừa số chính xác một lần cho sai số
  logit cuối nhỏ hơn `ρ` lần) và nêu mức tăng sai số logit của nhánh một lần,
  tức phần do điều kiện hóa.
- *Not supported*: bỏ các câu nói tích lũy là nguyên nhân (tóm tắt, Mục 4 cuối,
  Mục 5.4); mô tả mức tăng sai số logit như hệ quả của việc hệ xấu điều kiện dần
  và của sai số lượng tử hóa ở từng tác vụ.
- Báo cáo mọi kết quả, kể cả khi bất lợi.

## Gate cấu trúc

Không có gate độ chính xác (`accuracy_gate = null`). Trạng thái là
`PASS_M21_ACCUMULATION_GRAM_LOAD_TRAIN_ONLY` nếu mọi gate sau đạt, ngược lại là
`COMPLETE_M21_ACCUMULATION_GRAM_LOAD_WITH_WARNINGS`; artifact luôn được xuất.

1. đủ 8 unit;
2. ba nhánh Priority 3 tái lập AIA của artifact Priority 3 (chênh lệch không quá
   `10^-6` điểm);
3. residual của ba nhánh đó không quá `2·10^-5`;
4. nhánh tải tối thiểu hoàn tất 10 tác vụ, mỗi tác vụ có ít nhất một lần thử và
   tải không âm;
5. luồng `s2025` khớp danh tính M6 (thứ tự lớp, chỉ số, phép chiếu) ở cả hai
   chiều rộng;
6. nhánh Exact và nhánh đệ quy tái lập M6/M20: AIA và độ chính xác cuối chênh
   không quá `10^-6` điểm, byte trạng thái bằng nhau;
7. sai số logit của nhánh đệ quy trên `s2025` khớp đường cong M7 (chênh lệch
   tương đối không quá `10^-3` ở mọi tác vụ);
8. ở tác vụ 1, trọng số nhánh một lần trùng nhánh đệ quy (chênh lệch tương đối
   không quá `10^-6`);
9. residual của mọi nhánh trong Phần 2 không quá `2·10^-5`;
10. mọi unit chạy trên Tesla T4.

Gate 2, 6, 7 kiểm tra rằng môi trường Colab hiện tại cho cùng số như các lần chạy
trước. Nếu một gate trong số này không đạt (ví dụ vì phiên bản thư viện khác),
kết luận vẫn được tính trên các nhánh chạy trong cùng unit, nhưng phải báo cáo
chênh lệch.

## Chạy tiếp và artifact

Mỗi unit là một checkpoint nguyên tử trong `output/units/`; chạy lại cùng lệnh sẽ
bỏ qua unit đã xong. Cache mã WTA và cache đặc trưng là hạ tầng thí nghiệm (dữ
liệu mức mẫu), không được đưa vào artifact. Artifact
`srq_generalization_m21_accumulation_gram_load_t4.zip` gồm các unit JSON,
`m21_results.json`, config và protocol này.

Thời gian ước tính trên T4: khoảng 10 phút trích đặc trưng, vài phút dựng cache
mã WTA, khoảng 5 phút cho Phần 1 và 15–25 phút cho Phần 2.
