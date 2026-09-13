# M16 — xác nhận Stanford Cars/ResNet-50 cho RanPAC Phase-2

## 1. Artifact và tính toàn vẹn

- Artifact: `srq_generalization_m16_cars_phase2_locked.zip`
- SHA-256 toàn ZIP:
  `9d904939c2e3dfcb1ee62d2e39223508833f6855d2eb77aab3566a2dcb6b653a`
- SHA-256 của `m16_results.json`:
  `551c26c91b9ad66f0bf524392c30021fc97a482201459cdc6c06e0ed38857cc5`
- Source commit của lần chạy:
  `30d78a40eb13736f9de0ced29979a9e6c3156078`
- Trạng thái: `PASS_M16_CARS_RANPAC_PHASE2_LOCKED`
- Phạm vi: official Stanford Cars train/test split, sáu replicate ghép cặp,
  ResNet-50 ImageNet-1K V2 và random-ReLU width 10.000.

ZIP có 24 member: một manifest và 23 file được manifest khai báo. Toàn bộ tên,
kích thước và SHA-256 của 23 file đều khớp. Artifact chứa đủ 18/18 unit, tương
ứng sáu seed `1993--1998` và ba backend Exact, P2B, Adaptive; không unit nào bị
loại khỏi tổng hợp.

## 2. Ranh giới train/test và lựa chọn Ridge

M16 chọn Ridge bằng MSE trên split 80/20 của task train đầu tiên, rồi khóa hệ số
trước khi tạo test-feature cache. Selection ID và authorization ID trong kết quả
khớp các record gốc. Cả sáu replicate đều chọn `lambda=100000`. Đây không phải
một giá trị được chọn từ test accuracy.

Các thuộc tính kiểm soát trong artifact:

- `test_tuning_allowed=false`;
- `accuracy_based_method_selection=false`;
- `allow_post_test_retry=false`;
- `accuracy_gate=null`.

Do đó, accuracy chỉ là kết quả báo cáo; nó không quyết định PASS/FAIL và không
được dùng để đổi backend sau khi xem test.

## 3. Kết quả chính

Các giá trị là mean ± sample standard deviation trên sáu replicate. Chênh lệch
accuracy được ghép cặp với Exact theo cùng class order, projection và Ridge.

| Backend | AIA (%) | Final (%) | ΔAIA so với Exact (điểm) | ΔFinal (điểm) | State (MiB) | Giảm state | Update/Exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact | 59,3517 ± 0,8116 | 48,5574 ± 0,2595 | 0 | 0 | 474,55 | — | 1,00× |
| P2B INT8/FP32 | 59,2855 ± 0,7964 | 48,4351 ± 0,2728 | −0,0663 ± 0,0594 | −0,1223 ± 0,0780 | 143,78 | 69,70% | 1,58× |
| Adaptive INT8/FP16 | 59,3511 ± 0,8143 | 48,5450 ± 0,2759 | −0,0006 ± 0,0076 | −0,0124 ± 0,0352 | 154,95 | 67,35% | 3,64× |

Residual cực đại lần lượt là `8,3592e-6`, `6,3287e-6` và `6,0676e-6`, đều nhỏ
hơn gate khóa trước `2e-5`. Cả bốn gate hoàn thành unit, residual, state nén nhỏ
hơn Exact và không dùng accuracy gate đều đạt.

## 4. Diễn giải khoa học

M16 bổ sung một thay đổi đồng thời về dataset và backbone so với dòng phát triển
CIFAR/ViT. P2B tạo điểm nén nhỏ hơn và nhanh hơn Adaptive, nhưng mất trung bình
0,0663 điểm AIA. Adaptive dùng thêm khoảng 11,17 MiB, gần như khớp Exact về AIA
và final accuracy, nhưng analytic update chậm 3,64 lần. Sai khác AIA
`−0,0006` không phải bằng chứng Adaptive cải thiện accuracy; nó là bằng chứng
bảo toàn accuracy trong thiết lập này.

Kết quả cho phép nói SRQ đã được xác nhận như một backend dùng lại được trên
FLY và trên một frontend random-ReLU kiểu RanPAC, qua hai backbone và một dataset
ngoài dòng phát triển. Nó củng cố tính khả chuyển của backend trong họ hệ Ridge
cộng dồn với đặc trưng cố định.

## 5. Giới hạn tuyên bố

M16 không tái lập RanPAC end-to-end và không tái lập số accuracy công bố. Nó chỉ
kiểm tra đường Phase-2 không PETL với backbone đóng băng. RanPAC chính thức có thể
chọn lại Ridge theo task; M16 khóa một Ridge sau lựa chọn train-only để mọi backend
duy trì cùng một hệ cộng dồn. Dataset được tải qua một Kaggle mirror theo thư mục
lớp của official split. Vì vậy, không được gọi SRQ là “universal plug-in” hoặc
suy rộng kết quả sang representation đang thay đổi, mọi analytic learner hay mọi
dataset/backbone.
