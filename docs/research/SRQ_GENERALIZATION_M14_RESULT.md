# M14 — xác nhận đa seed cho đối chứng LoRanPAC cùng byte

## 1. Artifact và trạng thái

- Artifact: `srq_generalization_m14_loranpac_multiseed_train_only.zip`
- SHA-256 toàn ZIP:
  `4beb726bf7f29f8569e5c6de630d90284abdf7ae53928471c27bba178d148b0c`
- SHA-256 của `m14_results.json`:
  `cf4233b0363a2bfbfb6279428a0d28de492d1c85dbae8f75dc4b4b6f7cf19f86`
- Source commit của lần chạy: `2cf2093e407a54022d4d3e6de0ce6dd15a3cd1fe`
- Trạng thái bất biến: `FAIL_M14_LORANPAC_MULTISEED_TRAIN_ONLY`
- Phạm vi: train-only; `uses_test_set=false`; accuracy không phải completion gate.

Artifact hợp lệ và hoàn thành đủ 60/60 unit: sáu seed ghép cặp `4101`--`4106`,
hai width `10.000/20.000` và năm phương pháp trên mỗi seed/width. Không unit nào
bị crash hoặc bị loại khỏi tổng hợp.

## 2. Gate và nguyên nhân formal FAIL

Mười trên mười một gate đạt. Gate duy nhất thất bại là residual của projected
solver LoRanPAC ở task đầu:

| Đại lượng | Quan sát | Ngưỡng khóa trước | Kết quả |
|---|---:|---:|---|
| `max ||U^TU-I||_F/sqrt(rank)` | 0,0012783 | 0,0015 | Đạt |
| `max task-1 ||U^TU-I||_2` | 0,0034144 | 0,0035 | Đạt |
| `max task-1 solver relative residual` | 0,0019148 | 0,0010 | **Không đạt** |
| `max task-2--10 solver relative residual` | 4,392e-6 | 2,0e-5 | Đạt |

Cực đại đến từ seed `4105`, width `10.000`, LoRanPAC tại ngân sách P2B:
residual matched-Ridge là `0,0019131891`, còn residual Ridge-0 chính thức là
`0,0019147568`. Cùng unit này vẫn đạt gate trực giao chuẩn hóa
(`0,0011953`) và spectral (`0,0023147`). Vì ngưỡng solver `0,001` đã được khóa
trước khi chạy, không được nâng ngưỡng hoặc bỏ seed 4105 sau khi xem kết quả.

## 3. Kết quả accuracy--state mô tả

Các số dưới đây là mean trên đủ sáu seed. Vì formal status là FAIL, chúng là
bằng chứng mô tả đa seed, không phải một confirmation đã vượt toàn bộ gate.

| Width | Phương pháp | Validation AIA | Final | State (MiB) | Update (s) |
|---:|---|---:|---:|---:|---:|
| 10k | Exact | 92,7534 | 88,9100 | 418,40 | 4,77 |
| 10k | P2B | 92,6315 | 88,6683 | 87,62 | 8,77 |
| 10k | Adaptive | 92,7527 | 88,9233 | 98,80 | 13,74 |
| 10k | LoRanPAC, P2B budget | 92,1279 | 87,6533 | 87,59 | 204,67 |
| 10k | LoRanPAC, adaptive budget | 92,2272 | 87,8517 | 98,77 | 279,37 |
| 20k | Exact | 93,0042 | 89,3500 | 1.599,73 | 18,96 |
| 20k | P2B | 92,7088 | 88,8550 | 276,58 | 33,19 |
| 20k | Adaptive | 93,0089 | 89,3500 | 321,28 | 53,95 |
| 20k | LoRanPAC, P2B budget | 92,6160 | 88,5617 | 276,50 | 371,67 |
| 20k | LoRanPAC, adaptive budget | 92,6990 | 88,7383 | 321,21 | 589,34 |

P2B giảm state so với Exact khoảng 79,1% ở width 10k và 82,7% ở width 20k.
Adaptive giảm khoảng 76,4% và 79,9%, đồng thời gần như khớp Exact về AIA.
Tại cùng byte, AIA trung bình của SRQ cao hơn LoRanPAC ở cả bốn so sánh, nhưng
độ bền theo seed yếu nhất ở width 20k/P2B budget: SRQ chỉ cao hơn ở bốn trên
sáu seed. Không được diễn giải các số này thành bằng chứng LoRanPAC end-to-end.

## 4. Quyết định tiếp theo

M14 được giữ nguyên là formal FAIL. M15 là audit train-only, không predictive,
chỉ tái dựng task 1 của seed lỗi `4105` và một seed sentinel `4101`. M15 phải
phân biệt sai số trực giao FP32 với lỗi solver/adapter bằng đánh giá FP64 trên
cùng factor và QR có biến đổi core bảo toàn hệ. M15 không được sửa prediction,
không được nới gate M14 và không được biến kết quả chẩn đoán thành một phương
pháp LoRanPAC mới.
