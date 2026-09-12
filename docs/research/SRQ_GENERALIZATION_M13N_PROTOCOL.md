# M13-N — kiểm toán số cho LoRanPAC task 1

## 1. Câu hỏi duy nhất

M13 đã hoàn thành đủ bốn đơn vị width/budget nhưng vẫn mang trạng thái
`FAIL_M13_LORANPAC_TRAIN_ONLY`, vì hai engineering gate đạt giá trị xấu nhất ở
task 1. M13-N chỉ trả lời câu hỏi:

> Sai lệch task 1 đến từ adapter/toán học sai, hay chủ yếu từ độ trực chuẩn hữu
> hạn của các singular vector CUDA/FP32 khi raw Frobenius norm cộng sai số qua
> hàng nghìn cột?

M13-N không so sánh chất lượng dự đoán, không chọn phương pháp và không hợp
thức hóa lại M13 sau khi đã thấy kết quả.

## 2. Ranh giới dữ liệu và provenance

- Chỉ `train.pt` được phép hiện diện; runner từ chối chạy nếu thấy `test.pt`.
- Không tính accuracy, không gọi prediction và không có accuracy gate.
- Artifact M13 được xác minh bằng SHA-256 trên raw bytes nhưng **không được mở
  container** và không đọc JSON bên trong. Bốn rank cùng các số task-1 dùng để
  đối chiếu đã được khóa độc lập trong config trước khi chạy.
- Train cache, checkpoint, seed thứ tự lớp và hash của train/validation indices
  phải khớp M13.
- M13 vẫn được công bố là `FAIL_M13_LORANPAC_TRAIN_ONLY` bất kể M13-N pass hay
  fail.

## 3. Phép đo chính

Với basis task 1 (U\in\mathbb{R}^{d\times r}), đặt

\[
E=U^\top U-I.
\]

M13-N báo bốn đại lượng:

1. raw Frobenius (\lVert E\rVert_F), giữ để đối chiếu đúng gate M13;
2. Frobenius chuẩn hóa (\lVert E\rVert_F/\sqrt r), phản ánh RMS lỗi theo
   phương trực chuẩn;
3. spectral norm (\lVert E\rVert_2), tính chính xác bằng `eigvalsh` trên ma
   trận đối xứng;
4. phần tử tuyệt đối lớn nhất của (E).

Hai residual của projected solve được đo với Ridge chính thức bằng 0 và Ridge
đã khóa từ M6 bằng (10^6). Runner sau đó QR lại cùng basis và đo lại toàn bộ
đại lượng. Đường QR chỉ là một **can thiệp chẩn đoán**: QR có thể thay đổi hệ
truncated được biểu diễn khi singular values vẫn giữ nguyên, nên kết quả này
không được báo như một biến thể LoRanPAC/SRQ mới.

Để tách ảnh hưởng bài toán lớn CUDA khỏi lỗi công thức, một oracle nhỏ cố định
chạy cùng phép toán ở CPU FP32 và FP64. Không có siêu tham số nào được chọn từ
kết quả oracle.

## 4. Gate đã khóa

M13-N pass khi và chỉ khi:

- raw artifact SHA, train identity và cả bốn rank contract đều khớp;
- mọi numerical metric hữu hạn;
- QR chẩn đoán cải thiện raw Frobenius ít nhất 10 lần ở mọi đơn vị;
- QR chẩn đoán cải thiện residual xấu nhất ít nhất 10 lần ở mọi đơn vị;
- normalized orthogonality của oracle FP64 không quá (10^{-10}).

Các tỷ lệ so với số M13 gốc chỉ dùng để kiểm tra khả năng tái lập giữa phần
cứng/phần mềm; chúng không phải gate, vì CUDA SVD không được giả định
bitwise-identical giữa mọi GPU.

## 5. Cách diễn giải và bước kế tiếp

- Nếu M13-N pass, bằng chứng phù hợp với chẩn đoán rằng gate raw Frobenius của
  M13 không scale-aware và residual task 1 chủ yếu đi cùng sai số trực chuẩn
  FP32. Khi đó M14 mới được phép đăng ký trước một protocol multi-seed dùng cả
  raw, normalized và spectral metrics.
- Nếu M13-N fail ở identity/rank, dừng vì phép tái lập không cùng thí nghiệm.
- Nếu identity/rank pass nhưng QR không cải thiện residual, phải kiểm tra lại
  công thức projected solve/adapter trước M14.
- Nếu oracle FP64 fail, phải kiểm tra chính implementation metric; không được
  nới threshold.

M13-N không cung cấp bằng chứng rằng QR nên được thêm vào LoRanPAC, không chứng
minh accuracy-state trade-off bền qua seed và cũng không mở quyền truy cập test.

## 6. Recovery disclosure trước phép đo

Lần chạy đầu tiên dừng tại `pre_svd_train_identity_check`, trước khi in
`M13-N SVD START` và trước khi tạo bất kỳ numerical metric nào. Kiểm tra cho
thấy raw SHA của `train.pt`, checkpoint, feature dimension và class inventory
đều khớp, nhưng hai split hash đã điền trong config không khớp kết quả tất định
của chính cache, seed và hàm split đã khóa. Các hash đúng được tái lập là:

- training indices: `ff06d5687dc8069c599b73c29c435cd84e2be160fc878fcfde8e37a565f326c9`;
- validation indices: `979e3bea647ed5f51b8a354c8adb13be9fb82739a7773e3c941080bd848cb313`.

Recovery chỉ sửa hai expected identity hash này và thêm disclosure vào output.
Không thay source artifact, cache, seed, rank, Ridge, metric, threshold, method
hay gate. Vì chưa quan sát kết quả số trước recovery, thay đổi này sửa một lỗi
preflight có thể kiểm chứng độc lập, không phải điều chỉnh theo kết quả.
