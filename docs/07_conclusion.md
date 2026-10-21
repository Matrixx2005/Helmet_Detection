# 7. Kết luận

## 7.1. Kết quả đạt được

1. **Pipeline hoàn chỉnh** Ảnh → YOLO26n → ghép người–mũ → hệ luật → ảnh kết
   quả + CSV/JSON + biểu đồ, chạy được trên CPU (~32–45 ms suy luận/ảnh).
2. **Thuật toán ghép** dựa trên vị trí tương đối, hiệu chỉnh từ nhãn chuẩn của
   Construction-PPE, ghép một-một toàn cục; xử lý đúng 8 tình huống bắt buộc
   (kiểm chứng bằng test tự động).
3. **Hệ luật thận trọng R0–R5**: chỉ kết luận nghi vi phạm khi có bằng chứng
   tích cực (`no_helmet`), mỗi kết luận kèm mã luật, lý do và số liệu bằng chứng.
4. **Phát hiện và xử lý một vấn đề thật của mô hình**: lớp `no_helmet` gần như
   không được phát hiện ở ngưỡng mặc định (recall = 0). Dùng ngưỡng riêng theo lớp
   (dò trên val) kết hợp association làm bộ lọc nhiễu đưa F1 lớp vi phạm trên tập
   test từ **không xác định (recall 0)** lên **0,815**, cao hơn luật đối chứng
   ngây thơ (0,727) — xem [06_results.md](06_results.md).
5. **Đánh giá trung thực**: tham số chọn trên val, báo cáo trên test; ảnh không
   có nhãn chỉ được thống kê, không gắn độ chính xác.
6. **Sản phẩm dùng được**: CLI, xử lý hàng loạt chịu lỗi từng ảnh, giao diện
   Streamlit tiếng Việt, 53 test tự động, tài liệu, script `.bat` một-nhấp.

## 7.2. Giới hạn của hệ thống

* **Phụ thuộc vào chất lượng phát hiện lớp `no_helmet`** (mAP50 test chỉ 0,11).
  Hệ thống phải dùng ngưỡng 0,03 cho lớp này; độ tin cậy hiển thị trên các ca
  "nghi không đội mũ" vì vậy thấp (0,03–0,2) dù kết luận thường đúng.
* **Nhãn chuẩn cấp người được suy ra tự động** bằng chính quy tắc ghép: con số
  đánh giá đo ảnh hưởng của lỗi phát hiện, chưa phải kiểm chứng độc lập của quy
  tắc ghép. Cần gán nhãn tay một tập nhỏ để có kết luận chắc chắn.
* **Tập test nhỏ** (40 người vi phạm) → khoảng tin cậy rộng; ±1 ca thay đổi
  precision/recall khoảng 2,5 điểm phần trăm. Ngoài ra 8/141 ảnh test gần trùng
  ảnh train nên kết quả có thể lạc quan đôi chút.
* **Bỏ sót người** (43/236 trên test, phần lớn là người nhỏ ở hậu cảnh); người bị
  bỏ sót thì không thể được kết luận.
* **Chưa có ảnh công trường độc lập**: 29 ảnh trong `data/raw` hóa ra đều là ảnh
  của Construction-PPE (5 ảnh thuộc tập huấn luyện), nên chưa đánh giá được khả
  năng tổng quát hóa ra ảnh mới.
* **Miền dữ liệu**: mũ bảo hiểm xe đạp, ảnh đêm, ảnh xoay 90° gây lỗi; ngưỡng
  `conf_person = 0,4` làm giảm số người phát hiện trong ảnh đám đông xa camera.
* Chỉ xử lý **ảnh tĩnh**, không theo dõi người qua các khung hình video.

## 7.3. Hướng cải tiến

1. **Cải thiện lớp `no_helmet`**: huấn luyện lại với trọng số lớp / oversampling
   ảnh có `no_helmet`, bổ sung dữ liệu (ví dụ SHWD, Hard Hat Workers), hoặc dùng
   mô hình lớn hơn (YOLO26s/m) — đây là nút thắt lớn nhất.
2. **Bộ phân loại vùng đầu**: cắt vùng đầu ước lượng của mỗi người và phân loại
   có/không mũ bằng một CNN nhỏ, thay vì chỉ dựa vào hộp `no_helmet`.
3. **Thu thập ảnh công trường mới** (không thuộc Construction-PPE) và **gán nhãn
   tay cấp người** (định dạng CSV đã hỗ trợ) để đánh giá độc lập.
4. **Video**: thêm tracking (ByteTrack có sẵn trong Ultralytics) và luật theo thời
   gian (vi phạm kéo dài N khung hình mới cảnh báo) để giảm báo nhầm.
5. Hiệu chỉnh ngưỡng theo từng camera/công trường; xuất ONNX/OpenVINO để tăng tốc
   trên CPU Intel.
