# PneumoAI - Multi-Modal CXR Diagnostic PACS Web Service

Hệ thống Web Demo chẩn đoán hình ảnh X-quang ngực tự động (CADe/CADx) kết hợp hai kiến trúc chuyên biệt:
1. **U-Net++ (SE-ResNeXt-50 + scSE Dual Attention)**: Phân vùng chính xác ranh giới lá tạng màng phổi trong Tràn khí màng phổi (Pneumothorax Segmentation).
2. **YOLOv11m @ 1280px**: Định vị đa tổn thương lồng ngực (Tim to, Quai ĐMC vồng, Bệnh lý màng phổi, Đám mờ nhu mô phổi).

---

## 1. Cấu trúc thư mục

```
pneumothorax-ai-service/
├── app/
│   ├── core/
│   │   ├── config.py              # Cấu hình ngưỡng, nhãn, màu sắc
│   │   ├── unet_segmenter.py      # Pipeline suy luận U-Net++ (CLAHE, TTA, Post-filtering)
│   │   ├── yolo_detector.py       # Pipeline suy luận YOLOv11m & Single Cardiac Envelope
│   │   ├── clinical_metrics.py    # Tính tỷ lệ CTR, % Xẹp phổi Phi, Phân tầng cấp cứu Triage
│   │   ├── visualizer.py          # Render lớp phủ (Overlay) y tế chuẩn xác
│   │   └── inference.py           # Điều phối toàn bộ quy trình suy luận
│   ├── static/                    # CSS giao diện Dark PACS và JavaScript xử lý tương tác
│   └── templates/
│       └── index.html             # Giao diện Web PACS Station
├── weights/
│   ├── best_unet_plus_plus.pth    # Trọng số U-Net++ (213MB)
│   └── best_B_m1280.pt            # Trọng số YOLOv11m (40.7MB)
├── test_images/
│   └── demo_cases_raw/            # 5 ca X-quang mẫu đại diện các bệnh cảnh
├── results/                       # Thư mục lưu kết quả phân tích & lớp phủ ảnh
├── main.py                        # FastAPI Server chính
├── run.sh                         # Script 1-click khởi chạy dịch vụ
└── requirements.txt
```

---

## 2. Cách khởi chạy

Mở Terminal và chạy lệnh:

```bash
cd Regional_diagnosis_based_on_X_ray/pneumothorax-ai-service
./run.sh
```

Sau khi server khởi động xong, mở trình duyệt truy cập:
👉 **`http://localhost:8000`**

---

## 3. Các tính năng nổi bật trên Giao diện Web

1. **Bộ sưu tập Demo 1-Click (Demo Cases Gallery)**:
   - Ca 01: *Severe Tension Pneumothorax* (Tràn khí màng phổi nặng, xẹp phổi $\approx 42\%$).
   - Ca 02: *Classic Apical Pneumothorax* (Tràn khí đỉnh phổi kinh điển).
   - Ca 03: *Lateral Pleural Separation* (Vệt tràn khí viền sườn).
   - Ca 04: *Subtle Small Pneumothorax* (Tràn khí kín đáo ranh giới mỏng).
   - Ca 05: *Normal Healthy Chest* (Phổi hoàn toàn sạch - Đối chứng âm).
2. **Kéo thả ảnh bất kỳ (Drag & Drop)**: Hỗ trợ nạp ảnh X-quang mới từ máy tính để phân tích ngay lập tức.
3. **Trình xem PACS 2 màn hình (Dual Viewport)**:
   - Xem song song Ảnh gốc và Ảnh lớp phủ AI.
   - Các nút chuyển lớp: **Composite (Cả hộp và màng phổi)**, **U-Net++ Pleura Only**, **YOLO11 Detection Only**.
4. **Báo cáo Chẩn đoán Định lượng Lâm sàng (RadLex Structured Report)**:
   - **Triage Level**: Tự động gắn cờ `EMERGENCY` (Đỏ), `PRIORITY` (Vàng) hoặc `ROUTINE` (Xanh).
   - **Chỉ số Tim - Lồng ngực (CTR)**: Định lượng chính xác tỷ lệ $CTR = W_{\text{heart}} / W_{\text{thorax}}$.
   - **Mức độ Xẹp phổi ($\Phi$)**: Định lượng % diện tích phổi bị xẹp và đưa ra khuyến nghị can thiệp (Dẫn lưu ngực cấp cứu / Theo dõi).
   - **Bản kết luận X-quang có cấu trúc**: Tự động định dạng văn bản theo tiêu chuẩn RadLex.
