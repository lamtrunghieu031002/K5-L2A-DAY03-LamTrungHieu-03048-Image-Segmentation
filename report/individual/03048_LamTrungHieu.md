# Báo cáo thực hành Day 3

- Mã học viên theo lớp: 03048
- Ngày / CVAT local: 2026-11-11 (lab) – 2026-11-12 (QC, sửa) / không dùng CVAT; ZIP sinh bằng script theo đúng format CVAT
- Công cụ đã dùng (đúng như thực tế):
  - Gợi ý tự động bằng script Python ngoài CVAT ([generate_annotations_m2f.py](../../generate_annotations_m2f.py)), chạy CPU:
    - Semantic (easy, cp3, cp4, cp6): Mask2Former Swin-L Cityscapes (`facebook/mask2former-swin-large-cityscapes-semantic`), cạnh ngắn 1024 + lật ngang (TTA); chỉ giữ class của task, pixel khác để trống. Thêm bước dùng model COCO để bỏ nhãn trên mui xe của chính mình (cp4, cp6), trên xe buýt ở cp6 (task không có class `bus`), và bỏ `vegetation` ở vùng COCO nhận là núi/cỏ/đất (theo Cityscapes đó là `terrain`, không thuộc task) — ví dụ hai sườn đồi cỏ khô ở `easy_semantic/7ee6d192-89e2408b.jpg` trước bị gán `vegetation` 15% ảnh; nay sườn cỏ khô/đất để trống và chỉ các cụm cây/bụi sẫm trên đồi được giữ `vegetation` (3.8% ảnh), tách theo màu: tán cây ánh xanh lạnh (B − R ≥ 14), cỏ khô/đất ngả nâu.
    - Instance (medium, cp1, cp2, cp5) và Hard panoptic: Mask2Former Swin-L COCO-panoptic (`facebook/mask2former-swin-large-coco-panoptic`), ngưỡng 0.5; mỗi vật một mask không chồng lấn, lấp lỗ bên trong vật (kính/khe), stuff Hard gộp một mask/class. Mask lưu dạng RLE.
    - Bản thử trước (SegFormer-b2 + YOLO11x/SAM2.1) bị bỏ vì sai rõ: xe buýt gán `car`, cột nhà gán `vegetation`, vỉa hè gán `road`.
  - Kiểm trực quan bằng overlay ([visualize_results.py](../../visualize_results.py)) và `python3 scripts/inspect_submissions.py --dir submissions` (tất cả 9 ZIP: OK về cấu trúc).
  - Trợ lý AI (Claude Code) hỗ trợ viết script, kiểm overlay và soạn report này.
  - Trọng số model (`*.pt`) và ảnh overlay `viz/` không đưa lên fork.

## 1. Bài đã nộp

| Task | File ZIP đúng tên | Hoàn thành mấy ảnh | Điểm tối đa (coach chấm sau) |
| --- | --- | ---: | ---: |
| easy_semantic | easy_semantic.zip | 3 / 3 | 20 |
| medium_instance | medium_instance.zip | 3 / 3 | 32 |
| hard_panoptic | hard_panoptic.zip | 2 / 2 | 30 |
| cp1_holes | cp1_holes.zip | 1 / 1 | 3 |
| cp2_slice | cp2_slice.zip | 1 / 1 | 3 |
| cp5_occlusion | cp5_occlusion.zip | 1 / 1 | 3 |
| cp3_thin | cp3_thin.zip | 1 / 1 | 3 |
| cp4_curb | cp4_curb.zip | 1 / 1 | 3 |
| cp6_coverage | cp6_coverage.zip | 1 / 1 | 3 |
| **Tổng tối đa** | | | **100** |

Ghi chú trung thực: các ZIP được sinh bằng script (đúng format Segmentation mask 1.1 / COCO 1.0), không phải export từ CVAT. Số annotation đã nộp: medium 95, hard 65 (gồm 5 stuff/ảnh), cp1 7, cp2 16, cp5 33.

## 2. Một quyết định trước khi dùng gợi ý

- Ảnh, vị trí và object Medium đầu tiên tự vẽ: **không có** — tôi không tự vẽ object nào trước khi dùng gợi ý; toàn bộ mask Medium do Mask2Former COCO-panoptic sinh. Tôi ghi rõ điều này thay vì điền một object không có thật.
- Một quyết định gán nhãn đã kiểm: `medium_instance` / `000000181542.jpg` — hình người phụ nữ trên poster trong tủ kính cửa hàng CHANEL (phía trên, giữa ảnh). Bản gợi ý đầu (YOLO11x + SAM2.1) gán nó là `person`; bản hiện tại không gán.
- Quy tắc và lý do giữ: `person` là người thật đếm được trong cảnh; hình in trên poster là một phần của mặt tiền cửa hàng, không phải một instance người. Vì vậy giữ đề xuất hiện tại (không có mask `person` ở poster). Ngược lại, những người nhìn thấy qua kính xe buýt bên trái vẫn là người thật nên được giữ là `person`.

## 3. Một lỗi tôi tìm thấy và sửa

- Task/ảnh/vùng: `cp6_coverage` / `7daa6479-67988f3f.jpg` — xe buýt MTA ở giữa ảnh và mui xe của chính xe quay (đáy ảnh).
- Lỗi thuộc loại: sai lớp.
- Bằng chứng tôi nhìn thấy: trên overlay của bản đầu, phần lớn thân xe buýt được tô màu `car`, và mui xe đang lái ở đáy ảnh cũng bị tô `car`/`road`. Task cp6 không có class `bus`, nên xe buýt phải để trống chứ không gán `car`; mui xe của chính mình không thuộc class nào của task.
- Quy tắc và hành động sửa: chỉ gán class có trong `classes.json`; vùng không thuộc schema để trống thay vì ép vào class gần nhất (phiếu quy tắc: “không ép đoán cho đủ coverage”). Tôi đổi sang Mask2Former Swin-L Cityscapes, thêm bước dùng model COCO tìm vùng bus/truck (20.8% ảnh) và mui xe (5.2% ảnh) rồi bỏ nhãn `car`/`road` ở đó. Sau sửa, `car` chỉ còn ở các xe con thật (2.8% ảnh); vỉa hè bên phải cũng đổi từ `road` sang đúng `sidewalk`.
- Sau sửa đã Save và export lại chưa? Đã sinh lại `cp6_coverage.zip` và kiểm bằng `inspect_submissions.py` (OK) + xem lại overlay.

Kết quả tự đánh giá: chưa có điểm (chưa chạy scorer với reference). Scorecard ba tier tối đa 82, không phải điểm cuối.

## 4. Ba ca chưa chắc hoặc đã cân nhắc

| Ảnh/vị trí | Hai cách hiểu có thể | Quy tắc/chứng cứ | Quyết định hoặc câu hỏi cho coach |
| --- | --- | --- | --- |
| `cp4_curb` / `7d83710e-4697c3b2.jpg` — dải đất/cỏ giữa bó vỉa và vỉa hè bê tông (giữa–phải ảnh, quanh cây và túi rác) | `sidewalk` (nằm phía trên bó vỉa, thuộc phần dành cho người đi bộ) hay không thuộc class nào (là đất/cỏ, kiểu `terrain`) | Bó vỉa là ranh chức năng road–sidewalk; nhưng mặt đất/cỏ không phải bề mặt đi bộ. Task chỉ có `road`, `sidewalk` | Mask hiện tại gán phần đất trống là `sidewalk`, phần cỏ để trống. Hỏi coach: dải đất trồng cây sau bó vỉa có tính là `sidewalk` không? |
| `cp1_holes` / `000000144300.jpg` — kính chắn gió và khe giữa khung, động cơ của xe máy Honda | Khoét kính/khe khỏi mask hay giữ trong mask vật | Quy tắc cp1: kính/khe nằm trong vật, không khoét | Giữ trong mask: script lấp mọi lỗ kín bên trong mask `motorcycle`. Riêng chân chống màu đỏ phía sau không được tính vào xe máy vì là dụng cụ đứng riêng |
| `hard_panoptic` / `000000460147.jpg` — xe tải chở ô tô ở giữa ảnh | Một `truck` bao cả các xe đang chở, hay `truck` + từng `car` riêng | Panoptic: thing đếm từng vật; chỉ vẽ phần nhìn thấy | Mask hiện tại: một `truck` cho phần thân/sàn xe tải và các xe đang được chở là `car` riêng. Hỏi coach: ô tô đang được chở có tính là instance `car` không? |
