# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Lê Thanh Điệp
**Ngày:** 30/6/2026
**Judge model:** gpt-4o-mini

---

## 1. Pairwise Judge Results

*(Chạy pairwise_judge() trên ít nhất 5 cặp answers)*

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | Số ngày nghỉ phép khi kết hôn | B | Câu B cung cấp đầy đủ thông tin về việc nghỉ phép có lương và không trừ vào phép năm, trong khi A chỉ đưa ra con số ngày nghỉ đơn thuần. |
| 2 | Hạn mức bảo hiểm PVI cho nhân viên | tie | Cả hai câu trả lời đều đưa ra hạn mức chính xác, nhưng hoán đổi vị trí làm thay đổi quyết định do thiên vị vị trí nên hệ thống ghi nhận hòa (tie). |
| 3 | Mức phụ cấp ăn trưa hàng tháng | B | Câu B bổ sung thêm thông tin hữu ích về thời điểm chi trả (cùng kỳ lương) giúp thông tin đầy đủ hơn. |
| 4 | Mentor và buddy có trùng nhau không | B | Câu B trả lời chính xác và đầy đủ các câu hỏi phụ, trong khi A không tìm thấy thông tin hữu ích. |
| 5 | Phê duyệt mua thiết bị 55 triệu | B | Câu B cung cấp thêm ngữ cảnh về ngưỡng giá trị phê duyệt (trên 50 triệu) giúp câu trả lời rõ ràng hơn. |

---

## 2. Swap-and-Average Results

*(Chạy swap_and_average() trên cùng các cặp)*

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | B | B | B | True |
| 2 | B | A | tie | False |
| 3 | B | B | B | True |
| 4 | B | B | B | True |
| 5 | B | B | B | True |

**Position bias rate:** 32.0% (= số case NOT consistent / tổng)

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 5 label=1, 5 label=0)  
**Judge labels:** Nhãn thu được từ việc chạy LLM Judge trên 10 câu tương ứng (1 nếu model_answer thắng hoặc hòa ground_truth, 0 nếu thua).

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | 0 | Disagree |
| 5 | 0 | 0 | Agree |
| 12 | 1 | 1 | Agree |
| 21 | 1 | 0 | Disagree |
| 23 | 1 | 0 | Disagree |
| 29 | 0 | 0 | Agree |
| 33 | 1 | 0 | Disagree |
| 41 | 0 | 1 | Disagree |
| 46 | 1 | 1 | Agree |
| 50 | 0 | 0 | Agree |

**Cohen's κ:** 0.0741  
**Interpretation:** poor (thoả hiệp kém)

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: 1 / 32 cases
- B thắng + B dài hơn A: 24 / 32 cases  
- **Verbosity bias rate:** 78.1%

**Kết luận:** LLM có xu hướng rất mạnh mẽ trong việc lựa chọn câu trả lời dài hơn (chiếm tới 78.1% số case phân định thắng thua). Điều này là do LLM thường đánh đồng độ dài và sự chi tiết của câu chữ với chất lượng câu trả lời, ngay cả khi thông tin dài hơn đó chứa các nội dung dư thừa hoặc không được yêu cầu trong câu hỏi. Điều này tạo ra rủi ro phạt oan các câu trả lời ngắn gọn, trực diện và chính xác.

---

## 5. Nhận xét chung

> - Chỉ số đồng thuận κ = 0.0741 ở mức rất thấp (poor), chứng tỏ LLM judge chưa thực sự hiểu sâu sắc các quy tắc nghiệp vụ nội bộ tinh tế như con người và dễ bị đánh lừa bởi hình thức câu trả lời.
> - Tỷ lệ position bias khá cao (32.0%) cho thấy thứ tự trình bày câu trả lời ảnh hưởng lớn tới phán quyết của LLM.
> - Cơ chế Swap-and-average tỏ ra cực kỳ hiệu quả khi giúp lọc bỏ các phán quyết thiếu nhất quán do position bias bằng cách đưa chúng về kết quả hòa (tie), tránh việc chọn bừa bãi.
> - Trên môi trường production, chúng ta không nên sử dụng LLM Judge thô mà cần thiết lập một bộ tiêu chí đánh giá (rubrics) cực kỳ chi tiết, áp dụng kỹ thuật Few-shot Prompting với các ví dụ chuẩn mực, và bắt buộc chạy Swap-and-average hoặc kết hợp biểu quyết đồng thuận từ nhiều Judge model khác nhau để triệt tiêu bias.
