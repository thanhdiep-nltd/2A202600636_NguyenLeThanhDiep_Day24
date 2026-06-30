# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Lê Thanh Điệp  
**Ngày:** 30/6/2026

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 0.775 | 0.498 | 0.775 |
| answer_relevancy | 0.000 | 0.000 | 0.000 |
| context_precision | 0.958 | 0.896 | 0.892 |
| context_recall | 0.900 | 0.750 | 0.650 |
| **avg_score** | 0.658 | 0.536 | 0.579 |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | multi_hop | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 về độ dài tối thiểu, thời hạn đổi và MFA. | 0.0000 | faithfulness |
| 2 | multi_hop | So sánh quyền lợi bảo hiểm giữa nhân viên thử việc và nhân viên chính thức. | 0.3750 | faithfulness |
| 3 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hàng tháng và số ngày phép năm theo v2024 là bao nhiêu? | 0.3750 | faithfulness |
| 4 | multi_hop | Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu? | 0.3958 | faithfulness |
| 5 | adversarial | Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không? | 0.4167 | faithfulness |
| 6 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân (như NordVPN) khi WFH để tăng bảo mật thêm không? | 0.4167 | faithfulness |
| 7 | adversarial | Theo chính sách nghỉ phép cũ (v2023), nhân viên được nghỉ bao nhiêu ngày? Còn chính sách nào đang có hiệu lực hiện tại? | 0.4167 | answer_relevancy |
| 8 | factual | Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt? | 0.4583 | faithfulness |
| 9 | factual | Nhân viên được nghỉ bao nhiêu ngày khi kết hôn? | 0.5000 | faithfulness |
| 10 | factual | Nghỉ phép không lương 20 ngày cần ai phê duyệt? | 0.5000 | answer_relevancy |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 4 | 6 | 2 | 12 |
| answer_relevancy | 16 | 14 | 8 | 38 |
| context_precision | 0 | 0 | 0 | 0 |
| context_recall | 0 | 0 | 0 | 0 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** answer_relevancy  

**Lý do phân tích:**

> Chỉ số `answer_relevancy` bị chấm điểm thấp nhất (đều bằng 0.0) do sự không tương thích giữa tham số yêu cầu đánh giá `n>1` của thư viện Ragas cũ với proxy API ShopAIKey (trả về lỗi xác thực khi gọi song song nhiều phản hồi). Do đó, hàm làm sạch điểm số `clean_score` đã chuyển đổi các điểm lỗi này về 0.0 để tránh làm lỗi luồng chạy.
> Nếu xét trên các lỗi thực tế, `multi_hop` là distribution có điểm số thực tế thấp nhất (avg_score = 0.536), với `faithfulness` là điểm yếu chủ đạo (6 câu lỗi). Điều này xuất phát từ việc các câu hỏi multi-hop yêu cầu tổng hợp thông tin phức tạp từ nhiều văn bản khác nhau (như tính toán phép năm dựa trên cả cấp bậc và thâm niên nghỉ việc), khiến LLM dễ bị ảo giác (hallucination) hoặc tính toán sai lệch khi các đoạn văn bản được cắt nhỏ lẻ và thiếu ngữ cảnh liên kết.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating | Thêm kỹ thuật Self-Consistency (CoT), tinh chỉnh Prompt Template để ép mô hình suy luận từng bước trước khi đưa ra con số cuối cùng. |
| context_recall | Missing relevant chunks | Tối ưu hóa kích thước Chunk Size, sử dụng chiến lược Parent Document Retrieval để lưu giữ nhiều ngữ cảnh hơn khi tìm kiếm thông tin. |
| context_precision | Too many irrelevant chunks | Nâng cấp thuật toán Reranker (ví dụ sử dụng Cohere hoặc Cohere Rerank mạnh mẽ hơn) để loại bỏ các chunk rác nhiễu thông tin. |
| answer_relevancy | Answer doesn't match question | Căn chỉnh lại System Prompt để mô hình trả lời trực diện vào câu hỏi và loại bỏ các thông tin rườm rà không liên quan. |

---

## 6. Nhận xét về Adversarial Distribution

> Điểm số trung bình (`avg_score`) của `adversarial` (0.579) cao hơn `multi_hop` (0.536) nhưng thấp hơn `factual` (0.658). Pipeline RAG dễ bị nhầm lẫn khi gặp các câu hỏi chứa bẫy về phiên bản (như chính sách nghỉ phép v2023 cũ đã hết hiệu lực so với v2024 mới). 
> Trong Bottom 10, có 3 câu rơi vào adversarial (ID 48, 50, 49). Nguyên nhân là do các câu hỏi này cố tình đưa ra các tiền đề sai lệch hoặc hỏi về các chính sách cũ đã bị thay thế, khiến mô hình RAG nếu truy xuất nhầm tài liệu cũ sẽ đưa ra câu trả lời sai lệch hoàn toàn so với thực tế hiện hành.
