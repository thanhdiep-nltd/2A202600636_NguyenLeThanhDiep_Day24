# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Lê Thanh Điệp
**Ngày:** 30/6/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~18.38ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~931.72ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Điền từ kết quả Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 10.80 | 18.38 | 18.38 | <10ms |
| NeMo Input Rail | 762.63 | 931.72 | 931.72 | <300ms |
| RAG Pipeline | ~1200.00 | ~1500.00 | ~1800.00 | <2000ms |
| NeMo Output Rail | ~600.00 | ~800.00 | ~900.00 | <300ms |
| **Total Guard** | 769.68 | **940.97** | 940.97 | **<500ms** |

**Budget OK?** [ ] Yes / [x] No  
**Comment:** NeMo Input Rail là bottleneck chính do thực hiện cuộc gọi API từ xa (OpenAI qua proxy) để sinh và so khớp vector embedding (text-embedding-3-small) cũng như sinh văn bản từ LLM (gpt-4o-mini) cho bộ chuyển đổi ý định. Để tối ưu hóa và đưa tổng thời gian Guard về dưới 500ms, ta có thể dùng mô hình nhúng cục bộ nhanh hơn (ví dụ: FastEmbed trên CPU) thay vì gọi API từ xa, tối ưu hóa bộ sinh suy diễn song song (speculative generation), hoặc sử dụng các mô hình phân loại cục bộ cực nhanh như SetFit thay cho LLM trong khâu phân loại ý định (intent classification).

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.5935 |
| Worst metric | answer_relevancy (0.0) |
| Dominant failure distribution | factual |
| Cohen's κ | 0.0741 |
| Adversarial pass rate | 17 / 20 |
| Guard P95 latency | 940.97 ms |

---

## Nhận xét & Cải tiến

> Hệ thống hoạt động tốt ở khâu lọc thông tin PII (Presidio kết hợp chính xác regex tiếng Việt cho CCCD/SĐT) giúp ngăn chặn rò rỉ dữ liệu cá nhân tức thì mà không có độ trễ đáng kể (<20ms). Cơ chế embeddings_only giúp NeMo khớp ý định nhanh hơn nhiều so với dùng hội thoại LLM truyền thống. 
> Tuy nhiên, do proxy mạng và cuộc gọi API ngoài, độ trễ P95 của NeMo vẫn vượt ngưỡng ngân sách (budget) 500ms của ứng dụng thực tế. Nếu đưa lên môi trường Production thực tế, tôi sẽ đề xuất chạy mô hình phân loại cục bộ gọn nhẹ (như HuggingFace Optimum/ONNX) thay thế hoàn toàn API gọi ngoài của NeMo, đồng thời tích hợp cache (như Redis) cho các câu hỏi phổ biến để bỏ qua bước Guardrail khi trùng lặp câu hỏi cũ.
