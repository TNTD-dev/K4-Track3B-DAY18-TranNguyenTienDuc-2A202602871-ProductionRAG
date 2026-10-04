# Failure Analysis — Production RAG

**Học viên:** Trần Nguyễn Tiến Đức · 2A202602871 · K4 Track 3B

## Điều kiện và kết quả thực nghiệm

20 câu cố định, BGE-M3, BGE reranker, Qdrant server, gpt-4o-mini và evaluator Việt giữ nguyên. Ground truth chỉ dùng khi chấm. Baseline/production dùng cùng prompt và công cụ tính. Báo cáo cuối là **một lượt đo đầy đủ sau khi sửa coverage nhiều ý và filter phiên bản**. Các bản trước (gồm lượt chấm 3 lần) lưu riêng ở reports/optimization_v*. Không trộn điểm từ các bản khác nhau. Bộ lab đã được dùng để debug, nên cần holdout độc lập.

| Metric | Baseline | Production | Δ |
|---|---:|---:|---:|
| faithfulness | 0.8108 | 0.8924 | +0.0817 |
| answer_relevancy | 0.7644 | 0.7873 | +0.0228 |
| context_precision | 0.9375 | 0.9917 | +0.0542 |
| context_recall | 0.9250 | 0.8833 | -0.0417 |

Production đạt 4/4 metric ≥0,75 và Faithfulness ≥0,85: đủ ngưỡng RAGAS chính cùng cả hai bonus về score. Điểm toàn bài do giảng viên chấm code/phân tích/reflection.

## Bottom-5 — diagnosis, fix, Error Tree

### #1. Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?

- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Ước tính phí quá hạn là **50000 VNĐ** nếu áp dụng pro-rata và giả định tháng 30 ngày. Thanh toán sau 20 ngày, thời hạn 15 ngày nên quá hạn 5 ngày. Công thức: 15000000 × 0.02 × 5/30. Nguồn quy định 2%/tháng (300000 VNĐ/tháng); quy ước pro-rata/30 ngày là giả định tính toán, chưa được quy định trong nguồn. (Nguồn: tam_ung.md)
- **Worst metric:** faithfulness = 0.1818; trung bình 0.6362.
- **Diagnosis:** Kết quả số đúng: quá hạn 5 ngày, ước tính pro-rata 50.000 theo tháng 30 ngày. Nguồn chỉ ghi 2%/tháng; convention và dữ kiện tình huống không nằm trong nguồn, nên faithfulness 0,1818 cần audit từng claim. Không được coi 50.000 là quy định chắc chắn.
- **Suggested fix:** Human audit nguồn/assumption; xác nhận convention với Tài chính. Nhánh grounded_fee_answer tính từ quy định duy nhất và dữ kiện người hỏi, không đọc ground truth.
- **Error Tree:** Output đúng có điều kiện → nguồn đủ hạn/rate nhưng thiếu convention → phân biệt source/user/assumption, audit scoring.

**Nguồn truy hồi:** [Nguồn: tam_ung.md], [Nguồn: chi_phi_expense.md], [Nguồn: nghi_phep_nam_v2024.md]

**Trích bằng chứng đầu tiên** (toàn bộ context ở JSON):
```text
[Nguồn: tam_ung.md]
# Chính sách tạm ứng
> Phiên bản: 1.0 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Tài chính

## Hạn mức tạm ứng
Nhân viên có thể tạm ứng tối đa **20.000.000 VNĐ** cho mỗi lần đề xuất. Tạm ứng phải ghi rõ mục đích: công tác, mua sắm, tổ chức sự kiện hoặc chi phí dự án.

## Thời hạn thanh toán
Khoản tạm ứng phải được thanh toán (nộp chứng từ hoàn ứng) trong vòng **15 ngày** kể từ ngày nhận tiền. Trường hợp công tác dài ngày, thời hạn tính từ ngày kết thúc chuyến công tác.

## Phạt quá hạn
Khoản tạm ứng chưa thanh toán sau 15 ngày sẽ bị tính phí **2%/tháng** trên số tiền chưa hoàn ứng. Phí này được khấu trừ vào lương tháng kế tiếp. Nhân viên có khoản tạm ứng chưa thanh toán sẽ không được phê duyệt tạm ứng mới.

## Phê duyệt
Tạm ứng dưới 5.000.000 VNĐ: trưởng phòng phê duyệt. Từ 5.000.000 VNĐ trở lên: cần thêm phê duyệt Kế toán trưởng.
```

### #2. Mentor và buddy của nhân viên mới có thể là cùng một người không? Quản lý trực tiếp có thể làm mentor không?

- **Expected:** KHÔNG cho cả hai. Mentor và buddy phải là hai người khác nhau. Quản lý trực tiếp không được làm mentor hoặc buddy.
- **Got:** Mentor và buddy của nhân viên mới **không thể là cùng một người**. Điều này nhằm đảm bảo nhân viên mới có nhiều nguồn hỗ trợ đa dạng.   Quản lý trực tiếp **không được làm mentor hoặc buddy**.   (Nguồn: mentor_buddy.md)
- **Worst metric:** answer_relevancy = 0.0000; trung bình 0.7500.
- **Diagnosis:** Đáp án đúng cả hai câu phủ định, nguồn mentor_buddy.md đầy đủ. AR=0 là bất đồng judge cần kiểm tra, không đủ bằng chứng kết luận mất retrieval.
- **Suggested fix:** Human adjudication, log intermediate noncommittal trên holdout; giữ nguyên score và không thêm đáp án test vào evaluator.
- **Error Tree:** Output đúng hai vế → context đầy đủ → nghi vấn evaluator, không tăng top-k vô ích.

**Nguồn truy hồi:** [Nguồn: mentor_buddy.md], [Nguồn: thu_viec.md], [Nguồn: lam_viec_tu_xa.md]

**Trích bằng chứng đầu tiên** (toàn bộ context ở JSON):
```text
[Nguồn: mentor_buddy.md]
# Chương trình Mentor & Buddy
> Phiên bản: 1.0 | Ngày hiệu lực: 01/06/2024 | Phòng ban: Nhân sự

## Phân biệt Mentor và Buddy
- **Mentor**: hỗ trợ **chuyên môn kỹ thuật**, hướng dẫn phát triển nghề nghiệp, chia sẻ kiến thức chuyên sâu. Mentor phải là nhân viên cấp Senior trở lên, khác phòng ban hoặc cùng phòng ban nhưng không phải quản lý trực tiếp.
- **Buddy**: hỗ trợ **văn hóa và hòa nhập**, giúp nhân viên mới làm quen môi trường, giới thiệu đồng nghiệp, hướng dẫn quy trình hàng ngày. Buddy là nhân viên cùng phòng ban, thâm niên từ 6 tháng.

## Quy tắc quan trọng
Mentor và Buddy phải là **hai người khác nhau** để đảm bảo nhân viên mới có nhiều nguồn hỗ trợ đa dạng. Quản lý trực tiếp không được làm Mentor hoặc Buddy.

## Thời gian chương trình
Buddy hỗ trợ trong 3 tháng đầu. Mentor đồng hành trong 6 tháng đầu. Sau đó, nhân viên có thể tự chọn mentor dài hạn thông qua chương trình Mentor tự nguyện.
```

### #3. Thông tin lương thuộc cấp độ phân loại dữ liệu nào?

- **Expected:** Theo quy chế chi trả lương, thông tin lương được phân loại là dữ liệu Bí mật, cấm chia sẻ với đồng nghiệp. Theo chính sách phân loại dữ liệu, dữ liệu Bí mật (cấp 3) phải mã hóa khi truyền và hạn chế truy cập theo need-to-know.
- **Got:** Thông tin lương thuộc cấp độ phân loại dữ liệu **Bí mật**. Theo quy định, dữ liệu lương là thông tin bí mật và cấm chia sẻ với đồng nghiệp (Nguồn: ky_luong.md).
- **Worst metric:** answer_relevancy = 0.0000; trung bình 0.7500.
- **Diagnosis:** Nhãn Bí mật và citation ky_luong.md đúng. Context có bảng cấp 3 nhưng answer chưa tổng hợp số cấp và quy tắc xử lý. AR=0 cần human review; thiếu tổng hợp hai nguồn là điểm cải tiến thật.
- **Suggested fix:** Dẫn ky_luong.md cho nhãn, phan_loai_du_lieu.md cho cấp 3 và handling; audit coverage của multi-source synthesis.
- **Error Tree:** Retrieval đã có cả hai nguồn → generation chỉ nêu nhãn → bổ sung tổng hợp bảng/citation.

**Nguồn truy hồi:** [Nguồn: ky_luong.md], [Nguồn: phan_loai_du_lieu.md], [Nguồn: bang_luong_2024.md]

**Trích bằng chứng đầu tiên** (toàn bộ context ở JSON):
```text
[Nguồn: ky_luong.md]
# Quy chế chi trả lương
> Phiên bản: 1.1 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Tài chính & Nhân sự

## Ngày chi trả
Lương tháng được chi trả vào **ngày mùng 5** hàng tháng. Nếu ngày 5 rơi vào cuối tuần hoặc ngày lễ, lương được chi trả vào **ngày làm việc liền trước đó**. Lương được chuyển khoản vào tài khoản ngân hàng đã đăng ký.

## Lương tháng 13
Nhân viên chính thức được hưởng **lương tháng 13**, chi trả vào tháng 1 hàng năm (trước Tết Nguyên đán). Mức lương tháng 13 bằng 1 tháng lương cơ bản, tính pro-rata cho nhân viên chưa đủ 12 tháng.

## Phiếu lương
Phiếu lương điện tử được gửi qua email công ty vào ngày chi trả. Thắc mắc về lương liên hệ phòng Tài chính trong vòng 5 ngày làm việc. Thông tin lương là dữ liệu **Bí mật**, cấm chia sẻ với đồng nghiệp.
```

### #4. Có cần kích hoạt xác thực đa yếu tố (MFA) không?

- **Expected:** Có, theo chính sách mật khẩu v2.0 hiện hành, tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và hệ thống nội bộ. Chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** Có, tất cả nhân viên **bắt buộc** phải kích hoạt xác thực đa yếu tố (MFA) cho tài khoản email, VPN và các hệ thống nội bộ. (Nguồn: mat_khau_v2.md)
- **Worst metric:** context_recall = 0.5000; trung bình 0.7767.
- **Diagnosis:** Đáp án dùng v2 đúng, context đã lọc v1 cũ. Recall 0,5 có thể do reference còn câu v1 không yêu cầu MFA; v1 chỉ không đề cập, không chứng minh không yêu cầu. Không thêm claim thiếu nguồn để khớp reference.
- **Suggested fix:** Reference audit phân biệt không đề cập với không yêu cầu; giữ filter theo ngày khi hỏi hiện hành và lưu lịch sử cho truy vấn năm cũ.
- **Error Tree:** Output đúng hiện hành → source v2 đầy đủ → reference có claim lịch sử thiếu bằng chứng → audit reference.

**Nguồn truy hồi:** [Nguồn: mat_khau_v2.md], [Nguồn: mua_sam.md], [Nguồn: vpn_truy_cap.md]

**Trích bằng chứng đầu tiên** (toàn bộ context ở JSON):
```text
[Nguồn: mat_khau_v2.md]
# Chính sách mật khẩu (Phiên bản hiện hành)
> Phiên bản: 2.0 | Ngày hiệu lực: 01/07/2024 | Phòng ban: CNTT

## Yêu cầu mật khẩu
Mật khẩu phải có tối thiểu **12 ký tự**, bao gồm ít nhất 1 chữ hoa, 1 chữ thường, 1 số và 1 ký tự đặc biệt (!@#$%^&*). Khuyến khích sử dụng passphrase dài hơn 16 ký tự.

## Xác thực đa yếu tố (MFA)
Tất cả nhân viên **bắt buộc** kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ. Phương thức MFA được chấp nhận: ứng dụng Authenticator (ưu tiên), SMS OTP, hoặc YubiKey.

## Chu kỳ thay đổi
Mật khẩu phải được thay đổi **mỗi 120 ngày**. Hệ thống tự động nhắc nhở trước 14 ngày. Mật khẩu mới không được trùng với 5 mật khẩu gần nhất. Tài khoản bị khóa sau 5 lần nhập sai liên tiếp.

## Chính sách thay thế
Văn bản này thay thế Chính sách mật khẩu v1.0 ban hành ngày 01/01/2022.
```

### #5. Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?

- **Expected:** Nhân viên phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Got:** Nhân viên phải hoàn trả **100% chi phí** đào tạo đã được tài trợ, tức là **25.000.000 VNĐ**. Điều này được quy định trong chính sách hoàn chi đào tạo, yêu cầu nhân viên cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. (Nguồn: hoan_chi_dao_tao.md)
- **Worst metric:** faithfulness = 0.5000; trung bình 0.7883.
- **Diagnosis:** Đáp án 25.000.000 đúng reference với hoàn 100%. Số tiền và 8 tháng đến từ người hỏi; nguồn nêu cam kết 1 năm/100%. Faithfulness 0,5 cần kiểm tra attribution của dữ kiện tình huống và phép suy luận, chưa đủ chứng cứ hallucination.
- **Suggested fix:** Tách rõ số liệu user với quy tắc trích nguồn; kiểm chứng từng claim trên holdout. Không chèn question/ground truth vào source contexts để nâng score.
- **Error Tree:** Output đúng phép tính → nguồn có rule → inputs thuộc question → audit attribution/judge claims.

**Nguồn truy hồi:** [Nguồn: hoan_chi_dao_tao.md], [Nguồn: tam_ung.md], [Nguồn: dao_tao_noi_bo.md]

**Trích bằng chứng đầu tiên** (toàn bộ context ở JSON):
```text
[Nguồn: hoan_chi_dao_tao.md]
# Chính sách hoàn chi đào tạo
> Phiên bản: 1.1 | Ngày hiệu lực: 01/07/2023 | Phòng ban: Nhân sự & Tài chính

## Điều kiện được tài trợ
Công ty chi trả chi phí đào tạo bên ngoài (chứng chỉ, khóa học dài hạn, hội thảo) cho nhân viên có **thâm niên từ 1 năm trở lên**. Chi phí được tài trợ tối đa **30.000.000 VNĐ/khóa** và cần phê duyệt của Giám đốc phòng ban.

## Cam kết hoàn chi
Nhân viên được tài trợ đào tạo phải **cam kết làm việc tại công ty ít nhất 1 năm** sau khi hoàn thành khóa học. Nếu nghỉ việc trước thời hạn cam kết, nhân viên phải hoàn trả **100% chi phí** đào tạo đã được tài trợ.

## Quy trình
1. Nộp đơn đề xuất đào tạo kèm chương trình chi tiết
2. Trưởng phòng đánh giá mức độ liên quan đến công việc
3. Phòng Nhân sự xem xét ngân sách
4. Ký cam kết hoàn chi trước khi đăng ký khóa học
```

## Latency breakdown

Build/query đo wall time; enrichment dùng cache, không đại diện cold LLM cost. Calculator có thể thêm vòng gọi LLM, riêng nhánh phí rõ quy định tính trực tiếp. Evaluation là lượt judge production cuối.

| Bước | ms |
|---|---:|
| Build: load_chunk_ms | 28.87 |
| Build: enrichment_ms | 12.48 |
| Build: index_ms | 7210.25 |
| Build: reranker_load_ms | 2453.95 |
| Query trung bình: hybrid_search_ms | 164.55 |
| Query trung bình: rerank_ms | 620.79 |
| Query trung bình: generation_ms | 1481.77 |
| Evaluation production | 128706.39 |

## Giới hạn

- Hai PDF scan chưa OCR; không có bằng chứng từ chúng.
- Các lượt tối ưu trước lưu riêng, kể cả đáp án sai. Calculator bảo đảm số học; nhánh phí chỉ xử lý phrasing và quy định không mâu thuẫn, không phải bộ suy luận tài chính tổng quát.
- Query facets dạng liên từ và filter ngày hiệu lực đã được đo thực tế. Câu hỏi phức tạp hơn cần decomposition và audit date semantics trên holdout.
- Một lượt cuối và các thí nghiệm trên tập lab không chứng minh tổng quát hóa; Relevancy bị ảnh hưởng bởi abstention/caveat. Không sửa điểm hoặc đưa reference vào generation.
