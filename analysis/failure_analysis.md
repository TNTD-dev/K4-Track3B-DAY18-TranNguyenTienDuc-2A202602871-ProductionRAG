# Failure Analysis — Production RAG

**Học viên:** Trần Nguyễn Tiến Đức · 2A202602871 · K4 Track 3B

## Điều kiện thực nghiệm

Cùng corpus text, test set 20 câu, BGE-M3, gpt-4o-mini temperature=0 và evaluator RAGAS 0.1.22. Answer Relevancy dùng ví dụ chung tiếng Việt; công thức metric không đổi. Ground truth chỉ được dùng trong evaluation. Production thêm hierarchy, combined enrichment, hybrid/RRF, cross-encoder và parent expansion.

Backend production: `qdrant_server`. Báo cáo đo thật, `scores_valid=true`; timestamps, per-question, provenance và model được lưu trong JSON.

| Metric | Naive baseline | Production | Δ |
|---|---:|---:|---:|
| faithfulness | 0.7969 | 0.9100 | +0.1131 |
| answer_relevancy | 0.7452 | 0.7384 | -0.0068 |
| context_precision | 0.9250 | 0.9500 | +0.0250 |
| context_recall | 0.9083 | 0.9333 | +0.0250 |

Production đạt 4/4 metric ≥ 0,70 và Faithfulness ≥ 0,85. Answer Relevancy giảm 0,0068; chưa đạt bonus toàn bộ metric ≥ 0,75. Một run 20 câu không đủ để khẳng định chênh lệch nhỏ có ý nghĩa thống kê.

Baseline được chấm lại trên cùng đáp án/context đã lưu sau lượt evaluator không hợp lệ; production dùng lượt chấm thành công. Đây là hai lượt judge riêng, không phải paired deterministic evaluation. Dữ liệu thô baseline lưu ở `reports/naive_baseline_report_raw.json`; intermediate invalid run chưa lưu được raw values nên chưa kết luận nguyên nhân. Validator chỉ dung sai 1e-9 tại biên [0,1] cho roundoff; NaN hoặc sai lệch lớn vẫn bị từ chối.

## Bottom-5 — câu có điểm trung bình thấp nhất

Bottom-5 là ứng viên cần triage, không có nghĩa cả năm đáp án đều sai. Diagnosis từ metric chỉ là giả thuyết. Cần đối chiếu câu trả lời, điều kiện và nguồn trước khi chốt root cause.

### #1. Mentor và buddy của nhân viên mới có thể là cùng một người không? Quản lý trực tiếp có thể làm mentor không?

- **Expected:** KHÔNG cho cả hai. Mentor và buddy phải là hai người khác nhau. Quản lý trực tiếp không được làm mentor hoặc buddy.
- **Got:** Mentor và Buddy của nhân viên mới **không thể là cùng một người** để đảm bảo nhân viên mới có nhiều nguồn hỗ trợ đa dạng. Quản lý trực tiếp **không được làm Mentor hoặc Buddy**. (Nguồn: mentor_buddy.md)
- **Worst metric:** `answer_relevancy` = 0.0000; trung bình = 0.6875.
- **Diagnosis (giả thuyết):** Đáp án trả lời đúng cả hai vế và trích đúng mentor_buddy.md. AR=0 không đủ để kết luận lỗi retrieval/generation; đây là bất đồng evaluator cần kiểm tra câu hỏi reverse-generation và nhãn noncommittal. Faithfulness 0,75 cũng cần kiểm tra từng claim thay vì đổi nguồn đúng.
- **Suggested fix:** Giữ kết quả gốc, bổ sung human adjudication và log intermediate evaluator. Dùng holdout cho câu phủ định và câu hai vế; không chỉnh few-shot theo đáp án test.
- **Error Tree:** Output đúng hai vế → context có quy tắc hai người khác nhau và cấm quản lý trực tiếp → nguồn đúng → nghi vấn scoring, chưa có bằng chứng mất retrieval.

**Bằng chứng được truy hồi:**

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

```text
[Nguồn: thu_viec.md]
# Chính sách thử việc
> Phiên bản: 1.2 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự

## Thời gian thử việc
Thời gian thử việc tiêu chuẩn là **60 ngày** kể từ ngày bắt đầu làm việc. Đối với vị trí quản lý cấp cao (Manager trở lên), thời gian thử việc có thể kéo dài đến 90 ngày.

## Lương thử việc
Nhân viên thử việc được nhận **85% mức lương** của cấp bậc tương ứng theo bảng lương công ty. Phụ cấp ăn trưa được áp dụng đầy đủ từ ngày đầu tiên.

## Quyền lợi trong thử việc
Nhân viên thử việc **KHÔNG được nghỉ phép năm**. Trường hợp cần nghỉ việc riêng, nhân viên thử việc phải xin nghỉ không lương và được trưởng phòng phê duyệt. Nhân viên thử việc được tham gia bảo hiểm xã hội bắt buộc nhưng chưa được hưởng gói bảo hiểm sức khỏe PVI.

## Đánh giá thử việc
Quản lý trực tiếp thực hiện đánh giá vào ngày thứ 30 (giữa kỳ) và ngày thứ 55 (cuối kỳ). Kết quả đánh giá quyết định việc ký hợp đồng chính thức.
```

```text
[Nguồn: lam_viec_tu_xa.md]
# Chính sách làm việc từ xa (WFH)
> Phiên bản: 2.0 | Ngày hiệu lực: 01/07/2024 | Phòng ban: Nhân sự

## Quy định chung
Nhân viên chính thức được phép làm việc từ xa tối đa **2 ngày/tuần**. Lịch WFH phải được đăng ký và phê duyệt bởi quản lý trực tiếp vào **thứ Hai** đầu tuần.

## Yêu cầu kỹ thuật
- Kết nối **VPN** bắt buộc khi làm việc từ xa
- Đảm bảo đường truyền internet ổn định (tối thiểu 20Mbps)
- Sử dụng thiết bị công ty hoặc thiết bị cá nhân đã cài MDM
- Tham gia đầy đủ các cuộc họp online theo lịch

## Hạn chế
Một số vị trí không được áp dụng WFH: lễ tân, nhân viên kho, bảo vệ, nhân viên vận hành hạ tầng IT. Trong các tuần có sự kiện toàn công ty (All-hands, Team building), WFH không được áp dụng.

## Đánh giá
Quyền WFH có thể bị thu hồi nếu nhân viên không đáp ứng KPI hoặc vi phạm quy định WFH quá 2 lần trong 1 quý.
```

### #2. Thông tin lương thuộc cấp độ phân loại dữ liệu nào?

- **Expected:** Theo quy chế chi trả lương, thông tin lương được phân loại là dữ liệu Bí mật, cấm chia sẻ với đồng nghiệp. Theo chính sách phân loại dữ liệu, dữ liệu Bí mật (cấp 3) phải mã hóa khi truyền và hạn chế truy cập theo need-to-know.
- **Got:** Thông tin lương thuộc cấp độ phân loại dữ liệu **Bí mật**. Điều này được nêu rõ trong nguồn [ky_luong.md]: "Thông tin lương là dữ liệu **Bí mật**, cấm chia sẻ với đồng nghiệp."
- **Worst metric:** `answer_relevancy` = 0.0000; trung bình = 0.7500.
- **Diagnosis (giả thuyết):** Context đã có ky_luong.md xác nhận Bí mật và bảng phan_loai_du_lieu.md ánh xạ Bí mật sang cấp 3. Đáp án đúng nhãn nhưng chưa nói cấp 3 hoặc quy tắc xử lý như reference. Đây là thiếu tổng hợp hai nguồn; AR=0 là mức phạt cần human review, không phản ánh việc câu trả lời hoàn toàn sai.
- **Suggested fix:** Query decomposition: xác định nhãn trong quy chế lương, rồi ánh xạ cấp và quy tắc xử lý trong bảng phân loại; trả lời rõ cấp 3 và dẫn cả hai nguồn.
- **Error Tree:** Output đúng nhãn nhưng thiếu cấp → hai nguồn đều đã có → retrieval đủ → generation chưa tổng hợp bảng và chính sách.

**Bằng chứng được truy hồi:**

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

```text
[Nguồn: phan_loai_du_lieu.md]
# Chính sách phân loại dữ liệu
> Phiên bản: 1.0 | Ngày hiệu lực: 01/06/2024 | Phòng ban: CNTT & An ninh thông tin

## Bốn cấp độ phân loại
| Cấp độ | Nhãn | Ví dụ |
|--------|------|-------|
| 1 | **Công khai** | Thông tin marketing, bài blog, tuyển dụng |
| 2 | **Nội bộ** | Quy trình nội bộ, biên bản họp, danh bạ nhân viên |
| 3 | **Bí mật** | Dữ liệu khách hàng, hợp đồng, chiến lược kinh doanh |
| 4 | **Tối mật** | Mã nguồn core, bí mật thương mại, kế hoạch M&A |

## Quy tắc xử lý
- **Công khai**: chia sẻ tự do, không hạn chế
- **Nội bộ**: chỉ chia sẻ nội bộ, không ra ngoài công ty
- **Bí mật**: mã hóa khi truyền, hạn chế quyền truy cập theo need-to-know
- **Tối mật**: mã hóa end-to-end, truy cập cần phê duyệt Giám đốc CNTT, log tất cả truy cập

## Vi phạm
Vi phạm chính sách phân loại dữ liệu sẽ bị xử lý kỷ luật theo Quy chế kỷ luật công ty, mức cao nhất là sa thải và chịu trách nhiệm pháp lý.
```

```text
[Nguồn: bang_luong_2024.md]
# Bảng lương năm 2024
> Phiên bản: 3.0 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự & Tài chính

## Khung lương theo cấp bậc
| Cấp bậc | Lương gross (VNĐ/tháng) |
|----------|--------------------------|
| Junior (P1-P2) | 12.000.000 - 20.000.000 |
| Senior (P3-P4) | 20.000.000 - 35.000.000 |
| Lead (P5) | 35.000.000 - 50.000.000 |
| Manager (M1-M2) | 45.000.000 - 70.000.000 |

## Lương thử việc
Nhân viên trong thời gian thử việc được nhận **85% lương** của cấp bậc tương ứng. Sau khi hoàn thành thử việc, lương được điều chỉnh lên 100% kể từ ngày ký hợp đồng chính thức.

## Điều chỉnh lương
Lương được xét tăng 2 lần/năm vào tháng 1 và tháng 7 dựa trên kết quả đánh giá hiệu suất. Mức tăng lương trung bình từ 8-15% tùy theo kết quả đánh giá và ngân sách công ty.
```

### #3. Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?

- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Theo chính sách tạm ứng (Nguồn: tam_ung.md), nhân viên tạm ứng tối đa 20.000.000 VNĐ cho mỗi lần đề xuất. Khoản tạm ứng phải được thanh toán trong vòng 15 ngày kể từ ngày nhận tiền. Nếu không thanh toán sau 15 ngày, sẽ bị tính phí 2%/tháng trên số tiền chưa hoàn ứng.  Trong trường hợp này, nhân viên tạm ứng 15.000.000 VNĐ và thanh toán sau 20 ngày. Thời gian quá hạn là 5 ngày (20 ngày - 15 ngày).  Phí phạt được tính như sau: - Số tiền tạm ứng: 15.000.000 VNĐ - Phí phạt: 2%/tháng = 0.02 * 15.000.000 VNĐ = 300.000 VNĐ/tháng  Vì nhân viên đã quá hạn 5 ngày, tính phí phạt cho 1 tháng (do phí phạt được tính theo tháng):  - Tổng phí phạt: 300.000 VNĐ  Như vậy, nhân viên sẽ bị phạt **300.000 VNĐ**.
- **Worst metric:** `faithfulness` = 0.5000; trung bình = 0.7545.
- **Diagnosis (giả thuyết):** Nguồn quy định 15 ngày và 2%/tháng nhưng không quy định làm tròn cả tháng hay tính theo ngày. Model suy diễn quá hạn 5 ngày thành một tháng và kết luận 300.000 VNĐ như phí phải trả, sai reference pro-rata. Đây là lỗi generation/giả định tính toán, không phải thiếu chunk tạm ứng.
- **Suggested fix:** Tách facts và giả định: quá hạn 5 ngày, phí tháng 300.000; nếu giả định tháng 30 ngày và pro-rata thì 15.000.000 × 0,02 × 5/30 = 50.000 VNĐ. Phải nêu quy tắc pro-rata chưa có trong nguồn và cần xác nhận Tài chính, không khẳng định 50.000 là chính sách đã ghi.
- **Error Tree:** Output sai so với reference → context có deadline và rate → thiếu convention theo ngày → generation tự áp dụng tròn tháng → bổ sung kiểm tra giả định, không tăng top-k vô ích.

**Bằng chứng được truy hồi:**

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

```text
[Nguồn: chi_phi_expense.md]
# Quy trình chi phí và hoàn ứng
> Phiên bản: 1.3 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Tài chính

## Thời hạn nộp hồ sơ
Nhân viên phải nộp hồ sơ hoàn ứng chi phí trong vòng **7 ngày** kể từ ngày phát sinh chi phí. Hồ sơ nộp trễ quá 30 ngày sẽ không được xử lý, trừ trường hợp đặc biệt có giải trình của trưởng phòng.

## Chứng từ yêu cầu
Mọi chi phí **trên 200.000 VNĐ** phải có **hóa đơn hoặc biên lai** hợp lệ. Chi phí dưới 200.000 VNĐ có thể khai báo không cần hóa đơn nhưng phải ghi rõ mục đích và ngày phát sinh.

## Phê duyệt
Tất cả chi phí cần được phê duyệt bởi **quản lý trực tiếp** trước khi nộp phòng Tài chính. Thời gian xử lý hoàn ứng: 5-10 ngày làm việc sau khi hồ sơ đầy đủ và được phê duyệt.

## Lưu ý
Chi phí công tác phải tuân theo Chính sách công tác phí riêng. Hai chính sách cùng áp dụng: chi phí công tác theo định mức công tác phí, chi phí khác theo quy trình hoàn ứng chung.
```

```text
[Nguồn: nghi_phep_nam_v2024.md]
# Chính sách nghỉ phép năm (Phiên bản 2024)
> Phiên bản: 2.0 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự

## Số ngày phép năm
Mỗi nhân viên chính thức được hưởng **15 ngày phép năm** có lương, tăng từ 12 ngày so với chính sách năm 2023. Chính sách này thay thế hoàn toàn phiên bản 1.0 ban hành ngày 01/01/2023.

## Thâm niên công tác
Nhân viên có thâm niên từ **3 năm trở lên** được cộng thêm **1 ngày phép** cho mỗi 3 năm làm việc liên tục. Ví dụ: nhân viên 9 năm thâm niên được 18 ngày phép (15 + 3).

## Quy định sử dụng
Phép năm phải được đăng ký trước ít nhất 2 ngày làm việc qua hệ thống HR Portal. Nghỉ phép từ 5 ngày trở lên cần được trưởng phòng phê duyệt. Phép năm không sử dụng hết có thể chuyển tối đa 5 ngày sang năm sau, hết hạn vào 31/03 năm kế tiếp. Nhân viên cũng có thể yêu cầu thanh toán bằng tiền cho số ngày phép còn lại.
```

### #4. Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?

- **Expected:** Nhân viên phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Got:** Nhân viên được tài trợ khóa học 25 triệu VNĐ và nghỉ việc sau 8 tháng hoàn thành khóa học. Theo chính sách hoàn chi đào tạo (Nguồn: hoan_chi_dao_tao.md), nhân viên phải cam kết làm việc tại công ty ít nhất 1 năm sau khi hoàn thành khóa học. Nếu nghỉ việc trước thời hạn cam kết, nhân viên phải hoàn trả 100% chi phí đào tạo đã được tài trợ.  Vì nhân viên nghỉ việc sau 8 tháng, tức là chưa đủ 1 năm, nên nhân viên phải hoàn trả 100% chi phí đào tạo.  **Số tiền phải hoàn trả: 25.000.000 VNĐ.**
- **Worst metric:** `faithfulness` = 0.3750; trung bình = 0.7792.
- **Diagnosis (giả thuyết):** Đáp án 25.000.000 VNĐ đúng reference: 8 tháng < 12 tháng, hoàn 100%. Context có quy tắc 1 năm/100%, còn 25 triệu và 8 tháng đến từ câu hỏi. Faithfulness 0,375 có thể do judge đánh dấu các facts trong tình huống là không xuất hiện trong context hoặc sai attribution; chưa thể khẳng định hallucination từ điểm này.
- **Suggested fix:** Viết ngắn, phân biệt số liệu do người hỏi cung cấp với quy tắc được trích dẫn. Human audit từng claim và evaluator trace; giữ metric chuẩn, không thêm ground truth hoặc dữ liệu câu hỏi vào context chỉ để tăng điểm.
- **Error Tree:** Output đúng phép tính → nguồn có điều kiện/100% → facts tình huống nằm trong question → nghi vấn đánh giá provenance và claim decomposition, cần kiểm chứng trace.

**Bằng chứng được truy hồi:**

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

```text
[Nguồn: dao_tao_noi_bo.md]
# Chính sách đào tạo nội bộ
> Phiên bản: 1.3 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự & Đào tạo

## Yêu cầu đào tạo
Mỗi phòng ban phải tổ chức tối thiểu **2 buổi đào tạo/quý** về chuyên môn hoặc kỹ năng mềm. Nhân viên mới (trong 6 tháng đầu) **bắt buộc** tham gia tất cả các buổi đào tạo của phòng ban.

## Ngân sách đào tạo
Công ty dành ngân sách đào tạo **5.000.000 VNĐ/người/năm** cho các khóa học bên ngoài. Nhân viên có thể đăng ký khóa học qua hệ thống Learning Portal và cần phê duyệt của trưởng phòng.

## Đánh giá hiệu quả
Sau mỗi khóa đào tạo, nhân viên cần hoàn thành bài đánh giá và khảo sát phản hồi. Phòng Đào tạo tổng hợp kết quả hàng quý để cải thiện chương trình đào tạo.
```

### #5. Có cần kích hoạt xác thực đa yếu tố (MFA) không?

- **Expected:** Có, theo chính sách mật khẩu v2.0 hiện hành, tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và hệ thống nội bộ. Chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** Có, tất cả nhân viên **bắt buộc** phải kích hoạt xác thực đa yếu tố (MFA) cho tài khoản email, VPN và các hệ thống nội bộ. (Nguồn: mat_khau_v2.md)
- **Worst metric:** `context_recall` = 0.5000; trung bình = 0.7824.
- **Diagnosis (giả thuyết):** Đáp án dùng v2 đúng và nêu bắt buộc MFA cho email/VPN/hệ thống nội bộ. Context chứa cả v1 đã thay thế. Recall=0,5 có thể do reference có thêm câu v1 không yêu cầu MFA: v1 chỉ không đề cập MFA, không chứng minh không yêu cầu. Không nên thêm claim thiếu bằng chứng để khớp reference.
- **Suggested fix:** Gắn metadata effective_date/status/superseded_by và lọc phiên bản cho truy vấn hiện hành; reference audit phân biệt không đề cập với không yêu cầu. Có thể giải thích v1 đã bị thay thế bằng v2 khi cần lịch sử chính sách.
- **Error Tree:** Output đúng chính sách hiện hành → nguồn v2 đầy đủ → v1 là obsolete context → generation đã chọn đúng → ưu tiên version filtering và kiểm tra reference.

**Bằng chứng được truy hồi:**

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

```text
[Nguồn: mua_sam.md]
# Quy trình mua sắm
> Phiên bản: 2.2 | Ngày hiệu lực: 01/04/2024 | Phòng ban: Hành chính & Tài chính

## Thẩm quyền phê duyệt
| Giá trị đơn hàng | Người phê duyệt |
|-------------------|-----------------|
| Dưới **5.000.000 VNĐ** | Trưởng phòng (Manager) |
| Từ **5.000.000 - 50.000.000 VNĐ** | Giám đốc phòng ban (Director) |
| Trên **50.000.000 VNĐ** | Tổng Giám đốc (CEO) |

## Quy trình đề xuất
1. Tạo phiếu đề xuất mua sắm trên hệ thống Procurement Portal
2. Đính kèm ít nhất 3 báo giá cho đơn hàng trên 10.000.000 VNĐ
3. Chờ phê duyệt theo thẩm quyền tương ứng
4. Phòng Mua sắm đặt hàng và theo dõi giao nhận

## Lưu ý đặc biệt
Mua sắm thiết bị CNTT (laptop, server, phần mềm) cần có xác nhận của phòng CNTT về cấu hình kỹ thuật trước khi đề xuất. Đơn hàng khẩn cấp có thể bỏ qua yêu cầu 3 báo giá nhưng phải có giải trình bằng văn bản.
```

```text
[Nguồn: mat_khau_v1.md]
# Chính sách mật khẩu (Phiên bản cũ)
> Phiên bản: 1.0 | Ngày hiệu lực: 01/01/2022 | Phòng ban: CNTT | Trạng thái: ĐÃ THAY THẾ bởi v2.0

## Yêu cầu mật khẩu
Mật khẩu phải có tối thiểu **8 ký tự**, bao gồm ít nhất 1 chữ hoa, 1 chữ thường và 1 số. Không được sử dụng tên đăng nhập hoặc các thông tin cá nhân dễ đoán làm mật khẩu.

## Chu kỳ thay đổi
Mật khẩu phải được thay đổi **mỗi 90 ngày**. Hệ thống sẽ tự động nhắc nhở trước 7 ngày. Mật khẩu mới không được trùng với 3 mật khẩu gần nhất.

## Lưu ý
Chính sách này đã được thay thế bởi Chính sách mật khẩu v2.0 từ ngày 01/07/2024. Vui lòng tham khảo phiên bản mới để biết quy định hiện hành.
```

## Latency breakdown

Thời gian đơn vị ms. Build một lần; các query dùng model đã load. Cache enrichment và tải model bên ngoài build ảnh hưởng cách tái lập. Query timing đo wall time, không chỉ compute GPU. Không dùng trung bình để khẳng định p95.

| Bước | Thời gian / trung bình (ms) |
|---|---:|
| Build: load_chunk_ms | 22.38 |
| Build: enrichment_ms | 15.39 |
| Build: index_ms | 47212.65 |
| Build: reranker_load_ms | 13184.23 |
| Query: hybrid_search_ms | 124.52 |
| Query: rerank_ms | 477.85 |
| Query: generation_ms | 4718.29 |
| Evaluation: cả 20 câu | 260958.37 |

## Giới hạn và ưu tiên tiếp theo

- Hai PDF scan cần OCR; hiện không có bằng chứng từ hai file này.
- Semantic comparison nối corpus khác cách pipeline chunk từng nguồn; không dùng số chunk của phép compare thay cho số chunk index.
- Tập 20 câu cố định nhỏ, judge có tính ngẫu nhiên. Cần holdout, ablation từng module và lặp nhiều run để xác nhận mức cải thiện.
- Ưu tiên audit version/negation, query decomposition cho multi-hop và kiểm chứng giả định trong phép tính tài chính.
