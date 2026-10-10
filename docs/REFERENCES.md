# Nguồn tham khảo

## NLP và các thành phần đang dùng

- [GoEmotions, ACL 2020](https://aclanthology.org/2020.acl-main.372/): bài nền tảng và taxonomy.
- [Dataset GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions): cấu hình simplified.
- [Google Research GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions): nguồn tác giả.
- [TF-IDF](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html),
  [Logistic Regression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html),
  [One-vs-Rest](https://scikit-learn.org/1.7/modules/generated/sklearn.multiclass.OneVsRestClassifier.html).
- [Chọn ngưỡng](https://scikit-learn.org/1.7/modules/classification_threshold.html): tách dữ liệu fit và chọn ngưỡng.
- [BART-large-MNLI](https://huggingface.co/facebook/bart-large-mnli): zero-shot NLI.
- [BERT](https://huggingface.co/google-bert/bert-base-cased),
  [RoBERTa](https://huggingface.co/FacebookAI/roberta-base),
  [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased).
- [PyTorch Get Started](https://pytorch.org/get-started/locally/): chọn wheel cho thiết bị.

## Repo tham khảo cách trình bày

Đã đọc trang GitHub chính thức khi chuẩn bị bản source.
Số sao dưới đây là số xấp xỉ trang hiển thị lúc kiểm tra ngày 10/10/2026; thay đổi theo thời gian.

| Repo | Sao hiển thị | Điều áp dụng vào repo này |
|---|---:|---|
| [huggingface/transformers](https://github.com/huggingface/transformers) | khoảng 167k | Installation/quick start, ví dụ ngắn, tách source/tests/docs |
| [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | khoảng 67,5k | Giới thiệu ngắn, hướng dẫn cài và đóng góp dễ tìm |
| [pytorch/examples](https://github.com/pytorch/examples) | khoảng 24,1k | Ví dụ và script chia theo nhiệm vụ |
| [huggingface/sentence-transformers](https://github.com/huggingface/sentence-transformers) | khoảng 19,2k | Tách cách dùng cơ bản khỏi hướng dẫn chuyên sâu |

Đây là nguồn tham khảo cách tổ chức/trình bày; số sao không chứng minh chất lượng
mô hình đồ án và không phải số sao repo của nhóm.
Các thuật toán, split và yêu cầu của GoEmotions được giữ theo thiết kế của đồ án.
