"""Tạo notebook hướng dẫn tiếng Việt: python -m scripts.baseline.build_baseline_notebook --execute.

Notebook đọc model/validation đã chạy, không huấn luyện lại và không mở test.
"""

import argparse
from pathlib import Path

import nbformat

from src.paths import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Execute all cells and save actual outputs")
    args = parser.parse_args()
    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(text))

    def code(text):
        cells.append(nbformat.v4.new_code_cell(text))

    md("""# Học và kiểm chứng phần A: baseline GoEmotions

Đọc từ trên xuống. Mỗi cell nối một khái niệm NLP với dữ liệu hoặc kết quả thật.
Notebook chỉ mở train/validation và model đã fit; test dành cho protocol cuối.

Nếu chưa có model: chạy `python -m scripts.baseline.run_baseline`, tiếp đến `--variant balanced`,
rồi `python -m scripts.baseline.analyze_baseline` từ gốc repo. Cài `requirements-notebook.txt`
để chạy notebook bằng Jupyter hoặc VS Code. Phần triển khai chính nằm trong script,
nên nhóm có thể dùng cùng mã và metric khi tổng hợp A/B/C.""")
    code("""from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / 'src').is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from src.datasets.goemotions import load_goemotions, multi_hot
from src.models.baseline import load_run_metadata, load_aligned_scores, load_thresholds
from src.evaluation.metrics import evaluate_multilabel
print('Python:', sys.version.split()[0])""")
    md("""## 1. Dữ liệu và 28 nhãn

Một câu có thể có nhiều cảm xúc. Không dùng argmax để ép chọn duy nhất một nhãn.
Dữ liệu simplified có 43.410 train và 5.426 validation. Revision/SHA-256 được kiểm
trong `src/datasets/goemotions.py`; `id` dùng để ghép kết quả các mô hình.""")
    code("""frames, labels, manifest = load_goemotions(ROOT, write_metadata=False, splits=('train', 'validation'))
train, val = frames['train'], frames['validation']
y_train = multi_hot(train['labels'].tolist(), len(labels))
y_val = multi_hot(val['labels'].tolist(), len(labels))
print('Y train:', y_train.shape, '| Y validation:', y_val.shape)
pd.DataFrame({'label_id': range(len(labels)), 'label': labels,
              'train_positive': y_train.sum(axis=0), 'val_positive': y_val.sum(axis=0)}).head(8)""")
    md("""## 2. Multi-hot có nghĩa gì?

Hai nhãn admiration và gratitude được giữ đồng thời. Neutral là một cột riêng và
được giữ nguyên như dữ liệu nguồn. Ma trận sau chỉ là ví dụ minh họa.""")
    code("""example = multi_hot([[labels.index('admiration'), labels.index('gratitude')],
                     [labels.index('neutral')]], len(labels))
pd.DataFrame(example, columns=labels)[['admiration', 'gratitude', 'neutral']]""")
    md("""## 3. Nạp pipeline đã fit trên train

Pipeline gồm TF-IDF và 28 Logistic Regression nhị phân. Model và scores được
kiểm hash để tránh trộn kết quả từ các lần chạy. TF-IDF chỉ fit trên train.
Lệnh `transform`/`predict_proba` trên validation không thêm từ vào từ vựng.""")
    code("""folder = ROOT / 'data/processed/baseline/full'
metadata = load_run_metadata(folder, 'standard', labels)
model = joblib.load(folder / 'model.joblib')
scores = load_aligned_scores(folder / 'validation_scores.npz', val['id'].tolist(), labels)
print('So dac trung TF-IDF:', len(model.named_steps['tfidf'].vocabulary_))
print('So Logistic Regression:', len(model.named_steps['classifier'].estimators_))
print('Scores validation:', scores.shape)
print('So vong lap toi uu lon nhat:', max(metadata['optimizer_iterations']))""")
    md("""## 4. Tokenization mặc định và hạn chế

Baseline này dùng từ/cặp từ, lowercase và token pattern mặc định của scikit-learn.
Nó bỏ dấu câu, emoji và token một ký tự. `don't` có thể thành `don`; vì vậy chưa
thể nói baseline hiểu tốt mọi phủ định hoặc mỉa mai. Dùng cell sau để quan sát.""")
    code("""analyzer = model.named_steps['tfidf'].build_analyzer()
analyzer("I don't like this! 😢")""")
    md("""## 5. Điểm khác nhãn cuối như thế nào?

Chọn một ví dụ pride bị bỏ sót ở ngưỡng 0,5. Điểm các nhãn độc lập và không cần
cộng lại bằng 1. Bạn có thể đổi ID trong cell để xem một câu khác.""")
    code("""sample_id = 'eczwil0'
i = val.index[val['id'] == sample_id][0]
print(val.loc[i, 'text'])
print('Nhan that:', [labels[j] for j in val.loc[i, 'labels']])
view = pd.DataFrame({'label': labels, 'score': scores[i], 'predicted_at_0_5': scores[i] >= 0.5})
view.sort_values('score', ascending=False).head(6)""")
    md("""## 6. Micro/Macro-F1 và Hamming Loss

Macro-F1 lấy trung bình đều 28 F1, Micro-F1 cộng TP/FP/FN trước khi tính.
Hamming Loss là tỷ lệ các ô nhãn sai; cần đọc cùng Recall và F1 nhãn hiếm.
Hàm metric còn lưu TP/FP/FN/TN để tự kiểm công thức.""")
    code("""metric = evaluate_multilabel(y_val, scores, labels)
pd.Series({key: metric[key] for key in ('macro_f1', 'micro_f1', 'micro_precision',
                                     'micro_recall', 'hamming_loss', 'empty_prediction_count')})""")
    md("""## 7. So sánh ngưỡng 0,5, ngưỡng chung và ngưỡng riêng

Tất cả lựa chọn ngưỡng được thực hiện trên validation. Hàng chọn ngưỡng đo trên
chính tập dùng chọn ngưỡng, nên có thể lạc quan. Test sau freeze mới xác nhận.
Class weighting là biến thể model khác, phải có ngưỡng của chính model đó.""")
    code("""rows = []
for variant, location in [('standard', ROOT/'data/processed/baseline/full'),
                          ('balanced', ROOT/'data/processed/baseline/balanced/full')]:
    saved = json.loads((location/'analysis_validation.json').read_text(encoding='utf-8'))
    for mode, key in [('fixed', 'fixed_0_5'), ('global', 'global_on_validation'),
                      ('tuned', 'tuned_on_validation')]:
        item = saved[key]
        rows.append({'variant': variant, 'threshold_mode': mode,
                     **{key: item[key] for key in ('macro_f1', 'micro_f1', 'hamming_loss')}})
pd.DataFrame(rows)""")
    md("""## 8. Nhãn hiếm và cặp lỗi

Năm nhãn hiếm được xác định từ train. Bảng FN(A)+FP(B) chỉ đếm hai lỗi trong
cùng câu; khác với nhãn thật đồng xuất hiện. Một câu có thể góp nhiều cặp.""")
    code("""per_label = pd.read_csv(folder/'per_label_validation.csv')
per_label.nsmallest(5, 'train_support')[['label', 'train_support', 'validation_support',
                                      'f1_fixed', 'f1_global_val', 'f1_tuned_val']]""")
    code("""pairs = pd.read_csv(folder/'label_error_pairs_validation.csv')
pairs[(pairs['missed_label'] != 'neutral') & (pairs['extra_label'] != 'neutral')].head(6)""")
    md("""## 9. Tự nhập văn bản, xem score và ngưỡng

Đổi câu tiếng Anh bên dưới. Dùng model balanced và ngưỡng riêng đã chọn từ
validation. Demo cuối của nhóm vẫn phải dùng C tốt nhất theo yêu cầu cô.""")
    code("""text = 'I am so proud of you!'
balanced_folder = ROOT/'data/processed/baseline/balanced/full'
balanced_meta = load_run_metadata(balanced_folder, 'balanced', labels)
balanced_model = joblib.load(balanced_folder/'model.joblib')
thresholds = np.broadcast_to(load_thresholds(balanced_folder, balanced_meta, 'tuned'), (len(labels),))
new_scores = balanced_model.predict_proba([text])[0]
pd.DataFrame({'label': labels, 'score': new_scores, 'threshold': thresholds,
              'predicted': new_scores >= thresholds}).sort_values('score', ascending=False).head(6)""")
    md("""## 10. Việc bạn cần tự giải thích

1. Vì sao số feature là 58.338 còn số nhãn là 28?
2. Vì sao không fit TF-IDF trên validation/test?
3. Vì sao weighting có thể tăng Recall và cả FP?
4. Vì sao score pride 0,3776 không vượt ngưỡng 0,5?
5. Vì sao điểm tuned-validation chưa phải cải thiện trên test?
6. Vì sao bảng cặp lỗi không phải bảng nhãn thật đồng xuất hiện?

Xem `docs/BASELINE.md` để chạy lại và `reports/BASELINE_RESULTS.md` để dùng số thật.
Nguồn: [bài báo GoEmotions](https://aclanthology.org/2020.acl-main.372/),
[TF-IDF](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html),
[One-vs-Rest](https://scikit-learn.org/1.7/modules/generated/sklearn.multiclass.OneVsRestClassifier.html),
[chọn ngưỡng](https://scikit-learn.org/1.7/modules/classification_threshold.html).""")
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    if args.execute:
        from nbclient import NotebookClient
        NotebookClient(notebook, timeout=120, kernel_name='python3',
                       resources={'metadata': {'path': str(ROOT)}}).execute()
    output = ROOT / ('reports/notebooks/02_baseline.executed.ipynb' if args.execute
                     else 'notebooks/02_baseline.ipynb')
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f"Notebook saved with {len(cells)} cells: {output.relative_to(ROOT)}")


if __name__ == '__main__':
    main()
