"""Tạo notebook EDA. Chạy từ gốc repo: python -m scripts.data.build_notebook."""
from pathlib import Path
import textwrap
import nbformat as nbf

from src.paths import ROOT
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))


md("""
# Khám phá dữ liệu GoEmotions cho phân loại cảm xúc đa nhãn

**Đầu ra:** số mẫu, phân bố đầy đủ 28 nhãn, 30 ví dụ có ID, kiểm tra chất lượng,
biểu đồ và báo cáo có thể chạy lại. Các kết quả được tính từ dữ liệu thật ở revision cố định.

**Cách học:** đọc phần giải thích → chạy ô code ngay bên dưới bằng `Shift+Enter` →
đối chiếu bảng/biểu đồ → đọc nhận xét. Có thể dùng **Restart Kernel and Run All Cells**.
Notebook trong Git chỉ giữ source; chạy các ô để tạo output trên máy mình.

**Phạm vi:** EDA cho GoEmotions: phân bố nhãn, chất lượng dữ liệu, multi-hot,
mất cân bằng, độ dài và đồng xuất hiện. Xem `docs/NOTEBOOKS.md` và bài GoEmotions
ACL 2020 để hiểu vị trí của bước này trong đồ án.
Không có kết quả huấn luyện hay F1 trong notebook này.

**Ngôn ngữ và công cụ:** Python 3.12/3.13; pandas/NumPy để đếm; PyArrow đọc Parquet;
Matplotlib/Seaborn vẽ hình; Hugging Face Tokenizers đo token; Jupyter để theo dõi từng bước.
Chạy CPU, không cần GPU, khóa API hay tải trọng số mô hình.

**Dữ liệu tiếng Anh:** bình luận Reddit, 27 cảm xúc + `neutral`. Diễn giải tiếng Việt
chỉ giúp học; không dịch hay sửa văn bản đầu vào, không thay tên nhãn chuẩn.

## Lộ trình

1. Môi trường và đường dẫn
2. Tải và kiểm chứng dữ liệu
3. Cấu trúc và mapping nhãn
4. Chất lượng và số mẫu
5. Mã hóa đa nhãn
6. Phân bố 28 nhãn
7. Mất cân bằng và nhãn hiếm
8. Số nhãn trên mỗi mẫu
9. Neutral và đồng xuất hiện
10. Trùng lặp trong/giữa các split
11. Độ dài văn bản
12. Độ dài theo ba tokenizer
13. Ba mươi ví dụ có thể truy vết
14. Đối chiếu snapshot
15. Xuất dữ liệu và báo cáo
16. Kết luận, giới hạn và câu hỏi tự kiểm tra

**Quy tắc sử dụng split:** thống kê test chỉ mô tả bộ dữ liệu, không dùng để chọn mô hình,
ngưỡng, nhãn hiếm hoặc độ dài cắt. Phân tích sâu, ví dụ minh họa và đồng xuất hiện dùng train.
Giữ nguyên split chính; không tự ý xóa bản ghi trùng hay ép mỗi mẫu về một nhãn.
""")
md("""
## Bước 1 — Chuẩn bị môi trường

Ở terminal tại thư mục dự án, chạy `python -m pip install -e ".[eda,notebooks]"`, rồi
`python -m jupyterlab`. Chọn kernel của đúng môi trường đã cài thư viện.
Nếu dùng Colab: tải cả repo lên `/content/goemotions-multilabel-classification`,
chuyển thư mục bằng `%cd /content/goemotions-multilabel-classification` và cài bằng
`%pip install -e ".[eda,notebooks]"`, sau đó khởi động lại runtime nếu được yêu cầu.
Chỉ tải riêng file ipynb sẽ thiếu `src/datasets/goemotions.py` và cấu hình tokenizer.

Ô dưới tìm thư mục gốc, tạo thư mục output và ghi lại phiên bản thật.
Không cần thay đường dẫn tuyệt đối của máy người viết.
""")
code("""
from pathlib import Path
import sys, json, platform, importlib.metadata, re
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'src/datasets/goemotions.py').exists()), None)
if ROOT is None:
    raise RuntimeError('Hãy mở notebook trong repo đầy đủ, hoặc chuyển cwd đến gốc repo.')
sys.path.insert(0, str(ROOT))
from src.datasets.goemotions import load_goemotions, multi_hot, download, sha256, SPLITS, REVISION
SEED = 42
TABLES, FIGURES = ROOT / 'reports/tables', ROOT / 'reports/figures'
for folder in [TABLES, FIGURES, ROOT / 'data/processed']:
    folder.mkdir(parents=True, exist_ok=True)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 130)
sns.set_theme(style='whitegrid', font='DejaVu Sans', palette='colorblind')
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 180})

def save_table(df, name):
    df.to_csv(TABLES / f'{name}.csv', index=False, encoding='utf-8-sig')

def finish_figure(name):
    plt.tight_layout()
    plt.savefig(FIGURES / f'{name}.png', bbox_inches='tight')
    plt.show()
    plt.close()

packages = ['numpy','pandas','pyarrow','matplotlib','seaborn','tokenizers',
            'nbformat','nbclient','nbconvert','ipykernel','jupyterlab','tabulate']
environment = {'python': platform.python_version(), 'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in packages}}
(ROOT / 'reports/environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
print('Thư mục dự án:', ROOT)
display(pd.Series(environment['packages'], name='version').to_frame())
""")
md("""
## Bước 2 — Tải đúng phiên bản và kiểm tra dấu vân tay

Nguồn: [GoEmotions trên Hugging Face](https://huggingface.co/datasets/google-research-datasets/go_emotions).
Revision cố định: `add492243ff905527e67aeb8b80c082af02207c3`, cấu hình `simplified`.
Ba file tổng cộng khoảng 3,5 MB. Lần sau đọc cache cục bộ; luôn kiểm SHA-256.

`load_goemotions` tải file, đối chiếu hash cố định trong `src/datasets/goemotions.py`, đọc schema,
kiểm tra số dòng và bảo đảm mapping nhãn giống nhau giữa ba split.
Sai hash/schema sẽ dừng, tránh âm thầm phân tích một phiên bản khác.
Manifest lưu URL, revision, thời điểm kiểm chứng, dung lượng, hash đầy đủ và số dòng.
""")
code("""
frames, label_names, manifest = load_goemotions(ROOT)
display(pd.DataFrame(manifest['files'])[['split','rows','bytes','sha256']])
print('Revision:', manifest['revision'])
print('Tổng số mẫu:', sum(map(len, frames.values())))
""")
md("""
## Bước 3 — Hiểu cấu trúc dữ liệu và 28 nhãn

Mỗi dòng là một bình luận đã có tập nhãn: `id` là mã bình luận, `text` là nguyên văn,
`labels` là danh sách ID nhãn (đánh số từ 0). Đây không phải bảng từng lượt gán nhãn
của từng annotator trong cấu hình `raw`.

**Đa lớp** thường chọn đúng một lớp; **đa nhãn** cho phép nhiều nhãn cùng bằng 1.
Ví dụ `[8, 20]` tương ứng `desire` và `optimism`, không phải lớp số 820.
Mapping phải lấy từ metadata, không sắp tên theo tần suất rồi dùng thứ tự đó để mã hóa.
""")
code("""
vi_names = ['ngưỡng mộ','thích thú','tức giận','khó chịu','tán thành','quan tâm',
            'bối rối','tò mò','mong muốn','thất vọng','không tán thành','ghê tởm',
            'ngượng ngùng','hào hứng','sợ hãi','biết ơn','đau buồn sâu sắc','vui vẻ',
            'yêu thương','lo lắng hồi hộp','lạc quan','tự hào','nhận ra','nhẹ nhõm',
            'hối hận','buồn bã','ngạc nhiên','trung tính']
label_mapping = pd.DataFrame({'label_id': range(28), 'label': label_names,
                              'dien_giai_vi': vi_names})
save_table(label_mapping, 'label_mapping')
display(label_mapping)
display(frames['train'].head(3))
display(frames['train'].dtypes.to_frame('dtype'))
""")
md("""
## Bước 4 — Số mẫu và chất lượng dữ liệu

Đếm thiếu giá trị, text rỗng sau `strip`, nhãn rỗng, nhãn trùng trong một mẫu,
nhãn ngoài miền 0–27 và ID lặp. Text chỉ được strip trong phép kiểm tra;
cột `text` gốc được giữ nguyên. `duplicate_*_extra_rows` đếm các dòng dư sau
bản ghi đầu tiên, khác với số nhóm trùng hoặc tổng số dòng tham gia nhóm trùng.
""")
code("""
quality_rows = []
for split, df in frames.items():
    quality_rows.append({
        'split': split, 'n_samples': len(df),
        'missing_text': int(df.text.isna().sum()), 'missing_id': int(df.id.isna().sum()),
        'blank_text': int(df.text.str.strip().eq('').sum()),
        'empty_labels': int(df.labels.map(len).eq(0).sum()),
        'repeated_labels': int(df.labels.map(lambda v: len(v) != len(set(v))).sum()),
        'invalid_labels': int(df.labels.map(lambda v: any(not 0 <= j < 28 for j in v)).sum()),
        'unique_ids': df.id.nunique(), 'unique_texts': df.text.nunique(),
        'duplicate_id_extra_rows': int(df.id.duplicated().sum()),
        'duplicate_text_extra_rows': int(df.text.duplicated().sum())})
quality = pd.DataFrame(quality_rows)
save_table(quality, 'data_quality')
display(quality)
checks = ['missing_text','missing_id','blank_text','empty_labels','repeated_labels',
          'invalid_labels','duplicate_id_extra_rows']
assert quality[checks].to_numpy().sum() == 0, 'Có bất thường cần xem trước khi phân tích.'
split_sizes = quality[['split','n_samples']].copy()
split_sizes['percent_all_samples'] = split_sizes.n_samples / split_sizes.n_samples.sum() * 100
display(split_sizes.round(4))
save_table(split_sizes, 'split_sizes')
plt.figure(figsize=(8, 4))
ax = sns.barplot(data=split_sizes, x='split', y='n_samples', color='#3274A1')
ax.bar_label(ax.containers[0], fmt='%d')
ax.set(xlabel='Tập dữ liệu', ylabel='Số bình luận', title='Kích thước ba split chính thức')
ax.set_ylim(0, split_sizes.n_samples.max() * 1.15)
finish_figure('01_split_sizes')
""")
md("""
**Cách đọc:** tổng `simplified` phải là 54.263, khác khoảng 58 nghìn bình luận
được mô tả cho bộ gốc trong [bài báo ACL 2020](https://aclanthology.org/2020.acl-main.372/).
Khoảng 80%/10%/10% là split do nguồn cung cấp, không phải phép chia ngẫu nhiên mới của nhóm.
""")
md("""
## Bước 5 — Mã hóa multi-hot và kiểm chứng

Tạo `Y` kích thước N × 28: `Y[i, j] = 1` khi mẫu i có nhãn j, còn lại bằng 0.
Tổng một hàng là số nhãn của mẫu; tổng một cột là số mẫu mang nhãn đó (support).
EDA lưu `uint8` để tiết kiệm bộ nhớ; khi huấn luyện BCE cần chuyển nhãn sang float.
""")
code("""
Y = {split: multi_hot(df.labels.tolist()) for split, df in frames.items()}
for split, matrix in Y.items():
    assert matrix.shape == (len(frames[split]), 28)
    assert np.array_equal(matrix.sum(axis=1), frames[split].labels.map(len).to_numpy())
    print(split, matrix.shape, matrix.dtype)
example_i = next(i for i, labels in enumerate(frames['train'].labels) if len(labels) > 1)
display(frames['train'].iloc[[example_i]])
display(pd.DataFrame(Y['train'][[example_i]], columns=label_names))
print('Nhãn được giải mã:', [label_names[j] for j in np.flatnonzero(Y['train'][example_i])])
""")
md("""
## Bước 6 — Phân bố đầy đủ 28 nhãn

Với nhãn j và split s: `support = sum(Y_s[:, j])`,
`prevalence (%) = 100 × support / số mẫu của split`.
Một mẫu đa nhãn được tính vào nhiều cột, nên **tổng tỷ lệ có thể vượt 100%**.
`share_of_assignments` có mẫu số là tổng lượt gán nhãn, là một đại lượng khác.
Notebook xuất cả hai, nhưng so sánh mức phổ biến giữa split bằng prevalence.
""")
code("""
label_stats = label_mapping.copy()
for split in SPLITS:
    counts = Y[split].sum(axis=0).astype(int)
    label_stats[f'{split}_count'] = counts
    label_stats[f'{split}_prevalence_pct'] = counts / len(frames[split]) * 100
label_stats['total_count'] = label_stats[[f'{s}_count' for s in SPLITS]].sum(axis=1)
label_stats['total_prevalence_pct'] = label_stats.total_count / sum(map(len, frames.values())) * 100
label_stats['train_share_of_assignments_pct'] = label_stats.train_count / label_stats.train_count.sum() * 100
save_table(label_stats, 'label_distribution')
display(label_stats[['label_id','label','train_count','validation_count','test_count',
                     'total_count','train_prevalence_pct']].round(3))
ordered = label_stats.sort_values('train_count', ascending=True)
fig, ax = plt.subplots(figsize=(11, 10))
ax.barh(ordered.label, ordered.train_count, color='#3274A1')
for i, (_, row) in enumerate(ordered.iterrows()):
    ax.text(row.train_count + 80, i, f'{row.train_count:,} ({row.train_prevalence_pct:.2f}%)', va='center', fontsize=8)
ax.set(xlim=(0, ordered.train_count.max()*1.27), xlabel='Số mẫu mang nhãn trong train',
       ylabel='', title='Phân bố 28 nhãn — train; một bình luận có thể được tính nhiều lần')
finish_figure('02_label_distribution_train')
prevalence = label_stats.set_index('label')[[f'{s}_prevalence_pct' for s in SPLITS]]
prevalence.columns = list(SPLITS)
plt.figure(figsize=(8, 11))
sns.heatmap(prevalence, annot=True, fmt='.2f', cmap='Blues', cbar_kws={'label': '% mẫu của split'})
plt.title('Tỷ lệ mẫu dương theo nhãn và split (%)')
plt.xlabel('Tập dữ liệu'); plt.ylabel('Nhãn')
finish_figure('03_prevalence_by_split')
""")
md("""
## Bước 7 — Mất cân bằng và năm nhãn hiếm

Năm nhãn hiếm được xác định **chỉ bằng train**, sắp theo support tăng dần.
Tỷ số `max support / min support` mô tả độ mất cân bằng nhưng không đo chất lượng mô hình.
Xem thêm tỷ số khi bỏ neutral để thấy sự chênh lệch giữa riêng 27 cảm xúc.
""")
code("""
rare = label_stats.sort_values(['train_count','label_id']).head(5).copy()
rare_names = rare.label.tolist()
imbalance_ratio = label_stats.train_count.max() / label_stats.train_count.min()
emotions_only = label_stats[label_stats.label != 'neutral']
emotion_ratio = emotions_only.train_count.max() / emotions_only.train_count.min()
display(rare[['label','train_count','validation_count','test_count','train_prevalence_pct']].round(4))
save_table(rare, 'rare_labels_train_defined')
print(f'Tỷ số lớn nhất/nhỏ nhất (28 nhãn): {imbalance_ratio:.2f} lần')
print(f'Tỷ số lớn nhất/nhỏ nhất (27 cảm xúc): {emotion_ratio:.2f} lần')
display(Markdown('**Nhận xét:** ' + ', '.join(rare_names) +
    ' là 5 nhãn ít mẫu nhất trên train. Số mẫu ít làm việc học và đo F1 từng nhãn kém ổn định. '
    'Theo kế hoạch nhóm, cần báo Macro-F1 cùng Micro-F1 và support từng nhãn; '
    'mọi trọng số lớp phải tính trên train, mọi ngưỡng phải chọn trên validation.'))
""")
md("""
## Bước 8 — Một mẫu có bao nhiêu nhãn

**Cardinality** = tổng lượt gán nhãn / số mẫu, tức số nhãn trung bình trên một bình luận.
**Density** = cardinality / 28, tức tỷ lệ ô bằng 1 trong ma trận Y.
**Tỷ lệ đa nhãn** = số mẫu có ≥2 nhãn / số mẫu. Ba đại lượng này không đồng nghĩa.
""")
code("""
multi_rows, cardinality_rows = [], []
for split, df in frames.items():
    k = df.labels.map(len)
    multi_rows.append({'split': split, 'n_samples': len(df), 'label_assignments': int(k.sum()),
                      'one_label': int((k == 1).sum()), 'two_labels': int((k == 2).sum()),
                      'three_or_more': int((k >= 3).sum()), 'multi_label_samples': int((k >= 2).sum()),
                      'multi_label_pct': (k >= 2).mean()*100, 'cardinality': k.mean(),
                      'density': k.mean()/28, 'max_labels': int(k.max())})
    for n, count in k.value_counts().sort_index().items():
        cardinality_rows.append({'split': split, 'n_labels': n, 'n_samples': count, 'percent': count/len(df)*100})
multi_stats = pd.DataFrame(multi_rows)
cardinality_dist = pd.DataFrame(cardinality_rows)
display(multi_stats.round(5)); display(cardinality_dist.round(3))
save_table(multi_stats, 'multilabel_summary'); save_table(cardinality_dist, 'labels_per_sample')
plt.figure(figsize=(9, 4))
sns.barplot(data=cardinality_dist, x='n_labels', y='percent', hue='split')
plt.xlabel('Số nhãn trên một bình luận'); plt.ylabel('% số mẫu trong split')
plt.title('Đơn nhãn chiếm đa số, nhưng dữ liệu vẫn là bài toán đa nhãn')
finish_figure('04_labels_per_sample')
""")
md("""
## Bước 9 — Neutral và các nhãn cùng xuất hiện

Tách `neutral` đứng một mình khỏi `neutral` cùng nhãn khác. Không tự đặt quy tắc
“có neutral thì xóa mọi cảm xúc”, vì sẽ thay ground truth của dữ liệu.

Với train, `C = Y.T @ Y` đếm số mẫu cùng có hai nhãn. Đường chéo là support,
không phải cặp hai nhãn khác nhau. Chuyển sang **int64 trước khi nhân** để tránh
tràn số `uint8`. Heatmap dùng log(1 + count) để thấy cả cặp phổ biến lẫn hiếm.
Jaccard = giao / hợp, nằm trong [0,1]; không phải quan hệ nhân quả hay độ giống nghĩa.
""")
code("""
neutral_id = label_names.index('neutral')
neutral_rows = []
for split, matrix in Y.items():
    has_neutral = matrix[:, neutral_id].astype(bool)
    is_multi = matrix.sum(axis=1) > 1
    neutral_rows.append({'split': split, 'neutral_total': int(has_neutral.sum()),
        'neutral_only': int((has_neutral & ~is_multi).sum()),
        'neutral_with_other': int((has_neutral & is_multi).sum()),
        'neutral_with_other_pct_all': (has_neutral & is_multi).mean()*100})
neutral_stats = pd.DataFrame(neutral_rows)
display(neutral_stats.round(3)); save_table(neutral_stats, 'neutral_summary')
y64 = Y['train'].astype(np.int64)
co = y64.T @ y64
support = np.diag(co)
pairs = pd.DataFrame([
    {'label_a': label_names[a], 'label_b': label_names[b], 'count': int(co[a,b]),
     'percent_train': co[a,b]/len(y64)*100,
     'jaccard': co[a,b]/(support[a]+support[b]-co[a,b])}
    for a,b in combinations(range(28), 2)
]).sort_values(['count','label_a','label_b'], ascending=[False, True, True])
display(pairs.head(15).round(4)); save_table(pairs, 'label_cooccurrence_pairs_train')
pd.DataFrame(co, index=label_names, columns=label_names).to_csv(TABLES/'cooccurrence_matrix_train.csv', encoding='utf-8-sig')
plt.figure(figsize=(13, 11))
sns.heatmap(np.log1p(co), mask=np.eye(28, dtype=bool), cmap='YlGnBu',
            xticklabels=label_names, yticklabels=label_names, cbar_kws={'label':'log(1 + số mẫu)'})
plt.title('Đồng xuất hiện trên train — ẩn đường chéo'); plt.xticks(rotation=90); plt.yticks(rotation=0)
finish_figure('05_cooccurrence_train')
""")
md("""
## Bước 10 — Kiểm tra trùng lặp và nguy cơ rò rỉ

Kiểm tra ID giao nhau giữa các split và text **trùng nguyên chuỗi**, phân biệt:
(1) số chuỗi khác nhau bị trùng, (2) số dòng bị ảnh hưởng ở mỗi phía.
Hai dòng có text giống nhau vẫn có thể có ID và nhãn khác nhau.

Kiểm tra thêm chuẩn hóa nhẹ `casefold + gộp khoảng trắng` là phép chẩn đoán riêng,
không phải bằng chứng hai bản ghi có cùng nguồn gốc, không thay đổi dữ liệu gốc.
Không đủ metadata trong simplified để kết luận trùng tác giả hoặc cùng cuộc hội thoại.
""")
code("""
overlap_rows = []
for a,b in combinations(SPLITS, 2):
    left, right = frames[a], frames[b]
    id_overlap = set(left.id) & set(right.id)
    exact = set(left.text) & set(right.text)
    normalize = lambda s: s.str.casefold().str.replace(r'\\s+', ' ', regex=True).str.strip()
    ln, rn = normalize(left.text), normalize(right.text)
    norm_overlap = set(ln) & set(rn)
    overlap_rows.append({'left_split': a, 'right_split': b, 'shared_ids': len(id_overlap),
        'shared_exact_texts': len(exact), 'left_rows_exact': int(left.text.isin(exact).sum()),
        'right_rows_exact': int(right.text.isin(exact).sum()),
        'shared_normalized_texts': len(norm_overlap), 'right_rows_normalized': int(rn.isin(norm_overlap).sum())})
overlap = pd.DataFrame(overlap_rows)
assert overlap.shared_ids.sum() == 0
display(overlap); save_table(overlap, 'cross_split_overlap')
duplicate_rows = []
for split, df in frames.items():
    temp = df.assign(label_set=df.labels.map(lambda x: tuple(sorted(x))))
    groups = temp.groupby('text').agg(n_rows=('id','size'), n_label_sets=('label_set','nunique'))
    duplicate_rows.append({'split': split, 'duplicate_text_groups': int((groups.n_rows > 1).sum()),
        'rows_in_duplicate_groups': int(groups.loc[groups.n_rows > 1,'n_rows'].sum()),
        'duplicate_extra_rows': len(df)-len(groups),
        'duplicate_groups_with_different_labels': int(((groups.n_rows > 1)&(groups.n_label_sets > 1)).sum())})
duplicates = pd.DataFrame(duplicate_rows)
display(duplicates); save_table(duplicates, 'within_split_duplicates')
test_not_in_train = ~frames['test'].text.isin(set(frames['train'].text))
clean_test_ids = frames['test'].loc[test_not_in_train, ['id']]
save_table(clean_test_ids, 'test_ids_without_exact_train_overlap')
print('Test chính giữ nguyên:', len(frames['test']))
print('Test phân tích độ nhạy, loại text trùng train:', len(clean_test_ids))
print('Danh sách trên chưa loại text trùng validation; không gọi đây là tập hoàn toàn độc lập.')
""")
md("""
## Bước 11 — Độ dài văn bản

Đếm ký tự bằng `len(text)` (Unicode code points) và “từ” bằng `text.split()`
(đơn vị cách nhau bởi khoảng trắng). Đây là hai phép mô tả đơn giản,
**không phải số subword token** của BERT/RoBERTa.
P50 là trung vị; P95/P99 cho biết vùng đuôi của phân bố, không chứng minh rằng
chọn đúng mốc đó là cấu hình tối ưu.
""")
code("""
length_frames, length_rows = {}, []
for split, df in frames.items():
    lengths = pd.DataFrame({'id': df.id, 'n_chars': df.text.str.len(),
                             'n_words_whitespace': df.text.str.split().str.len()})
    length_frames[split] = lengths
    for feature in ['n_chars','n_words_whitespace']:
        values = lengths[feature]
        length_rows.append({'split': split, 'measure': feature, 'min': values.min(),
            'mean': values.mean(), 'std': values.std(), 'p50': values.quantile(.5),
            'p90': values.quantile(.9), 'p95': values.quantile(.95),
            'p99': values.quantile(.99), 'max': values.max()})
length_summary = pd.DataFrame(length_rows)
display(length_summary.round(2)); save_table(length_summary, 'text_length_summary')
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(length_frames['train'].n_words_whitespace, bins=35, ax=axes[0], color='#3274A1')
axes[0].set(xlabel='Đơn vị tách bằng khoảng trắng', ylabel='Số mẫu', title='Độ dài train theo khoảng trắng')
sns.histplot(length_frames['train'].n_chars, bins=50, ax=axes[1], color='#D48433')
axes[1].set(xlabel='Số ký tự Unicode', ylabel='Số mẫu', title='Độ dài train theo ký tự')
finish_figure('06_text_lengths_train')
display(Markdown('**Lưu ý:** giữ dấu câu, chữ hoa, emoji và phủ định trong dữ liệu gốc. '
    'Chưa chọn `max_length` từ số từ; bước kế tiếp đo đúng tokenizer.'))
""")
md("""
## Bước 12 — Độ dài theo tokenizer của ba kiến trúc

Đọc `tokenizer.json` chính thức của BERT-base-uncased, RoBERTa-base và DistilBERT-base-uncased,
ghim commit trong `data/tokenizer_revisions.json`. Dùng thư viện Rust-backed `tokenizers`;
chỉ tải tokenizer, không tải trọng số. Cấu hình mặc định cho **một văn bản**,
`add_special_tokens=True`, không padding/truncation khi đo.

Ví dụ BERT có `[CLS]` và `[SEP]`, nên số token đã gồm các token đặc biệt.
Báo tỷ lệ `length > max_length` tại 64/128/256 và số token sẽ bị cắt.
Đo trên **train và validation**; không dùng test để ra quyết định độ dài.
BERT và DistilBERT có thể trùng kết quả vì dùng cùng cách tách từ; đây không phải lỗi.
Khi chuyển sang Transformers, giữ đúng revision/default và kiểm tra lại nếu thay cấu hình tokenizer.
""")
code("""
from tokenizers import Tokenizer
tokenizer_revisions = json.loads((ROOT/'data/tokenizer_revisions.json').read_text(encoding='utf-8'))
token_rows, tokenizer_manifest, token_lengths = [], [], {}
for model_id, revision in tokenizer_revisions.items():
    model_short = {'google-bert/bert-base-uncased':'BERT', 'FacebookAI/roberta-base':'RoBERTa',
                   'distilbert/distilbert-base-uncased':'DistilBERT'}[model_id]
    url = f'https://huggingface.co/{model_id}/resolve/{revision}/tokenizer.json'
    path = download(url, ROOT/'data/tokenizers'/model_short/revision/'tokenizer.json')
    tok = Tokenizer.from_file(str(path))
    tok.no_truncation(); tok.no_padding()
    tokenizer_manifest.append({'name': model_short, 'model_id': model_id, 'revision': revision,
        'url': url, 'sha256': sha256(path), 'special_tokens_included': True})
    for split in ['train','validation']:
        texts = frames[split].text.tolist()
        counts = []
        for start in range(0, len(texts), 512):
            counts.extend(len(enc.ids) for enc in tok.encode_batch(texts[start:start+512], add_special_tokens=True))
        values = np.asarray(counts)
        token_lengths[(model_short, split)] = values
        row = {'tokenizer': model_short, 'split': split, 'n_samples': len(values),
               'mean': values.mean(), 'p50': np.quantile(values,.5), 'p95': np.quantile(values,.95),
               'p99': np.quantile(values,.99), 'max': int(values.max())}
        for limit in [64,128,256]:
            row[f'over_{limit}_count'] = int((values > limit).sum())
            row[f'over_{limit}_pct'] = (values > limit).mean()*100
            row[f'tokens_removed_at_{limit}'] = int(np.maximum(values-limit, 0).sum())
        token_rows.append(row)
token_summary = pd.DataFrame(token_rows)
save_table(token_summary, 'tokenizer_lengths_train_validation')
(ROOT/'data/tokenizer_manifest.json').write_text(json.dumps(tokenizer_manifest, indent=2), encoding='utf-8')
display(token_summary[['tokenizer','split','n_samples','mean','p50','p95','p99','max']].round(3))
truncation = pd.DataFrame([
    {'tokenizer': row['tokenizer'], 'split': row['split'], 'max_length': limit,
     'samples_truncated': row[f'over_{limit}_count'], 'percent_truncated': row[f'over_{limit}_pct'],
     'tokens_removed': row[f'tokens_removed_at_{limit}']}
    for row in token_rows for limit in [64,128,256]])
save_table(truncation, 'tokenizer_truncation')
display(truncation.round(5))
outlier_rows = []
for name in ['BERT','RoBERTa','DistilBERT']:
    values = token_lengths[(name,'train')]
    for i in np.argsort(values, kind='stable')[-3:][::-1]:
        row = frames['train'].iloc[i]
        outlier_rows.append({'tokenizer': name, 'id': row.id, 'n_tokens': int(values[i]),
            'n_chars': len(row.text), 'n_words_whitespace': len(row.text.split()),
            'text_preview': row.text[:100] + ('…' if len(row.text)>100 else '')})
token_outliers = pd.DataFrame(outlier_rows)
save_table(token_outliers, 'tokenizer_outliers_train')
display(token_outliers)
fig, ax = plt.subplots(figsize=(9, 4))
for name in ['BERT','RoBERTa','DistilBERT']:
    values = token_lengths[(name,'train')]
    xs = np.sort(values)
    ax.plot(xs, np.arange(1,len(xs)+1)/len(xs)*100, label=name,
            linestyle='--' if name == 'DistilBERT' else '-')
ax.set(xlabel='Số subword token, gồm special tokens', ylabel='% mẫu có độ dài ≤ x',
       title='Phân bố tích lũy độ dài token — train, vùng 0–80 token', ylim=(0,101), xlim=(0,80))
ax.legend(); finish_figure('07_token_length_ecdf_train')
display(Markdown('**Diễn giải:** 128 là ứng viên khởi đầu của kế hoạch. Bảng trên cho biết '
    'bao nhiêu mẫu sẽ bị cắt, chưa cho biết ảnh hưởng tới F1. '
    'Biểu đồ phóng vùng 0–80 token; các mẫu dài hơn vẫn có trong bảng và phép đếm. '
    'Chuỗi Unicode/emoji hoặc ký tự lặp có thể dài sau subword tokenization dù ít khoảng trắng. '
    'Chốt cấu hình sau pilot trên train/validation và giữ cố định trước đánh giá test.'))
""")
md("""
## Bước 13 — Ba mươi ví dụ từ train để nhóm xem thủ công

Chọn có chủ đích 28 ví dụ (mỗi nhãn một ví dụ) và thêm một mẫu ≥3 nhãn,
một mẫu neutral kèm cảm xúc. Ưu tiên 3–20 đơn vị khoảng trắng và đơn nhãn nếu có;
sắp theo ID rồi lấy mẫu với seed 42. Các ID không lặp, nguyên văn không bị sửa.
Đây là **bộ minh họa phủ nhãn**, không phải mẫu ngẫu nhiên đại diện để ước lượng tỷ lệ.

`selection_reason` giải thích cách chọn, không phải giải thích đúng/sai của nhãn.
Cột `group_review_note` để nhóm bổ sung nhận xét riêng. Nếu có file
`references/example_notes.json`, notebook ghép thêm ghi chú đọc nội dung của người chuẩn bị;
những ghi chú này là diễn giải, không thay ground truth hoặc chứng nhận nhãn đúng.
""")
code("""
train = frames['train']
used, example_rows = set(), []
def choose_example(mask, reason, prefer_single=False):
    candidates = train.loc[mask & ~train.id.isin(used)].copy()
    words = candidates.text.str.split().str.len()
    short = candidates.loc[words.between(3,20)]
    if len(short):
        candidates = short
    if prefer_single:
        single = candidates.loc[candidates.labels.map(len).eq(1)]
        if len(single):
            candidates = single
    row = candidates.sort_values('id').sample(n=1, random_state=SEED).iloc[0]
    used.add(row.id)
    example_rows.append({'split':'train', 'id':row.id, 'text':row.text,
        'label_ids': ','.join(map(str,row.labels)),
        'labels': ', '.join(label_names[j] for j in row.labels),
        'n_labels': len(row.labels), 'selection_reason': reason, 'group_review_note':''})
for j, name in enumerate(label_names):
    choose_example(train.labels.map(lambda labels: j in labels), f'Minh họa nhãn {name}', prefer_single=True)
choose_example(train.labels.map(len).ge(3), 'Minh họa một bình luận có ít nhất 3 nhãn')
choose_example(train.labels.map(lambda labels: neutral_id in labels and len(labels)>1), 'Neutral cùng cảm xúc khác')
examples = pd.DataFrame(example_rows)
notes_path = ROOT/'references/example_notes.json'
notes = json.loads(notes_path.read_text(encoding='utf-8')) if notes_path.exists() else {}
examples['reading_note'] = examples.id.map(notes).fillna('')
assert len(examples) == examples.id.nunique() == 30
assert set(range(28)) == {int(j) for value in examples.label_ids for j in value.split(',')}
save_table(examples, 'examples_30')
with pd.option_context('display.max_colwidth', None, 'display.max_rows', 40):
    display(examples[['id','text','labels','reading_note']])
display(Markdown('**Cách thảo luận:** câu có từ chỉ cảm xúc rõ hay cần ngữ cảnh? '
    'Có phủ định, hàm ý, nhãn gần nghĩa, hoặc neutral đồng xuất hiện không? '
    'Nhóm ghi nhận xét vào một bản sao của CSV để lần chạy lại không ghi đè phần ghi chú.'))
""")
md("""
## Bước 14 — Đối chiếu số đo với snapshot cố định

Các hằng số dưới đây là số đo EDA trước đó của snapshot cố định, dùng làm **mốc kiểm chứng**,
không dùng thay phép đếm. Nếu dữ liệu, mapping hoặc thuật toán sai, ô này dừng.
Đối chiếu toàn bộ 28 support của từng split, số mẫu theo số nhãn, neutral và trùng text.
""")
code("""
expected_counts = {
 'train':[4130,2328,1567,2470,2939,1087,1368,2191,641,1269,2022,793,303,853,596,2662,77,1452,2086,164,1581,111,1110,153,545,1326,1060,14219],
 'validation':[488,303,195,303,397,153,152,248,77,163,292,97,35,96,90,358,13,172,252,21,209,15,127,18,68,143,129,1766],
 'test':[504,264,198,320,351,135,153,284,83,151,267,123,37,103,78,352,6,161,238,23,186,16,145,11,56,156,141,1787]}
for split in SPLITS:
    np.testing.assert_array_equal(Y[split].sum(axis=0), expected_counts[split])
np.testing.assert_array_equal(multi_stats.one_label, [36308,4548,4590])
np.testing.assert_array_equal(multi_stats.two_labels, [6541,809,774])
np.testing.assert_array_equal(multi_stats.three_or_more, [561,69,63])
np.testing.assert_array_equal(neutral_stats.neutral_with_other, [1396,174,181])
np.testing.assert_array_equal(quality.unique_texts, [43227,5423,5421])
np.testing.assert_array_equal(overlap.shared_exact_texts, [41,32,10])
np.testing.assert_array_equal(overlap.right_rows_exact, [43,37,13])
assert rare_names == ['grief','pride','relief','nervousness','embarrassment']
assert len(clean_test_ids) == 5390
verification = {'status':'passed', 'dataset_revision': REVISION,
                'checks':['sha256_3_files','schema_and_label_mapping','split_sizes','supports_28x3',
                          'label_cardinality','neutral_cooccurrence','duplicate_text','multi_hot','30_examples']}
(ROOT/'reports/verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
print('PASS: số liệu tính trực tiếp khớp các mốc EDA của snapshot cố định.')
""")
md("""
## Bước 15 — Xuất kết quả để bàn giao

Xuất `Y_train/Y_validation/Y_test` và ID theo đúng thứ tự trong file `.npz`.
File này dành cho bước mô hình sau; `.npz` không chứa văn bản, phải đối chiếu ID với raw.
Xuất bảng CSV UTF-8 BOM để Excel đọc tiếng Việt, ảnh PNG và báo cáo Markdown.
Tất cả số liệu trong báo cáo dưới đây được lấy từ biến đã tính, tránh gõ lại nhầm.
""")
code("""
for split in SPLITS:
    np.savez_compressed(ROOT/'data/processed'/f'{split}_multihot.npz',
                        Y=Y[split], ids=frames[split].id.to_numpy(dtype=str),
                        label_names=np.asarray(label_names, dtype=str))
    saved = np.load(ROOT/'data/processed'/f'{split}_multihot.npz', allow_pickle=False)
    assert np.array_equal(saved['Y'], Y[split])
    assert saved['ids'].tolist() == frames[split].id.tolist()
    assert saved['label_names'].tolist() == label_names

total_samples = sum(map(len, frames.values()))
total_multi = int(multi_stats.multi_label_samples.sum())
total_assignments = int(multi_stats.label_assignments.sum())
summary = {'samples': total_samples, 'n_labels': len(label_names),
           'label_assignments': total_assignments, 'multi_label_samples': total_multi,
           'multi_label_pct': total_multi/total_samples*100,
           'cardinality': total_assignments/total_samples,
           'density': total_assignments/(total_samples*28),
           'train_imbalance_ratio': float(imbalance_ratio),
           'rare_labels': rare_names, 'neutral_with_other': int(neutral_stats.neutral_with_other.sum()),
           'test_without_exact_train_overlap': len(clean_test_ids)}
(ROOT/'reports/summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

sections = [
 '# Thống kê dữ liệu đã khám phá của đồ án GoEmotions',
 '## 1. Phạm vi và nguồn dữ liệu',
 'Phần thống kê phục vụ đề tài **Phân loại cảm xúc đa nhãn trên văn bản mạng xã hội với GoEmotions**. '
 'Sử dụng cấu hình simplified, bình luận tiếng Anh trên Reddit, 27 cảm xúc cùng neutral. '
 'Một bình luận có thể có nhiều nhãn. Ngôn ngữ thực hiện là Python; công cụ gồm pandas, NumPy, '
 'PyArrow, Matplotlib, Seaborn, Hugging Face Tokenizers và Jupyter. Chạy CPU, không cần GPU.',
 f'Revision được ghim: `{REVISION}`. Các số liệu dưới đây tính trực tiếp từ ba tệp Parquet '
 'và đã đối chiếu SHA-256 cố định trong module dữ liệu. Xem docs/NOTEBOOKS.md '
 'để hiểu vai trò của EDA trong pipeline.',
 'Nguồn: [Google Research README](https://github.com/google-research/google-research/blob/master/goemotions/README.md), '
 '[dataset card](https://huggingface.co/datasets/google-research-datasets/go_emotions), '
 '[bài báo ACL 2020](https://aclanthology.org/2020.acl-main.372/). '
 'Không nhầm khoảng 58 nghìn bình luận của bộ gốc với 54.263 mẫu trong simplified.',
 '## 2. Số mẫu và cấu trúc',
 split_sizes.round(4).to_markdown(index=False),
 'Mỗi dòng có `id` (mã bình luận), `text` (nguyên văn) và `labels` (danh sách ID nhãn từ 0 đến 27). '
 'Giữ nguyên split chính thức. ID không trùng giữa các split; không có text thiếu hoặc rỗng, '
 'không có tập nhãn rỗng hoặc nhãn ngoài miền.',
 '![Số mẫu](figures/01_split_sizes.png)',
 '## 3. Phân bố nhãn và mất cân bằng',
 'Support là số bình luận mang nhãn. Một dòng đa nhãn được tính vào nhiều nhãn; '
 'tổng support có thể lớn hơn số mẫu. Prevalence (%) = support / số mẫu của split × 100.',
 label_stats[['label_id','label','dien_giai_vi','train_count','validation_count','test_count',
              'total_count','train_prevalence_pct']].round(3).to_markdown(index=False),
 f'Train có tỷ số support nhãn phổ biến nhất/hiếm nhất bằng **{imbalance_ratio:.2f} lần**; '
 f'nếu chỉ xét 27 cảm xúc, tỷ số là **{emotion_ratio:.2f} lần**. '
 'Năm nhãn hiếm chỉ xác định từ train để không dùng test định hướng thí nghiệm:',
 rare[['label','train_count','validation_count','test_count']].to_markdown(index=False),
 'Neutral phổ biến nhất nhưng không nên chỉ dự đoán neutral. Theo kế hoạch, báo Macro-F1 '
 'làm chỉ số chính, kèm Micro-F1 và F1/support từng nhãn; chú ý độ bất ổn ở nhãn hiếm.',
 '![Phân bố train](figures/02_label_distribution_train.png)',
 '![Tỷ lệ theo split](figures/03_prevalence_by_split.png)',
 '## 4. Đặc tính đa nhãn',
 multi_stats.round(5).to_markdown(index=False),
 f'Toàn bộ dữ liệu có **{total_assignments:,} lượt gán nhãn**, **{total_multi:,} mẫu đa nhãn '
 f'({total_multi/total_samples*100:.2f}%)**, cardinality **{total_assignments/total_samples:.4f}** '
 f'và density **{total_assignments/(total_samples*28):.5f}**. '
 'Cardinality là số nhãn trung bình/mẫu; density là tỷ lệ ô 1 trong ma trận N × 28.',
 '![Số nhãn mỗi mẫu](figures/04_labels_per_sample.png)',
 neutral_stats.round(4).to_markdown(index=False),
 'Neutral có thể cùng xuất hiện với cảm xúc khác. Giữ nguyên ground truth; không dùng softmax '
 'để ép các lớp loại trừ nhau trong bài toán này. Các cặp đồng xuất hiện nhiều nhất trên train:',
 pairs.head(10).round(4).to_markdown(index=False),
 'Jaccard = số mẫu cùng có hai nhãn / số mẫu có ít nhất một trong hai nhãn. '
 'Đồng xuất hiện là quan hệ thống kê, không khẳng định hai cảm xúc giống nghĩa hay có quan hệ nhân quả.',
 '![Đồng xuất hiện](figures/05_cooccurrence_train.png)',
 '## 5. Chất lượng và trùng lặp',
 quality.to_markdown(index=False), duplicates.to_markdown(index=False), overlap.to_markdown(index=False),
 'Có 32 chuỗi trùng nguyên văn giữa train và test, ảnh hưởng 37 dòng test. '
 'Báo cáo benchmark chính giữ đủ 5.427 dòng test; phân tích độ nhạy dùng 5.390 dòng '
 'không trùng text với train. Danh sách này chưa loại các dòng trùng validation và không '
 'bảo đảm hết mọi dạng rò rỉ ngữ nghĩa. Phép chuẩn hóa nhẹ chỉ là chẩn đoán bổ sung. '
 'Không thể kết luận trùng người viết/cuộc hội thoại chỉ từ ba cột simplified.',
 '## 6. Độ dài văn bản và tokenizer',
 length_summary.round(2).to_markdown(index=False),
 'Ký tự được đếm theo Unicode code points; từ ở bảng trên chỉ là đơn vị cách nhau bởi khoảng trắng. '
 'Hai thước đo này khác subword token của mô hình.',
 '![Độ dài văn bản](figures/06_text_lengths_train.png)',
 token_summary[['tokenizer','split','n_samples','mean','p50','p95','p99','max']].round(3).to_markdown(index=False),
 truncation.round(5).to_markdown(index=False),
 'Token được đo bằng tokenizer.json chính thức, revision được lưu trong data/tokenizer_manifest.json. '
 'Đã gồm special tokens, không padding hay truncation khi đo, chỉ dùng train/validation. '
 'Mốc 128 là đề xuất thử nghiệm; số mẫu bị cắt không tự chứng minh tác động lên F1. '
 'Biểu đồ chỉ phóng vùng 0–80 token; bảng đã tính cả ngoại lệ dài. Các mẫu ngoại lệ '
 'và ID được lưu ở tables/tokenizer_outliers_train.csv. '
 'Giữ dấu câu, phủ định, emoji và text gốc; chốt max_length qua pilot trước khi đánh giá test.',
 '![Độ dài token](figures/07_token_length_ecdf_train.png)',
 '## 7. Ví dụ dữ liệu',
 'Bộ 30 ví dụ từ train được chọn để phủ đủ 28 nhãn, thêm trường hợp ≥3 nhãn và neutral '
 'đồng xuất hiện. ID không lặp, seed 42. Đây là mẫu minh họa có chủ đích; không dùng để '
 'ước lượng phân bố. Ghi chú đọc nội dung là diễn giải, không phải nhãn thay thế.',
 examples[['id','text','labels','reading_note']].to_markdown(index=False),
 '## 8. Kết luận đưa vào báo cáo tiến độ',
 f'Nhóm đã khảo sát {total_samples:,} bình luận thuộc GoEmotions simplified với 28 nhãn. '
 f'Dữ liệu gồm 43.410 mẫu train, 5.426 validation và 5.427 test; {total_multi/total_samples*100:.2f}% '
 'bình luận mang từ hai nhãn trở lên. Phân bố nhãn mất cân bằng rõ rệt: neutral phổ biến nhất, '
 'trong khi grief, pride, relief, nervousness và embarrassment là năm nhãn hiếm nhất trên train. '
 'Bộ dữ liệu không có văn bản rỗng nhưng có text trùng trong và giữa các split. '
 'Nhóm giữ nguyên bộ chia chuẩn, bảo toàn nhãn neutral đồng xuất hiện và lưu thêm danh sách '
 'test không trùng text với train để phân tích độ nhạy khi đánh giá mô hình. '
 'Các thống kê độ dài theo tokenizer là cơ sở thử max_length trên train/validation. '
 'Các kết luận hiện tại chỉ mô tả dữ liệu, chưa phải kết quả huấn luyện hoặc chất lượng dự đoán.',
 '## 9. Giới hạn và khả năng chạy lại',
 'Reddit tiếng Anh không đại diện cho toàn bộ mạng xã hội hoặc tiếng Việt. Nhãn do con người '
 'gán có thể mơ hồ và phụ thuộc ngữ cảnh; simplified không cung cấp đủ thông tin annotator '
 'để tái tính mức đồng thuận. Kiểm tra trùng nguyên văn không phát hiện diễn đạt lại. '
 'EDA test chỉ dùng mô tả; không dùng để chọn mô hình/ngưỡng. '
 'Theo [README nguồn](https://github.com/google-research/google-research/blob/master/goemotions/README.md), '
 'dữ liệu còn chịu thiên lệch từ nguồn Reddit và quá trình gán nhãn.',
 'Chạy `python -m scripts.data.run_eda` từ gốc repo sau khi cài extra eda,notebooks. '
 'Notebook, bảng, hình, manifest và báo cáo được sinh lại; phiên bản môi trường nằm ở '
 '`reports/environment.json`, trạng thái đối chiếu ở `reports/verification.json`. '
 'Nhận xét nhóm nên lưu ở bản sao CSV để không bị ghi đè khi chạy lại.',
]
(ROOT/'reports/THONG_KE_DU_LIEU.md').write_text('\\n\\n'.join(sections)+'\\n', encoding='utf-8')
display(pd.Series(summary, name='value').to_frame())
print('Đã lưu báo cáo:', ROOT/'reports/THONG_KE_DU_LIEU.md')
print('Số bảng CSV:', len(list(TABLES.glob('*.csv'))))
print('Số hình PNG:', len(list(FIGURES.glob('*.png'))))
""")
md("""
## Bước 16 — Tự kiểm tra và bàn giao cho phần mô hình

Sau notebook, nhóm cần giải thích được:

1. Vì sao tổng support lớn hơn số bình luận? Phân biệt prevalence với share of assignments.
2. Một hàng multi-hot có thể có bao nhiêu số 1? Vì sao không dùng `argmax` để lấy một nhãn duy nhất?
3. Vì sao neutral không được ép loại trừ những nhãn còn lại?
4. Vì sao xác định năm nhãn hiếm bằng train, chọn ngưỡng bằng validation?
5. Vì sao 32 chuỗi text trùng lại ảnh hưởng 37 dòng test?
6. Vì sao “30 từ” không có nghĩa “30 token BERT”?

**Bàn giao:** `data/manifest.json`, `data/labels.json`, ma trận N × 28 trong
`data/processed/`, 30 ví dụ và toàn bộ thống kê trong `reports/tables/`, bảy biểu đồ
trong `reports/figures/`, báo cáo `reports/THONG_KE_DU_LIEU.md`.
Mở `reports/eda.html` để đọc bản notebook đã chạy bằng trình duyệt.

**Bước tiếp theo của đồ án:** baseline TF-IDF + One-vs-Rest Logistic Regression;
fit vectorizer chỉ trên train. Zero-shot và fine-tune BERT/RoBERTa/DistilBERT
thuộc giai đoạn sau. EDA không thay thế các thực nghiệm A/B/C/D trong kế hoạch.

## Tài liệu tham khảo

- Cấu hình snapshot/checksum: `src/datasets/goemotions.py`; lộ trình: `docs/NOTEBOOKS.md`.
- [Demszky et al., ACL 2020](https://aclanthology.org/2020.acl-main.372/).
- [Google Research GoEmotions README](https://github.com/google-research/google-research/blob/master/goemotions/README.md).
- [Dataset card](https://huggingface.co/datasets/google-research-datasets/go_emotions).
- [Snapshot simplified được dùng](https://huggingface.co/datasets/google-research-datasets/go_emotions/tree/add492243ff905527e67aeb8b80c082af02207c3/simplified).
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/api/tokenizer).
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12.6'}})
nbf.validate(notebook)
(ROOT/'notebooks').mkdir(exist_ok=True)
nbf.write(notebook, ROOT/'notebooks/01_eda.ipynb')
print(f'Created notebooks/01_eda.ipynb: {len(cells)} cells')
