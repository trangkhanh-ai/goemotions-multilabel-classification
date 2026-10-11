from pathlib import Path
import json, csv, io, textwrap, hashlib, os, tempfile, sys, argparse, subprocess
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
runtime=ROOT/'notebook_runtime'
runtime.mkdir(exist_ok=True)
tempfile.tempdir=str(runtime)
for key in ('TMP','TEMP','JUPYTER_RUNTIME_DIR','JUPYTER_CONFIG_DIR','MPLCONFIGDIR','IPYTHONDIR'):
    os.environ[key]=str(runtime)
SOURCE_REF='3c44309ec287d3a2926acdd67a9b6a18da11a4d2'
try:
    SOURCE_REF=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
except (OSError,subprocess.CalledProcessError):
    pass
src={'ref':SOURCE_REF,'repository':'trangkhanh-ai/goemotions-multilabel-classification','files':{}}
def read(p):
    text=(ROOT/p).read_text(encoding='utf-8-sig')
    src['files'][p]={'content':text,'display_url':f"https://github.com/{src['repository']}/blob/{SOURCE_REF}/{p}"}
    return text
def rows(p): return list(csv.DictReader(io.StringIO(read(p))))
snapshot={
 'commit':src['ref'], 'repository':src['repository'],
 'all_runs':[r for r in rows('reports/project_results/all_runs.csv') if r['system'].startswith('C_')],
 'mean_std':[r for r in rows('reports/project_results/mean_std.csv') if r['system'].startswith('C_')],
 'rare':[r for r in rows('reports/project_results/rare_before_after.csv') if r['method']=='C' and r['architecture']=='roberta'],
 'metadata':{str(s):json.loads(read(f'reports/reproducibility/artifacts/data/processed/transformers/roberta/seed_{s}/full/standard/run_metadata.json')) for s in (42,123,2026)},
 'sources':{p:f"https://github.com/{src['repository']}/blob/{SOURCE_REF}/{p}" for p in ('reports/project_results/all_runs.csv','reports/project_results/mean_std.csv','reports/project_results/rare_before_after.csv','reports/error_case_studies.md','scripts/train_transformer.py','src/neural.py')},
}
nb=nbf.v4.new_notebook()
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s):cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))
md('''
# C2 — RoBERTa: phần của Đức Trí

**Mục tiêu:** hiểu code, trực quan hóa kết quả và dùng làm notebook báo cáo cá nhân.

Bạn phụ trách RoBERTa: dữ liệu/tokenizer → fine-tune → 3 seed → metrics → ngưỡng → lỗi → phần viết C2.
Zero-shot B là phần bàn giao/hỗ trợ Duy. BERT thuộc Khánh; DistilBERT và demo thuộc Huy.

## Notebook có hai phần

1. **Phần đã chạy được ở đây:** code đọc snapshot kết quả thực nghiệm chung, tính lại mean/std, vẽ learning curves,
   so sánh fixed/tuned và ba C, bảng nhãn hiếm, ví dụ lỗi. Output đã lưu trong notebook.
2. **Phần chạy RoBERTa của bạn:** các ô dùng trainer/metrics chung của nhóm. Mặc định tắt training để
   Run All không tải model hoặc train hàng giờ. Bật từng cờ sau khi có repo mới, môi trường và dữ liệu/model.

**Giới hạn:** các bảng benchmark là kết quả đã chạy trên máy nhóm, không phải lượt fine-tune mới trên máy Đức Trí.
Notebook này không chứa weights hoặc scores NPZ đầy đủ. Không tự nhận đóng góp cá nhân chỉ vì có kết quả chung.
''')
md('''
## 1. Môi trường để đọc bảng/biểu đồ

Mở bằng VS Code/Jupyter, chọn kernel Python có `numpy`, `pandas`, `matplotlib`.
Nếu thiếu, chạy `%pip install numpy pandas matplotlib` trong kernel đó. Không cần GPU để đọc phần báo cáo.
Snapshot nhúng bên dưới giúp notebook chạy độc lập, không cần tải CSV hoặc có thư mục evidence.
''')
code('''
from pathlib import Path
import json, sys, importlib.util, subprocess, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

pd.set_option('display.max_columns', 14)
pd.set_option('display.precision', 4)
plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})
print('Python:', sys.version.split()[0])
print('Chế độ mặc định: đọc/đối chiếu hồ sơ nhóm; chưa train lại RoBERTa.')
''')
code('SNAPSHOT = json.loads('+repr(json.dumps(snapshot,ensure_ascii=False))+')\n'
     "runs = pd.DataFrame(SNAPSHOT['all_runs'])\nmeans = pd.DataFrame(SNAPSHOT['mean_std'])\n"
     "rare = pd.DataFrame(SNAPSHOT['rare'])\nmetadata = SNAPSHOT['metadata']\n"
     "for frame in (runs, means, rare):\n    for col in frame.columns:\n        if col not in ['system','split','threshold_mode','method','architecture','label','before','after','status','validation_calibrated']:\n            frame[col] = pd.to_numeric(frame[col], errors='raise')\n"
     "print('Snapshot commit:', SNAPSHOT['commit'])\nprint('Seeds C2:', sorted(metadata))\nprint('Nguồn số liệu: bảng GitHub và run_metadata.json; KHÔNG phải dự đoán sinh ra trong notebook này.')")
md('''
## 2. Phần của Đức Trí và cấu hình thật

Ba seed là ba lần train cùng cấu hình/split với ngẫu nhiên khác nhau; không phải ba tập dữ liệu khác nhau.
`max_length` tính bằng token, không phải từ. Head có 28 logits vì bài toán có 27 cảm xúc và neutral.
`weighted=False` nghĩa là ba run chính ở đây **không dùng pos_weight**.
''')
code('''
role_table = pd.DataFrame([
    ['Duy', 'A và điều phối B; dữ liệu/metrics, tổng hợp'],
    ['Quốc Khánh', 'C1 BERT; trainer chung; tổng hợp báo cáo'],
    ['Đức Trí', 'C2 RoBERTa; 3 seed, ngưỡng, lỗi và phần viết C2'],
    ['Nhật Huy', 'C3 DistilBERT và demo từ C được chọn']
], columns=['Người', 'Phụ trách'])
display(role_table)
cfg = metadata['42']['config']
display(pd.DataFrame({'Thông số': list(cfg), 'Giá trị': [str(cfg[k]) for k in cfg]}))
display(pd.DataFrame([
    {'seed': int(seed), 'completed': m['completed'], 'selected_epoch': m['selected_epoch'],
     'train_samples': m['sizes']['train'], 'validation_samples': m['sizes']['validation'],
     'GPU máy đã chạy': m['environment']['gpu'], 'revision': m['checkpoint_revision']}
    for seed, m in metadata.items()
]))
''')
md('''
## 3. Multi-hot → logits → sigmoid → ngưỡng

Luồng thật: text → tokenizer RoBERTa → encoder → head 28 logits.
Khi train: `BCEWithLogitsLoss(logits, labels_float)`; khi dự đoán: sigmoid rồi so ngưỡng.
Không sigmoid trước BCEWithLogitsLoss; không dùng softmax/argmax ép đúng một nhãn.

**Ô sau là minh họa toán học**, không phải scores RoBERTa benchmark.
''')
code('''
label_names = metadata['42']['label_names']
chosen = ['joy', 'gratitude']
y_example = np.array([int(name in chosen) for name in label_names], dtype=np.float32)
print('Multi-hot shape:', y_example.shape, 'Số nhãn thật:', int(y_example.sum()))
display(pd.DataFrame({'label':label_names, 'y_true_example':y_example.astype(int)}).query('y_true_example == 1'))

# Chỉ 3 vị trí để giải thích số học; mô hình thật trả đủ 28 logits.
toy_logits = np.array([1.2, -0.4, -2.0])
toy_targets = np.array([1.0, 1.0, 0.0])
toy_scores = 1 / (1 + np.exp(-toy_logits))
# Công thức BCE ổn định trên logits, tương đương nguyên lý BCEWithLogitsLoss.
toy_bce = np.maximum(toy_logits, 0) - toy_logits * toy_targets + np.log1p(np.exp(-np.abs(toy_logits)))
display(pd.DataFrame({'label_example':['joy','gratitude','neutral'], 'logit_example':toy_logits,
    'sigmoid_example':toy_scores, 'true_example':toy_targets.astype(int),
    'pred_at_0.5': (toy_scores >= .5).astype(int), 'pred_at_0.35':(toy_scores >= .35).astype(int)}))
print('BCE mean minh họa:', toy_bce.mean())
print('Đổi ngưỡng chỉ đổi nhãn dự đoán; không thay trọng số model.')
''')
md('''
## 4. Learning curves C2 — đọc từ log thực nghiệm

Đây là train loss và Macro-F1 validation @0,5 sau mỗi epoch trong metadata ba run.
Không phải loss/metric được dựng giả để minh họa. Các run đều chọn epoch 3.
''')
code('''
history = pd.DataFrame([{'seed':int(seed), **epoch} for seed,m in metadata.items() for epoch in m['history']])
display(history[['seed','epoch','train_loss','macro_f1','micro_f1','epoch_seconds']])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for seed, frame in history.groupby('seed'):
    axes[0].plot(frame.epoch, frame.train_loss, marker='o', label=f'Seed {seed}')
    axes[1].plot(frame.epoch, frame.macro_f1, marker='o', label=f'Seed {seed}')
axes[0].set(title='RoBERTa: logged training loss', xlabel='Epoch', ylabel='BCE loss')
axes[1].set(title='RoBERTa: logged validation Macro-F1 @0.5', xlabel='Epoch', ylabel='Macro-F1')
for ax in axes: ax.set_xticks([1,2,3]); ax.legend(); ax.grid(alpha=.2)
plt.tight_layout(); plt.show()
''')
md('''
## 5. Từng seed và kiểm mean ± sample std

Macro-F1: tính F1 riêng mỗi nhãn rồi lấy trung bình đủ 28 nhãn.
Micro-F1: cộng TP/FP/FN trước khi tính. `ddof=1` là std mẫu qua ba seed.
Tuned-validation đã dùng chọn ngưỡng nên không là đánh giá độc lập.
''')
code('''
c2 = runs[runs.system == 'C_roberta'].copy()
display(c2[c2.threshold_mode.isin(['fixed','tuned'])][
    ['seed','split','threshold_mode','macro_f1','micro_f1','micro_precision','micro_recall','hamming_loss']
].sort_values(['split','threshold_mode','seed']))

metrics = ['macro_f1','micro_f1','micro_precision','micro_recall','hamming_loss']
summary = c2.groupby(['split','threshold_mode'])[metrics].agg(['mean','std'])
display(summary)
for (split, mode), group in c2.groupby(['split','threshold_mode']):
    official = means.query("system == 'C_roberta' and split == @split and threshold_mode == @mode").iloc[0]
    assert len(group) == 3
    for metric in metrics:
        assert np.isclose(group[metric].mean(), official[metric + '_mean'], atol=1e-12, rtol=0)
        assert np.isclose(group[metric].std(ddof=1), official[metric + '_std'], atol=1e-12, rtol=0)
print('PASS: mean/std tính từ từng seed khớp hồ sơ gốc ở cả 6 tổ hợp split/ngưỡng.')
''')
md('''
## 6. Fixed vs tuned: tăng F1 đi kèm đánh đổi

Fixed: ngưỡng 0,5. Tuned: mỗi nhãn có ngưỡng chọn từ validation của đúng model/seed.
Test dùng ngưỡng đã khóa, không dùng test tìm ngưỡng lại. Hamming Loss càng thấp càng tốt.
''')
code('''
test_c2 = means.query("system == 'C_roberta' and split == 'test' and threshold_mode in ['fixed','tuned']").set_index('threshold_mode').loc[['fixed','tuned']]
keys = ['macro_f1','micro_f1','micro_precision','micro_recall','hamming_loss']
comparison = pd.DataFrame({mode: test_c2.loc[mode,[key+'_mean' for key in keys]].to_numpy(dtype=float) for mode in ['fixed','tuned']}, index=keys)
comparison['delta'] = comparison.tuned - comparison.fixed
display(comparison)
fig, axes = plt.subplots(1,2,figsize=(11,3.8))
comparison.loc[['macro_f1','micro_f1','micro_precision','micro_recall'],['fixed','tuned']].plot.bar(ax=axes[0],rot=15,color=['#849bb7','#176c7a'])
axes[0].set(title='C2 locked test: F1 / Precision / Recall',ylim=(0,1),ylabel='Mean of 3 seeds')
comparison.loc[['hamming_loss'],['fixed','tuned']].plot.bar(ax=axes[1],rot=0,color=['#849bb7','#176c7a'])
axes[1].set(title='C2 Hamming Loss (lower is better)',ylim=(0,.05))
plt.tight_layout();plt.show()
display(Markdown('**Kết luận:** Macro-F1 tăng 0,4219 → 0,4872; Recall tăng, Precision giảm và Hamming Loss tăng. Không kết luận mọi chỉ số đều tốt hơn.'))
''')
md('''
## 7. So sánh ba C và tiêu chí chọn demo

Kiến trúc được chọn bằng mean Macro-F1 **validation @0,5**, đã chốt trước test.
Không đổi sang model khác chỉ vì thấy Micro-F1 test đẹp hơn. LR/epoch khác nhau giữa các C là giới hạn so sánh.
''')
code('''
base = means.query("split == 'validation' and threshold_mode == 'fixed'")[['system','macro_f1_mean','macro_f1_std']].sort_values('macro_f1_mean',ascending=False)
display(base)
assert base.iloc[0].system == 'C_bert'
print('Kiến trúc thắng tiêu chí nhóm:', base.iloc[0].system)
fig,ax = plt.subplots(figsize=(8,4))
architectures = ['C_bert','C_roberta','C_distilbert']; x=np.arange(3)
for k,mode in enumerate(['fixed','tuned']):
    frame = means.query("split == 'test' and threshold_mode == @mode").set_index('system').loc[architectures]
    bars=ax.bar(x + (k-.5)*.32, frame.macro_f1_mean, width=.32,
        yerr=frame.macro_f1_std,capsize=4,label=mode,color=['#849bb7','#176c7a'][k])
    ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=9)
ax.set(xticks=x,xticklabels=['BERT','RoBERTa','DistilBERT'],ylim=(0,.7),ylabel='Test Macro-F1',title='Locked test: mean ± sample std; 3 seeds each')
ax.legend();plt.tight_layout();plt.show()
''')
md('''
## 8. Nhãn hiếm C2 — giữ cả nhãn không cải thiện

Danh sách hiếm xác định từ train, không dùng test để chọn nhãn.
F1 grief sau tuning vẫn bằng 0. Support test nhỏ và std lớn cần được nêu trong báo cáo.
''')
code('''
rare_test=rare.query("split == 'test'").copy()
display(rare_test[['label','train_support','evaluation_support','before_f1_mean','before_f1_std','after_f1_mean','after_f1_std','delta_f1_mean']])
fig,ax=plt.subplots(figsize=(8,3.8));x=np.arange(len(rare_test))
for k,(prefix,label) in enumerate([('before','Fixed 0.5'),('after','Tuned on validation')]):
    ax.bar(x+(k-.5)*.32,rare_test[prefix+'_f1_mean'],width=.32,
        yerr=rare_test[prefix+'_f1_std'],capsize=3,label=label,color=['#849bb7','#176c7a'][k])
ax.set(xticks=x,xticklabels=rare_test.label.tolist(),ylabel='Test per-label F1',ylim=(0,.65),title='C2 rare labels: mean ± sample std (n=3)')
ax.legend();plt.tight_layout();plt.show()
''')
md('''
## 9. Ba ID có kết quả thật để đối chiếu lỗi

Các dòng sau chép từ `reports/error_case_studies.md` tại cùng commit. C2 dùng seed 2026 @0,5;
C1/C3 dùng seed đại diện chọn từ validation. Không chọn các seed bằng cách nhìn test.
Đây là ví dụ minh họa lỗi, không phải mọi dự đoán toàn test.
''')
code('''
cases = pd.DataFrame([
    {'id':'eczj48j','true':'admiration, excitement, neutral','C2_pred':'admiration','C2_FN':'excitement, neutral','C2_FP':'không',
     'C2_scores':'admiration=0.6858; excitement=0.0218; neutral=0.0426','C1_pred':'caring','C3_pred':'admiration'},
    {'id':'ed0jr9i','true':'embarrassment','C2_pred':'không nhãn','C2_FN':'embarrassment','C2_FP':'không',
     'C2_scores':'embarrassment=0.3418','C1_pred':'embarrassment','C3_pred':'không nhãn'},
    {'id':'eczcvgx','true':'neutral','C2_pred':'neutral','C2_FN':'không','C2_FP':'không',
     'C2_scores':'neutral=0.6429','C1_pred':'neutral','C3_pred':'approval'}
])
display(cases)
print('TP: chọn đúng nhãn thật. FN: bỏ sót nhãn thật. FP: chọn nhãn không có trong ground truth.')
print('Case 1: C2 nhận một phần nhãn. Case 2: bỏ sót nhãn hiếm. Case 3: C2 đúng, C3 có FN neutral / FP approval.')
''')
md('''
## 10. Các ô để tự chạy C2 trong project của bạn

**Bản local D:\\NLP\\Do_An_NhanDienCamXuc hiện mới có EDA.** Cần nhận code mới của nhóm và checkpoint/scores lớn
nếu muốn dùng lại model; Git chỉ mang các tệp được theo dõi, không tự tạo weights.

Notebook điều khiển trainer chung để cùng loss/metrics/protocol với Khánh và Huy. Các ô Python bên dưới là code thực thi,
không phải hình chụp lệnh. Khi bật, log chạy được in ngay trong output ô notebook.

Thứ tự: kiểm môi trường → smoke → full 3 seed → đọc kết quả của chính bạn.
Nếu máy có GPU 4 GB và batch 16 thiếu VRAM, có thể chọn batch 4 / accumulation 4 để effective batch 16;
ghi cấu hình mới và giữ thống nhất ba seed. Không gộp run mới khác cấu hình với số benchmark nhúng.

Không tạo/cài lại môi trường đang chạy thí nghiệm. Xem `docs/CHAY_THUC_NGHIEM.md` và `requirements-models-lock.txt`
trong bản bàn giao để chọn PyTorch phù hợp máy. Chưa có packages thì chạy cài đặt trong đúng kernel trước.
''')
code('''
# Đổi đường dẫn nếu bạn đặt repo mới ở chỗ khác.
PROJECT_ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p/'src/neural.py').is_file()), Path(r'D:\\NLP\\Do_An_NhanDienCamXuc'))
RUN_SMOKE = False       # True: pilot 64 train / 32 validation, 1 epoch; không dùng báo cáo benchmark.
RUN_FULL = False        # True: train lại RoBERTa đủ 3 seed; có thể tốn hàng giờ.
READ_LOCAL_RUNS = False # True: kiểm model/scores trên máy của bạn, không dùng snapshot GitHub.
RUN_LOCKED_TEST = False # True: đánh giá sau khi nhóm đã khóa final_protocol.json.
DEVICE = 'auto'
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 1

required_files=['scripts/train_transformer.py','src/neural.py','src/metrics.py','data/labels.json']
repo_checks = pd.DataFrame({'file':required_files,'exists':[(PROJECT_ROOT/name).is_file() for name in required_files]})
display(repo_checks)
print('Packages trong kernel:',{name:importlib.util.find_spec(name) is not None for name in ['torch','transformers','numpy','pandas','pyarrow','sklearn']})
print('Training flags:',RUN_SMOKE,RUN_FULL,'; local results flag:',READ_LOCAL_RUNS)
''')
md('''
### 10a. Hàm chạy trainer và giữ log trong output

Chỉ chạy trong repo cập nhật có đủ modules. Dùng chính `sys.executable` của kernel để không trộn Python khác.
`--resume` bỏ qua run đã hoàn tất hợp lệ, không resume optimizer ở giữa epoch. Không tự bật `--overwrite`.
''')
code('''
def run_group_script(module, arguments):
    missing=[name for name in required_files if not (PROJECT_ROOT/name).is_file()]
    if missing:
        raise RuntimeError('Project đang thiếu code mới: '+', '.join(missing))
    command=[sys.executable,'-u','-m',module,*map(str,arguments)]
    print('Chạy:',subprocess.list2cmdline(command),flush=True)
    env=os.environ.copy();env['PYTHONUTF8']='1';env['PYTHONUNBUFFERED']='1'
    process=subprocess.Popen(command,cwd=str(PROJECT_ROOT),env=env,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    try:
        for line in process.stdout:
            print(line,end='',flush=True)
        return_code=process.wait()
    except KeyboardInterrupt:
        process.terminate();process.wait()
        raise
    if return_code:
        raise RuntimeError(f'{module} kết thúc với mã lỗi {return_code}; kiểm output phía trên.')

if RUN_SMOKE:
    run_group_script('scripts.train_transformer',[
        '--architecture','roberta','--seed',42,'--smoke','--device',DEVICE,
        '--batch-size',BATCH_SIZE,'--gradient-accumulation',GRADIENT_ACCUMULATION,'--resume'])
else:
    print('Smoke chưa chạy: đổi RUN_SMOKE=True khi đã có đúng repo và môi trường.')
''')
md('''
### 10b. Chạy full đủ 3 seed C2

Đầu ra nằm trong `data/processed/transformers/roberta/seed_<seed>/full/standard/`:
checkpoint, validation_scores.npz, validation_metrics.json, run_metadata.json.
Trainer thật cập nhật toàn bộ encoder/head bằng BCE; chọn checkpoint bằng validation Macro-F1 @0,5.
''')
code('''
if RUN_FULL:
    for seed in [42,123,2026]:
        run_group_script('scripts.train_transformer',[
            '--architecture','roberta','--seed',seed,'--device',DEVICE,
            '--batch-size',BATCH_SIZE,'--gradient-accumulation',GRADIENT_ACCUMULATION,
            '--max-length',128,'--learning-rate',2e-5,'--epochs',3,
            '--revision','e2da8e2f811d1448a5b465c236feacd80ffbac7b','--resume'])
else:
    print('FULL chưa chạy trong notebook này. Các biểu đồ phía trên đọc hồ sơ full của nhóm.')
''')
md('''
### 10c. Đọc model artifacts mới và tự chấm validation bằng metrics chung

`load_transformer_run` kiểm completed/full, mapping, revision và SHA-256 của checkpoint/scores.
Ghép scores theo ID vào dữ liệu; không dựa vào việc CSV tình cờ có cùng thứ tự.
Kết quả ô này khi bật là từ artifacts trên máy bạn; không fallback sang snapshot nếu thiếu.
''')
code('''
if READ_LOCAL_RUNS:
    sys.path.insert(0,str(PROJECT_ROOT))
    from src.neural import load_transformer_run
    from src.data import load_goemotions, multi_hot
    from src.metrics import evaluate_multilabel
    from src.baseline import load_aligned_scores
    frames, names, _ = load_goemotions(PROJECT_ROOT,write_metadata=False,splits=('validation',))
    frame=frames['validation']; truth=multi_hot(frame.labels.tolist(),len(names))
    local_rows=[]
    for seed in [42,123,2026]:
        folder=PROJECT_ROOT/f'data/processed/transformers/roberta/seed_{seed}/full/standard'
        m=load_transformer_run(folder,require_full=True)
        scores=load_aligned_scores(folder/'validation_scores.npz',frame.id,names)
        result=evaluate_multilabel(truth,scores,names,threshold=.5)
        local_rows.append({'seed':seed,**{key:result[key] for key in metrics}})
    local_results=pd.DataFrame(local_rows).set_index('seed')
    display(local_results)
    display(local_results.agg(['mean','std']))
    print('Đây là metrics tự tính từ local scores; không sử dụng benchmark nhúng.')
else:
    print('Chưa đọc/chấm local scores. Đổi READ_LOCAL_RUNS=True sau khi có đủ artifacts full.')
''')
md('''
### 10d. Test sau khóa protocol

Chỉ bật khi từng run có `final_protocol.json` đã chốt từ validation theo quy trình nhóm.
Không dùng những câu test đã đọc để chọn lại model/seed/ngưỡng. Ô này kiểm protocol và
suy luận bằng checkpoint của đúng run; không tự tạo ngưỡng mới từ test.
''')
code('''
if RUN_LOCKED_TEST:
    folders=[PROJECT_ROOT/f'data/processed/transformers/roberta/seed_{seed}/full/standard' for seed in [42,123,2026]]
    if not all((folder/'final_protocol.json').is_file() for folder in folders):
        raise RuntimeError('Thiếu protocol đã khóa. Dừng test và phối hợp nhóm khóa lựa chọn bằng validation trước.')
    for folder in folders:
        run_group_script('scripts.evaluate_transformer_test',[
            '--run-dir',str(folder),'--protocol',str(folder/'final_protocol.json'),'--device',DEVICE])
else:
    print('Không đánh giá lại test trong phiên này; dùng kết quả locked đã có để đọc báo cáo.')
''')
md('''
## 11. Checklist báo cáo cá nhân

- [ ] Giải thích text → tokenizer → encoder → 28 logits; BCE, sigmoid và multi-hot.
- [ ] Giải thích seed, epoch được chọn và thông số C2 là cấu hình nhóm.
- [ ] Trình bày từng seed, mean/std, validation/test; không chỉ seed đẹp nhất.
- [ ] Trình bày tuning tăng Macro-F1 nhưng giảm Precision, tăng Hamming; grief vẫn chưa cải thiện.
- [ ] Phân tích 2–3 ID thật, đối chiếu cùng ID với Khánh/Huy.
- [ ] Xác nhận việc cá nhân đã làm và nơi lưu checkpoint/scores; không tự nhận đã chạy nếu mới đọc hồ sơ.
- [ ] Viết phần C2: phương pháp, cấu hình, kết quả, lỗi, hạn chế; đưa notebook này và artifacts thật cho nhóm.

**Bạn cần làm C2; không phải viết lại baseline A, BERT C1, DistilBERT C3 hoặc demo D.**
Notebook là nơi trình bày code/output; dùng trainer/metrics chung giúp kết quả ghép được với cả nhóm.
''')
code('''
display(Markdown('### Nguồn snapshot và code'))
for name,url in SNAPSHOT['sources'].items():
    display(Markdown(f'- [{name}]({url})'))
print('Phân công: nội dung Notion 10/10 người dùng cung cấp và docs/KE_HOACH_NHOM.md.')
print('Phạm vi đã chạy: đọc snapshot, bảng/biểu đồ/mean/std và ví dụ toán học. Training/suy luận C2 mới: CHƯA CHẠY.')
''')

# Giải thích và ô thực hành bổ sung của Đức Trí; trainer/metrics nhóm được giữ nguyên.
def markdown(text): return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())
def python_cell(text): return nbf.v4.new_code_cell(textwrap.dedent(text).strip())
sample_frame=rows('reports/tables/examples_30.csv')
samples=[{'id':r['id'],'text':r['text'],'label_ids':[int(v) for v in r['label_ids'].split(',')],
          'labels':r['labels']} for r in sample_frame[:5]]
extra_early=[
markdown("""
### 2a. Dữ liệu thật nhìn như thế nào?

`id` giúp truy vết/ghép dự đoán, `text` là đầu vào tiếng Anh, `labels` là danh sách ID nhãn.
Năm ví dụ dưới đây lấy từ bảng minh họa **train** của EDA, không phải sample benchmark mới.
Không học mô hình bằng chỉ năm câu này rồi gọi là full GoEmotions.
"""),
python_cell("TRAIN_EXAMPLES = "+repr(samples)+"\ndisplay(pd.DataFrame(TRAIN_EXAMPLES))\n"
    "print('3 split chuẩn: train=43410; validation=5426; test=5427. Giữ nguyên mapping 28 nhãn.')"),
markdown("""
### 3a. Tự viết multi-hot và kiểm với ví dụ

`y[i,j]=1` nếu câu i có nhãn j. Dạng ma trận N×28 cho phép một hàng có nhiều số 1.
Ô này là bài thực hành mã hóa; training dùng `src.data.multi_hot` của nhóm.
"""),
python_cell("""
def explain_multi_hot(label_lists, n_labels=28):
    matrix=np.zeros((len(label_lists),n_labels),dtype=np.float32)
    for i,ids in enumerate(label_lists):
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Nhãn rỗng hoặc lặp')
        if any(not isinstance(j,int) or j<0 or j>=n_labels for j in ids):
            raise ValueError('ID nhãn ngoài mapping')
        matrix[i,ids]=1
    return matrix

Y_demo=explain_multi_hot([[label_names.index('joy'),label_names.index('gratitude')],
                           [label_names.index('neutral')]])
display(pd.DataFrame(Y_demo.astype(int),columns=label_names))
assert Y_demo.shape == (2,28) and Y_demo.sum(axis=1).tolist()==[2,1]
print('Đúng shape N×28; BCE cần nhãn float, không phải một integer class cho cả câu.')
"""),
markdown("""
### 3b. BCE và backpropagation chạy thật bằng PyTorch

Đây là tensor logits minh họa (không phải logits RoBERTa). Chạy `loss.backward()` để xem gradient.
Với nhãn thật=1, gradient có xu hướng kéo logit lên; với nhãn thật=0, kéo logit xuống.
Trainer C2 dùng đúng loại loss này trên 28 đầu ra của **model thật**.
"""),
python_cell("""
if importlib.util.find_spec('torch') is not None:
    import torch
    demo_logits=torch.tensor([[1.2,-.4,-2.]],requires_grad=True)
    demo_labels=torch.tensor([[1.,1.,0.]])
    loss_fn=torch.nn.BCEWithLogitsLoss()
    demo_loss=loss_fn(demo_logits,demo_labels)
    demo_loss.backward()
    display(pd.DataFrame({'logit':demo_logits.detach().numpy()[0],
        'target':demo_labels.numpy()[0],
        'sigmoid':torch.sigmoid(demo_logits).detach().numpy()[0],
        'd_loss_d_logit':demo_logits.grad.numpy()[0]}))
    print('Loss minh họa PyTorch:',float(demo_loss.detach()))
    assert np.isclose(float(demo_loss.detach()),toy_bce.mean(),atol=1e-6)
else:
    print('Chưa có torch trong kernel. Phần bảng/biểu đồ khác vẫn chạy; cài đúng Torch để xem backward.')
""")]
# Thêm bài tập sau minh họa sigmoid, trước learning curves.
position=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and c.source.startswith('## 4.'))
cells[position:position]=extra_early

training_setup=[
markdown("""
### 10.1. Hiểu tokenizer và tensor đầu vào bằng checkpoint thật

Tokenizer RoBERTa chia subword theo checkpoint; token count khác word count.
`input_ids`: mã token; `attention_mask`: 1 cho token thật, 0 cho padding.
`RUN_TOKENIZER` chỉ tải tokenizer khi bạn bật; mặc định không có token output giả.
"""),
python_cell("""
RUN_TOKENIZER=False
TOKENIZER_REVISION='e2da8e2f811d1448a5b465c236feacd80ffbac7b'
if RUN_TOKENIZER:
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('FacebookAI/roberta-base',revision=TOKENIZER_REVISION)
    texts=[sample['text'] for sample in TRAIN_EXAMPLES[:2]]
    encoded=tokenizer(texts,truncation=True,padding=True,max_length=128,return_tensors='pt')
    print('input_ids shape:',tuple(encoded['input_ids'].shape))
    print('attention_mask:',encoded['attention_mask'])
    for text,ids,mask in zip(texts,encoded['input_ids'],encoded['attention_mask']):
        tokens=tokenizer.convert_ids_to_tokens(ids.tolist())
        display(pd.DataFrame({'token':tokens,'input_id':ids.tolist(),'attention':mask.tolist()}))
else:
    print('Tokenizer model thật chưa chạy: đặt RUN_TOKENIZER=True sau khi cài transformers và có mạng/cache.')
"""),
markdown("""
### 10.2. Xem một forward pass và loss của RoBERTa thật

Warning head mới khởi tạo là bình thường: checkpoint ngôn ngữ chưa có head 28 nhãn của đồ án.
Ô này chỉ kiểm tensor/forward, **không phải mô hình fine-tune hoàn tất** và không tạo số F1.
Chỉ bật sau tokenizer. Giải phóng model sau demo để không chiếm VRAM khi train.
"""),
python_cell("""
RUN_FORWARD=False
if RUN_FORWARD:
    if not RUN_TOKENIZER:
        raise RuntimeError('Bật/chạy ô tokenizer trước.')
    import gc, torch
    from transformers import AutoModelForSequenceClassification
    target=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fresh_model=AutoModelForSequenceClassification.from_pretrained(
        'FacebookAI/roberta-base',revision=TOKENIZER_REVISION,num_labels=28,
        problem_type='multi_label_classification',
        id2label=dict(enumerate(label_names)),label2id={name:i for i,name in enumerate(label_names)})
    fresh_model.to(target).eval()
    batch={k:v.to(target) for k,v in encoded.items()}
    true_tensor=torch.tensor(explain_multi_hot([r['label_ids'] for r in TRAIN_EXAMPLES[:2]]),device=target)
    with torch.no_grad():
        forward_logits=fresh_model(**batch).logits
        forward_loss=torch.nn.BCEWithLogitsLoss()(forward_logits,true_tensor)
    print('Model logits:',tuple(forward_logits.shape),'labels:',tuple(true_tensor.shape),'loss:',float(forward_loss))
    assert forward_logits.shape==true_tensor.shape==(2,28)
    del fresh_model,batch,true_tensor,forward_logits,forward_loss
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
else:
    print('Forward model chưa chạy; chỉ bật để kiểm luồng, không dùng scores head chưa học làm benchmark.')
"""),
markdown("""
### 10.3. Bên trong vòng lặp fine-tune của nhóm

Notebook dùng trainer chung để giữ artifact/metrics/protocol đồng nhất. Các dòng cốt lõi:

```python
model.train()                              # Bật chế độ huấn luyện
logits = model(**inputs).logits            # N×28, chưa sigmoid
loss = multilabel_loss(logits, targets)    # BCE, targets float
loss.backward()                           # Tính gradient
optimizer.step()                          # Cập nhật encoder và head
optimizer.zero_grad()                     # Xóa gradient cho bước tiếp
```

Trainer thật còn có AMP/GradScaler, clipping, scheduler và gradient accumulation;
đừng thay nguyên vòng lặp thật bằng snippet trên. Khi validation dùng `eval()` và `no_grad()`.
`src/neural.py` đặt seed, cấu hình; `scripts/train_transformer.py` thực hiện train và lưu checkpoint.
Các hàm thật sau được in để bạn đọc tại chỗ, không tải hoặc train model.
"""),
python_cell("""
if all((PROJECT_ROOT/name).is_file() for name in required_files):
    sys.path.insert(0,str(PROJECT_ROOT))
    import inspect
    from IPython.display import Code
    from src.neural import multilabel_loss, training_config
    from scripts.train_transformer import encoded_dataset,trim_padding_collate,predict_scores
    display(Code(inspect.getsource(multilabel_loss),language='python'))
    display(Code(inspect.getsource(encoded_dataset),language='python'))
    display(Code(inspect.getsource(predict_scores),language='python'))
else:
    print('Nhận repo mới để đọc các hàm trainer thật trong output ô này.')
""")]
position=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and c.source.startswith('### 10a.'))
cells[position:position]=training_setup

freeze_cells=[
markdown("""
### 10c-bổ sung. Chọn ngưỡng trên validation và khóa protocol

28 nhãn có phân bố/score khác nhau; chọn riêng trên validation của đúng checkpoint/seed.
Trainer của nhóm dùng lưới 0,05..0,95, bước 0,05; khi F1 hòa ưu tiên ngưỡng gần 0,5,
nếu vẫn hòa chọn ngưỡng lớn hơn. Tuning không train lại encoder/head.

`final_protocol.json` lưu ngưỡng, mapping, revision và hash metadata/scores. **Không ghi đè** nếu đã có protocol.
Trước lần chạy test đầu tiên, cả nhóm cần khóa lựa chọn; test đã mở thì giữ protocol.
"""),
python_cell("""
FREEZE_VALIDATION=False
if FREEZE_VALIDATION:
    for seed in [42,123,2026]:
        folder=PROJECT_ROOT/f'data/processed/transformers/roberta/seed_{seed}/full/standard'
        if (folder/'final_protocol.json').exists():
            print(f'Seed {seed}: protocol đã có; giữ nguyên, không chọn lại ngưỡng.')
        else:
            run_group_script('scripts.freeze_experiment',['--run-dir',folder,'--threshold-mode','tuned'])
else:
    print('Chưa khóa protocol mới. Snapshot phía trên đã được nhóm khóa/đánh giá trước đó.')
""")]
position=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and c.source.startswith('### 10d.'))
cells[position:position]=freeze_cells

analysis_cells=[
markdown("""
### 10e. Phân tích lỗi validation từ scores thật trên máy bạn

Chỉ bật sau khi có scores validation và data cache. Ghép bằng ID, dùng threshold 0,5 ở ô này.
In nhãn thật, dự đoán, FN/FP và nguyên văn; heatmap đếm FN nhãn hàng + FP nhãn cột cùng câu.
Đây là lỗi dự đoán, khác ma trận đồng xuất hiện nhãn thật. Không cộng số cặp thành số câu sai.
"""),
python_cell("""
ANALYZE_LOCAL_ERRORS=False
if ANALYZE_LOCAL_ERRORS:
    sys.path.insert(0,str(PROJECT_ROOT))
    from src.data import load_goemotions,multi_hot
    from src.baseline import load_aligned_scores,label_error_pairs
    from src.neural import load_transformer_run
    frames,names,_=load_goemotions(PROJECT_ROOT,write_metadata=False,splits=('validation',))
    frame=frames['validation']; seed=2026
    folder=PROJECT_ROOT/f'data/processed/transformers/roberta/seed_{seed}/full/standard'
    load_transformer_run(folder,require_full=True)
    scores=load_aligned_scores(folder/'validation_scores.npz',frame.id,names)
    truth=multi_hot(frame.labels.tolist(),28).astype(bool);pred=scores>=.5
    missed=truth & ~pred;extra=~truth & pred
    records=[]
    for row_index in np.flatnonzero(np.any(missed|extra,axis=1))[:10]:
        r=frame.iloc[int(row_index)]
        decode=lambda row:', '.join(np.array(names)[np.asarray(row,dtype=bool)]) or 'không nhãn'
        records.append({'id':r.id,'text':r.text,'true':decode(truth[row_index]),
            'pred':decode(pred[row_index]),'FN':decode(missed[row_index]),'FP':decode(extra[row_index])})
    display(pd.DataFrame(records))
    display(pd.DataFrame(label_error_pairs(truth,scores,names,.5)).head(10))
    matrix=missed.astype(np.int64).T @ extra.astype(np.int64)
    fig,ax=plt.subplots(figsize=(10,8));im=ax.imshow(matrix,cmap='YlOrRd')
    ax.set_xticks(range(28),names,rotation=90);ax.set_yticks(range(28),names)
    ax.set(xlabel='Extra predicted label (FP)',ylabel='Missed true label (FN)',title='C2 validation FN/FP pairs')
    fig.colorbar(im,ax=ax);plt.tight_layout();plt.show()
else:
    print('Chưa phân tích local scores; ba case đã lưu phía trên có nguồn hồ sơ nhóm.')
"""),
markdown("""
### 10f. Dự đoán câu mới bằng checkpoint đã fine-tune

Chỉ dùng checkpoint `full/standard` đã kiểm hash; không dùng head mới khởi tạo ở ô forward.
Ô này dùng ngưỡng cố định 0,5. Muốn dùng tuned phải đọc protocol đúng seed, không lấy ngưỡng mô hình khác.
Văn bản tiếng Anh; scores chưa được hiệu chuẩn xác suất. Nếu không có nhãn vượt ngưỡng, không tự gán neutral.
"""),
python_cell("""
RUN_INFERENCE=False
if RUN_INFERENCE:
    import torch,gc
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    sys.path.insert(0,str(PROJECT_ROOT))
    from src.neural import load_transformer_run
    folder=PROJECT_ROOT/'data/processed/transformers/roberta/seed_2026/full/standard'
    m=load_transformer_run(folder,require_full=True)
    tokenizer=AutoTokenizer.from_pretrained(folder/'checkpoint',local_files_only=True)
    trained_model=AutoModelForSequenceClassification.from_pretrained(folder/'checkpoint',local_files_only=True).eval()
    text='Thank you so much! I am happy you could help.'
    inputs=tokenizer(text,truncation=True,max_length=m['config']['max_length'],return_tensors='pt')
    with torch.no_grad():
        probabilities=torch.sigmoid(trained_model(**inputs).logits.float())[0].numpy()
    output=pd.DataFrame({'label':m['label_names'],'score':probabilities,'threshold':.5})
    output['selected']=output.score>=output.threshold
    display(output.sort_values('score',ascending=False))
    print('Nhãn chọn:',output.loc[output.selected,'label'].tolist())
    del trained_model;gc.collect()
else:
    print('Suy luận câu mới chưa chạy; notebook không gán sẵn joy/gratitude cho câu ví dụ.')
""")]
position=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and c.source.startswith('## 11.'))
cells[position:position]=analysis_cells

cells[0].source += '\n\n**Tác giả phần notebook/giải thích: Đức Trí — [Ductri2006](https://github.com/Ductri2006).**\nTrainer/metrics và kết quả thực nghiệm chung được dùng lại từ repo nhóm, giữ nguồn; không đổi tác giả lịch sử các run.'

for c in cells:
    if c.cell_type=='markdown':
        c.source=c.source.replace('**Bản local D:\\NLP\\Do_An_NhanDienCamXuc hiện mới có EDA.**',
            '**Bản local cũ kiểm tra ngày 11/10 mới có EDA; nhánh notebook này kế thừa code C chung của main.**')

nb.cells=cells
nb.metadata={'authors':[{'name':'Đức Trí','github':'Ductri2006'}], 'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
 'language_info':{'name':'python','version':sys.version.split()[0]},
 'c2_report':{'source_commit':snapshot['commit'],'fresh_training_executed':False,'benchmark_provenance':'GitHub group artifacts'}}
out=ROOT/'notebooks/roberta_duc_tri.ipynb'
out.parent.mkdir(exist_ok=True)
nbf.write(nb,out)
parser=argparse.ArgumentParser(description='Tạo notebook RoBERTa riêng Đức Trí từ hồ sơ nhóm; --execute chỉ chạy phần đọc/giải thích, không train.')
parser.add_argument('--execute',action='store_true')
args=parser.parse_args()
if args.execute:
    client=NotebookClient(nb,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},allow_errors=False)
    client.execute()
nbf.write(nb,out)
codes=[c for c in nb.cells if c.cell_type=='code']
errors=[o for c in codes for o in c.outputs if o.output_type=='error']
figures=sum('image/png' in o.get('data',{}) for c in codes for o in c.outputs)
assert not errors
if args.execute:
    assert all(c.execution_count is not None for c in codes) and figures>=4
print(json.dumps({'file':str(out),'code_cells':len(codes),'executed':sum(c.execution_count is not None for c in codes),'saved_errors':len(errors),'figures':figures,'bytes':out.stat().st_size,'fresh_training':False},ensure_ascii=False))
