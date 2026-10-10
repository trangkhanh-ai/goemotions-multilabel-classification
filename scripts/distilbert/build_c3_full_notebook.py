"""Notebook C3 full, chỉ đọc dữ liệu/outputs thật; không tự train lại khi Run All."""
from pathlib import Path
import nbformat as nbf

from src.paths import ROOT
cells=[]
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md('''# Nhật Huy — C3 DistilBERT: full ba seed và demo

Notebook hướng dẫn theo từng bước từ dữ liệu đến kết quả, ngưỡng và phân tích lỗi.
**Mọi bảng thực nghiệm được đọc/tính lại từ các run thật trên official train/validation.**
Không có dữ liệu tổng hợp hoặc số liệu tự điền. Không đánh giá test, không chọn người thắng C1/C2/C3.

Pilot 1.024/256 mẫu trước đây chỉ là kiểm kỹ thuật. Notebook này dùng **43.410 train / 5.426 validation**,
ba seed **42, 123, 2026** và cấu hình đã ghi trước khi huấn luyện trong `full/protocol.json`.
Mặc định Run All kiểm lại các kết quả đã có; không tự chạy thêm ba lượt train dài.
''')
md('## 1. Mở đúng môi trường và thư mục repo')
code('''from pathlib import Path
import sys
ROOT=Path.cwd().resolve()
if not (ROOT/'src').exists(): ROOT=ROOT.parent
assert (ROOT/'src/models/distilbert_study.py').exists()
sys.path.insert(0,str(ROOT))
from src.models.distilbert_study import read_json, prepare_frames, EmotionDataset, make_loader, load_bundle
from src.datasets.goemotions import multi_hot, sha256
from src.evaluation.metrics import evaluate_multilabel
from src.models.baseline import load_aligned_scores
from src.evaluation.distilbert_thresholds import predict_c3, load_c3_thresholds
import numpy as np
import pandas as pd
import torch
from IPython.display import display
torch.set_num_threads(4)
REPORT=ROOT/'reports/c3_distilbert/full'
protocol=read_json(REPORT/'protocol.json')
print('Python:',sys.version.split()[0], '| torch:',torch.__version__)
print('CUDA:',torch.cuda.is_available())
display(pd.DataFrame(read_json(ROOT/'reports/c3_distilbert/environment.json')['packages'].items(),columns=['Thư viện','Phiên bản']))''')
md('''## 2. Đọc cấu hình đã khóa trước khi chạy

Giữ nguyên cấu hình giữa các seed: DistilBERT-base-uncased, max length 128, lr 2e-5,
3 epoch, batch 16. Model và dataset có revision ghim. Khác pilot: batch vật lý 16,
không cần tích lũy; kiểm bộ nhớ trước full được ghi ở `full_preflight.json`.
Checkpoint chọn bằng Macro-F1 validation @0,5; khi hòa giữ epoch sớm hơn.
''')
code('''config=protocol['config']
assert sha256(ROOT/'configs/distilbert_full.json')==protocol['config_sha256']
display(pd.DataFrame(config.items(),columns=['Tham số','Giá trị']).astype(str))
print('Seeds:',protocol['seeds'])
print('Chọn checkpoint:',protocol['checkpoint_selection'])
print('Không dùng test:',not protocol['test_used'])''')
md('''## 3. Đọc dữ liệu thật và kiểm ID/mapping

`prepare_frames` chỉ đọc train/validation và kiểm hash nguồn. Giữ nguyên văn bản và official split.
Không oversampling, không thêm câu nhân tạo. Phần EDA toàn bộ dữ liệu nằm trong notebook EDA riêng.
''')
code('''frames,names,manifest=prepare_frames(config)
display(pd.DataFrame([{'split':s,'số mẫu':len(df),'ID duy nhất':df.id.nunique()} for s,df in frames.items()]))
assert len(frames['train'])==43410 and len(frames['validation'])==5426
assert not (set(frames['train'].id)&set(frames['validation'].id))
example=frames['train'].head(5).copy()
example['label_names']=example.labels.apply(lambda labels:[names[i] for i in labels])
display(example[['id','text','labels','label_names']])''')
md('''## 4. Chuyển nhãn sang multi-hot float32

Mỗi mẫu có 28 vị trí; một hoặc nhiều vị trí bằng 1. Target được đưa trực tiếp vào
BCEWithLogitsLoss, không biến thành một class index.
''')
code('''targets={s:multi_hot(df.labels.tolist(),len(names)).astype(np.float32) for s,df in frames.items()}
print({s:(y.shape,str(y.dtype)) for s,y in targets.items()})
display(pd.DataFrame({'label_id':range(28),'label':names,'train_support':targets['train'].sum(0).astype(int),
                      'validation_support':targets['validation'].sum(0).astype(int)}))
print('Multi-hot mẫu đầu:',targets['train'][0].tolist())''')
md('''## 5. Tokenize và xem batch thật

Đọc tokenizer của checkpoint C3 đại diện (seed có Macro-F1 validation @0,5 cao nhất trong C3).
Đây chưa phải kiến trúc thắng của nhóm. Padding theo batch và truncation ở 128 token.
''')
code('''selected=read_json(REPORT/'representative_c3.json')
RUN=ROOT/selected['run']
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(RUN/'best',local_files_only=True)
dataset=EmotionDataset(frames['train'].head(16),tokenizer,config['max_length'])
batch=next(iter(make_loader(dataset,tokenizer,config['batch_size'])))
display(pd.DataFrame([{'tensor':k,'shape':str(tuple(v.shape)),'dtype':str(v.dtype)} for k,v in batch.items()]))
print(tokenizer.convert_ids_to_tokens(batch['input_ids'][0].tolist()))
assert batch['labels'].dtype==torch.float32 and batch['labels'].shape[1]==28''')
md('''## 6. Kiểm loss bằng checkpoint và nhãn thật

Luồng: tokenizer → encoder → đầu phân loại 28 logits → BCEWithLogitsLoss khi học.
Sigmoid chỉ chuyển logits thành scores khi suy luận. Cell sau chỉ forward để kiểm loss,
không cập nhật trọng số, không tạo dữ liệu giả.
''')
code('''bundle=load_bundle(RUN,'cpu')
with torch.inference_mode():
    result=bundle['model'](**batch)
    expected=torch.nn.functional.binary_cross_entropy_with_logits(result.logits,batch['labels'])
torch.testing.assert_close(result.loss,expected)
print('Shape logits:',tuple(result.logits.shape))
print('Loss trên batch train thật để kiểm công thức:',float(result.loss))
print('PASS: loss của model đúng BCEWithLogitsLoss')''')
md('''## 7. Cách chạy ba seed

Runner đọc cấu hình, kiểm dữ liệu → seed → tokenizer/model → AdamW → train/validation từng epoch
→ giữ best epoch → scores/metrics → nạp lại checkpoint để kiểm tính nhất quán.
AMP và gradient checkpointing được bật; CPU threads=4. AdamW không decay bias/LayerNorm,
warmup 10%, linear decay, clip gradient 1,0. Xem toàn bộ vòng lặp tại `scripts/distilbert/train_distilbert.py`.

Lệnh đã dùng: `python -m scripts.distilbert.run_c3_full`. Nó chạy tuần tự để các seed không tranh GPU;
run hoàn thành và đúng hash được kiểm rồi bỏ qua, không ghi đè. Chạy lại một thí nghiệm riêng
cần `--output` mới với `scripts.distilbert.train_distilbert`; không ghép lẫn với ba seed đã khóa.
''')
code('''import subprocess
print('Lệnh từ interpreter hiện tại:')
print(subprocess.list2cmdline([sys.executable,'-u','-m','scripts.distilbert.run_c3_full']))
print('Cell này không khởi động train.')
for seed in protocol['seeds']:
    log=REPORT/f'train_seed_{seed}.log'
    print(seed,':', '\\n'.join(log.read_text(encoding='utf-8').splitlines()[-3:]))''')
md('## 8. Xem lịch sử học và tài nguyên của cả ba seed')
code('''history=pd.read_csv(REPORT/'history_all_seeds.csv')
display(history)
display(pd.read_csv(REPORT/'resources_all_seeds.csv'))
from IPython.display import Image
display(Image(filename=str(REPORT/'learning_curves.png')))
for seed in protocol['seeds']:
    meta=read_json(ROOT/f'data/processed/c3_distilbert/full/seed_{seed}/run.json')
    expected_epoch=int(history[history.seed==seed].sort_values(['validation_macro_f1','epoch'],ascending=[False,True]).iloc[0].epoch)
    assert meta['best_epoch']==expected_epoch
    assert meta['n_train']==43410 and meta['n_validation']==5426 and not meta['test_used']
print('PASS: best epoch đúng quy tắc, đủ dữ liệu, không dùng test.')''')
md('''## 9. Ghép scores theo ID và tính lại metrics

Không giả định file scores có cùng thứ tự dòng với dữ liệu. Ghép bằng ID, kiểm label_names,
shape N×28 và scores [0,1]. Tính lại đủ metrics bằng hàm chung, so với JSON đã lưu.
''')
code('''all_scores={}
for seed in protocol['seeds']:
    run=ROOT/f'data/processed/c3_distilbert/full/seed_{seed}'
    scores=load_aligned_scores(run/'validation_scores.npz',frames['validation'].id,names)
    recalculated=evaluate_multilabel(targets['validation'],scores,names,.5)
    assert recalculated==read_json(run/'validation_metrics.json')
    all_scores[seed]=scores
print('PASS: ID, mapping và metrics của cả ba seed khớp.')''')
md('''## 10. Kết quả cơ sở: từng seed và mean ± sample std

Tính Macro-F1 trên đủ 28 nhãn, zero_division=0. Mean/std dùng cả ba seed, `ddof=1`.
Không lấy seed cao nhất để thay cho kết quả trung bình kiến trúc.
''')
code('''from scripts.distilbert.summarize_distilbert import summarize
runs=[ROOT/f'data/processed/c3_distilbert/full/seed_{seed}' for seed in protocol['seeds']]
rows,stats=summarize(runs)
display(rows)
display(stats)
assert len(rows)==3
print('Seed C3 đại diện:',selected['seed'],'; kiến trúc thắng toàn nhóm:',selected['architecture_winner'])''')
md('## 11. Metrics từng nhãn và nhãn hiếm đã định nghĩa từ train')
code('''per_label=pd.read_csv(REPORT/'per_label_all_seeds.csv')
display(per_label[(per_label.seed==selected['seed'])&(per_label['mode']=='fixed_0.5')])
print('Nhãn hiếm đã định nghĩa trước:',protocol['rare_labels_train_defined'])''')
md('''## 12. Nâng cao đúng phạm vi: ngưỡng riêng cho từng nhãn

Mỗi seed dùng validation scores của chính checkpoint đó. Lưới 0,05 đến 0,95, bước 0,05;
hòa F1 thì gần 0,5 nhất, tiếp tục hòa chọn ngưỡng lớn hơn. Không lấy ngưỡng baseline A.
Giữ nguyên bảng cơ sở @0,5; không train thêm weighting hoặc mô hình khác.

**Tuned-validation được chọn và đo trên cùng validation nên có thể lạc quan.**
Không diễn giải là cải thiện test hoặc đánh giá độc lập.
''')
code('''threshold_table=[]
for seed,run in zip(protocol['seeds'],runs):
    meta=read_json(run/'run.json')
    values=load_c3_thresholds(run,meta,'per_label')
    for name,value in zip(names,values): threshold_table.append({'seed':seed,'label':name,'threshold':value})
display(pd.DataFrame(threshold_table).pivot(index='label',columns='seed',values='threshold').reindex(names))
display(pd.read_csv(REPORT/'threshold_comparison_runs.csv'))
display(pd.read_csv(REPORT/'threshold_comparison_mean_std.csv'))''')
md('''## 13. Nhãn hiếm trước/sau tuning

Giữ cả nhãn không tăng hoặc giảm; phân biệt support validation với train support.
Không dùng số liệu pilot để kết luận hiệu quả nhãn hiếm.
''')
code('''display(pd.read_csv(REPORT/'rare_labels_mean_std.csv'))
display(Image(filename=str(REPORT/'rare_labels_f1.png')))
display(pd.read_csv(REPORT/'rare_labels_before_after.csv')[['seed','mode','label','train_support','support','precision','recall','f1']])''')
md('''## 14. Phân tích lỗi C3 trên mẫu thật

Ba nhóm kiểm tra: FN và FP trong cùng câu; bỏ sót một phần nhãn thật của câu đa nhãn;
lỗi nhãn hiếm/neutral. Nhóm có thể giao nhau, không cộng số nhóm để ra tổng câu sai.
FN/FP cùng câu chưa đủ chứng minh hai cảm xúc gần nghĩa; cần đọc nội dung thủ công.
Ví dụ được lấy theo ID tăng dần trong mỗi nhóm, không chỉ chọn lỗi phù hợp nhận định có sẵn.
Đối chiếu ba C cần thêm scores C1/C2 của các bạn; ở đây chỉ phân tích C3.
Đọc bảy mẫu có nhận xét thủ công tại
[EXPERIMENTS.md](../reports/c3_distilbert/full/EXPERIMENTS.md): ranh giới cảm xúc gần nghĩa,
bỏ sót một phần nhãn, diễn đạt hàm ý/neutral. Các nhận xét không chứng minh nguyên nhân lỗi.
''')
code('''display(pd.read_csv(REPORT/'error_group_counts.csv'))
examples=read_json(REPORT/'error_examples.json')
display(pd.DataFrame([{k:v for k,v in e.items() if k!='scores'} for e in examples]).groupby('group',sort=False).head(3))
display(pd.read_csv(REPORT/f"error_pairs_seed_{selected['seed']}.csv").head(12))
for e in examples:
    row=frames['validation'].set_index('id').loc[e['id']]
    assert row.text==e['text'] and [names[i] for i in row.labels]==e['true_labels']
print('PASS: ví dụ/ID/nhãn thật đều khớp dataset.')''')
md('''## 15. Dự đoán và demo bằng ví dụ có thật

Dùng mẫu đầu validation, không chỉnh câu để làm đẹp đầu ra. Cơ sở và tuned giữ nguyên scores,
chỉ thay ngưỡng. Chỉ nạp ngưỡng khi hash model/scores và seed khớp.
''')
code('''text=frames['validation'].text.iloc[0]
print('ID:',frames['validation'].id.iloc[0], '\\nText:',text)
fixed=predict_c3(bundle,[text],RUN,'fixed')[0]
tuned=predict_c3(bundle,[text],RUN,'per_label')[0]
print('Nhãn thật:',[names[i] for i in frames['validation'].labels.iloc[0]])
print('Cơ sở:',fixed['labels'],'| Tuned:',tuned['labels'])
assert fixed['scores']==tuned['scores']
display(pd.DataFrame(fixed['scores'].items(),columns=['Nhãn','Score']).sort_values('Score',ascending=False))
print('Kiểm input rỗng:')
try: predict_c3(bundle,['  '],RUN)
except ValueError as exc: print(exc)
empty_indices=np.flatnonzero((all_scores[selected['seed']]>=.5).sum(1)==0)
print('Số mẫu validation không nhãn nào đạt 0,5:',len(empty_indices))
if len(empty_indices):
    i=int(empty_indices[0])
    actual=predict_c3(bundle,[frames['validation'].text.iloc[i]],RUN)[0]
    print('Ví dụ thật:',frames['validation'].id.iloc[i],actual['text'],actual['labels'])''')
md(r'''## 16. Kiểm tra và bàn giao

```powershell
.\.venv-huy\Scripts\python.exe -m streamlit run app_distilbert_huy.py
```
App mặc định dùng seed C3 đại diện, hiển thị đúng trạng thái chưa chọn kiến trúc thắng.
Chọn cơ sở 0,5 hoặc ngưỡng từng nhãn của đúng run. Các ví dụ lỗi, số liệu và minh chứng kiểm demo
nằm trong `reports/c3_distilbert/full/`, xem
`reports/c3_distilbert/full/DEMO_EVIDENCE.md` (tạo cục bộ sau khi thực nghiệm).
Xem `docs/DISTILBERT_STUDY.md` để chạy lại và bàn giao checkpoint.

Huy đã có thể bàn giao C3 train/validation, phần nâng cao ngưỡng, phân tích lỗi riêng,
phương pháp/kết quả và demo C3. Vẫn cần cả nhóm: kết quả C1/C2, đối chiếu cùng ID, chọn
C thắng rồi tích hợp demo cuối, và khóa protocol test trước đánh giá cuối.

Nguồn: [DistilBERT paper](https://arxiv.org/abs/1910.01108),
[model card](https://huggingface.co/distilbert/distilbert-base-uncased).
''')
code('''verification=read_json(REPORT/'verification.json')
assert verification['status']=='passed'
display(pd.DataFrame(verification['checks'].items(),columns=['Kiểm tra','Kết quả']))
print('App/CLI tuned:',verification['tuned_app_cli_parity'])
print('App tập nhãn rỗng:',verification['empty_prediction_ui'])
print('Notebook hoàn tất: không có cell train lại, không dùng test.')''')
notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{
    'display_name':'Python (C3 .venv-huy)','language':'python','name':'python3'},'language_info':{'name':'python'}})
nbf.validate(notebook)
nbf.write(notebook,ROOT/'notebooks/05_distilbert_huy.ipynb')
print(f'Created full C3 notebook: {len(cells)} cells')
