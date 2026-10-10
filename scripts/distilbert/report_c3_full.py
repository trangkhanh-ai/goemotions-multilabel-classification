"""Viết báo cáo/biểu đồ từ bảng đo thật; dừng nếu chưa đủ outputs ba seed."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models.distilbert_study import ROOT, read_json


def main():
    folder=ROOT/'reports/c3_distilbert/full'
    protocol=read_json(folder/'protocol.json')
    selected=read_json(folder/'representative_c3.json')
    rows=pd.read_csv(folder/'validation_runs.csv')
    stats=pd.read_csv(folder/'validation_mean_std.csv')
    compared=pd.read_csv(folder/'threshold_comparison_mean_std.csv')
    rare=pd.read_csv(folder/'rare_labels_mean_std.csv')
    history=pd.read_csv(folder/'history_all_seeds.csv')
    resources=pd.read_csv(folder/'resources_all_seeds.csv')
    groups=pd.read_csv(folder/'error_group_counts.csv')
    assert set(rows.seed)==set(protocol['seeds']) and len(rows)==3
    table=stats.copy()
    table['mean ± std']=table.apply(lambda r:f'{r["mean"]:.6f} ± {r.sample_std:.6f}',axis=1)
    comparison=[]
    for metric in stats.metric:
        a=compared[(compared.metric==metric)&(compared['mode']=='fixed_0.5')].iloc[0]
        b=compared[(compared.metric==metric)&(compared['mode']=='per_label_tuned_on_validation')].iloc[0]
        comparison.append({'metric':metric,'cơ sở @0,5':f'{a["mean"]:.6f} ± {a.sample_std:.6f}',
            'tuned trên cùng validation':f'{b["mean"]:.6f} ± {b.sample_std:.6f}',
            'delta trung bình':float(b['mean']-a['mean'])})
    rare_rows=[]
    for label in protocol['rare_labels_train_defined']:
        a=rare[(rare.label==label)&(rare['mode']=='fixed_0.5')].iloc[0]
        b=rare[(rare.label==label)&(rare['mode']=='per_label_tuned_on_validation')].iloc[0]
        rare_rows.append({'nhãn':label,'train support':int(a.train_support),'val support':int(a.validation_support),
            'F1 cơ sở':f'{a.f1_mean:.6f} ± {a.f1_std:.6f}',
            'F1 tuned':f'{b.f1_mean:.6f} ± {b.f1_std:.6f}', 'delta':float(b.f1_mean-a.f1_mean)})
    resource_table=resources[['seed','best_epoch','fit_and_validation_seconds','peak_cuda_allocated_mib','skipped_amp_steps','reload_max_abs_score_diff']]
    text=f'''# Kết quả C3 DistilBERT — Nhật Huy

## Phạm vi thực tế

Đã fine-tune **3 seed {protocol['seeds']}**, mỗi seed 3 epoch, trên **43.410 train**;
đánh giá **5.426 validation** thật của GoEmotions simplified. Không tạo dữ liệu bổ sung,
không dùng test. Snapshot dữ liệu/model, cấu hình và quy tắc chọn kết quả được ghi trước
khi train trong [protocol.json](protocol.json). Các số dưới đây được chương trình đọc từ
artifact đã kiểm hash/ID/mapping và tính lại bằng metric chung của repo.

Đây là kết quả C3 trên validation. Chưa xác định kiến trúc thắng C1/C2/C3, chưa có kết quả
test cuối, chưa hoàn thành đối chiếu lỗi giữa ba kiến trúc. Huy có thể bàn giao phần độc lập này.

## Cấu hình và phương pháp

- Checkpoint `{protocol['config']['model_id']}`, revision `{protocol['config']['revision']}`.
- Official split, 28 nhãn, multi-hot float32; tokenize tối đa 128 token, padding động.
- Fine-tune toàn bộ encoder và head 28 logits; BCEWithLogitsLoss, sigmoid khi suy luận.
- AdamW lr 2e-5, weight decay 0,01 (trừ bias/LayerNorm), warmup 10% rồi linear decay,
  clip gradient 1,0; batch vật lý/hiệu dụng 16, AMP, gradient checkpointing, CPU threads=4.
- Mỗi seed chọn epoch Macro-F1 validation @0,5 cao nhất, hòa giữ epoch sớm hơn.
- Metric đủ 28 nhãn, zero_division=0; std mẫu dùng ddof=1, n=3.
- GPU RTX 3050 Laptop 4 GB, PyTorch 2.11.0+cu128, Transformers 4.57.6.
- PyTorch cảnh báo một kernel attention backward không bảo đảm deterministic; seed và
  môi trường hỗ trợ tái lập, không bảo đảm mọi bit của lần train mới giống lần này.

DistilBERT là mô hình được tác giả tiền huấn luyện qua distillation từ BERT. Trong đồ án,
nhóm nạp checkpoint đã có và fine-tune cho 28 nhãn, không tự huấn luyện teacher hay tự
thực hiện distillation. Nguồn: [Sanh và cộng sự, 2019](https://arxiv.org/abs/1910.01108),
[model card](https://huggingface.co/distilbert/distilbert-base-uncased) (đọc 07/10/2026).

## Kết quả cơ sở từng seed

{rows.to_markdown(index=False,floatfmt=('.0f','.0f')+('.6f',)*(len(rows.columns)-2))}

## Trung bình và độ lệch chuẩn

{table[['metric','mean ± std','n']].to_markdown(index=False)}

![Loss train và Macro-F1 validation theo epoch của ba seed](learning_curves.png)

Không thay kết quả trung bình bằng seed tốt nhất. Seed **{selected['seed']}** chỉ được chọn
làm checkpoint đại diện C3 để đọc lỗi và minh họa app theo quy tắc đã ghi trước.
Đây không phải kết luận DistilBERT là mô hình thắng toàn nhóm.

## Ngưỡng riêng từng nhãn — phần nâng cao C3

Mỗi checkpoint/seed chọn ngưỡng riêng từ scores validation của chính nó trên lưới
0,05..0,95, bước 0,05. Hòa F1 chọn ngưỡng gần 0,5 nhất, tiếp tục hòa chọn ngưỡng lớn hơn.
Lưu ngưỡng gắn hash checkpoint, scores, revision và mapping. Không lấy ngưỡng từ A/C1/C2.
Giữ nguyên trọng số đã train; đây là thay luật quyết định, không phải train thêm mô hình.

**Cảnh báo diễn giải:** ngưỡng vừa được chọn vừa được đo trên cùng validation nên số tuned
có thể lạc quan. Chưa thể kết luận mức cải thiện tương ứng trên test hoặc dữ liệu mới.
Hamming Loss càng thấp càng tốt; F1/P/R càng cao càng tốt, nên dấu delta cần đọc theo metric.

{pd.DataFrame(comparison).to_markdown(index=False,floatfmt='.6f')}

Trên các run này, F1 và recall trung bình tăng sau tuning, nhưng micro precision giảm
và Hamming Loss tăng (xấu hơn). Cần báo cáo cả đánh đổi này, không kết luận mọi mặt đều cải thiện.

### Nhãn hiếm: đủ năm nhãn đã xác định từ train

{pd.DataFrame(rare_rows).to_markdown(index=False,floatfmt='.6f')}

Grief vẫn có F1 bằng 0 ở cả ba seed trước và sau tuning. Relief có mean F1 tuned thấp
và std lớn hơn mean; kết quả giữa các seed chưa ổn định. Không kết luận cả năm nhãn đều tăng.

![F1 nhãn hiếm trước và sau chọn ngưỡng trên cùng validation](rare_labels_f1.png)

Các bảng giữ đủ nhãn tăng, không tăng hoặc giảm. Chi tiết P/R/F1 từng nhãn và từng seed ở
[rare_labels_before_after.csv](rare_labels_before_after.csv); bảng đủ 28 nhãn ở
[per_label_all_seeds.csv](per_label_all_seeds.csv).

## Phân tích lỗi C3

Luật đếm dùng ngưỡng cơ sở 0,5:

- `FN_va_FP_cung_cau`: ít nhất một nhãn thật bị bỏ sót và ít nhất một nhãn thừa cùng câu.
  Đây là cặp lỗi dự đoán, không phải đồng xuất hiện nhãn và chưa tự chứng minh gần nghĩa.
- `thieu_mot_phan_nhan_that`: câu có ít nhất hai nhãn thật, nhận đúng ít nhất một nhãn nhưng bỏ sót nhãn khác.
- `sai_nhan_hiem_hoac_neutral`: FP hoặc FN thuộc một trong năm nhãn hiếm hoặc neutral.

{groups[['seed','group','n_samples']].to_markdown(index=False)}

Nhóm lỗi có thể chồng lấn, không cộng các hàng để suy ra tổng câu sai. Ví dụ nguyên văn,
ID, true/pred, FN/FP và scores đủ 28 nhãn của seed đại diện nằm trong
[error_examples.json](error_examples.json), bản bảng [error_examples.csv](error_examples.csv).
Chọn tối đa 12 mẫu mỗi nhóm theo ID tăng dần, không tạo câu minh họa. Xem phần nhận xét
cụ thể trong [EXPERIMENTS.md](EXPERIMENTS.md).

## Chi phí và kiểm checkpoint

{resource_table.to_markdown(index=False,floatfmt=('.0f','.0f','.3f','.3f','.0f','.6f'))}

Thời gian là train cộng validation của ba epoch, không gồm tải model hoặc kiểm reload cuối.
Nguồn điện/tải máy có thể thay đổi trong quá trình chạy; đây là nhật ký tài nguyên thực tế,
không phải benchmark tốc độ trong điều kiện được kiểm soát.
Không dùng những số này để khẳng định nhanh hơn C1/C2 khi chưa có phép đo cùng điều kiện.
Log thô ở `train_seed_42.log`, `train_seed_123.log`, `train_seed_2026.log`.
Số bước AMP bị bỏ qua được ghi trong bảng tài nguyên và từng `run_seed_*.json`; scheduler
chỉ tiến khi optimizer thực sự cập nhật. Một lượt seed 42 ban đầu dừng do xử lý AMP trước
khi hoàn tất; log được giữ ở [interrupted_fp16_gradcheck](../interrupted_fp16_gradcheck/README.md)
và **không** được tính vào bất kỳ bảng kết quả nào ở đây. Ba seed trên đều chạy từ đầu
bằng cùng bản mã sửa lỗi, không nối checkpoint của lượt dở dang.
Ngày 08/10/2026, một lượt seed 123 được phát hiện không còn tiến trình và log dừng trước
khi hoàn thành epoch 1, không có traceback xác định nguyên nhân. Log lưu ở
`train_seed_123_interrupted_20261008.log`; lượt đó cũng bị loại khỏi kết quả và seed 123
được chạy lại từ checkpoint gốc với cùng cấu hình/mã. Không suy đoán nguyên nhân gián đoạn.

## Bàn giao và phần còn phụ thuộc nhóm

- Code C3, config, notebook đã chạy, bảng số thật, ngưỡng, lỗi và báo cáo: sẵn sàng bàn giao.
- Checkpoint/tokenizer/scores: `data/processed/c3_distilbert/full/seed_<seed>/`, nằm cục bộ,
  không đưa file model lớn lên Git. Model dùng safetensors; nguồn code của mỗi run được lưu ở `source/`.
- App minh họa C3 dùng seed đại diện, hỗ trợ ngưỡng cơ sở và ngưỡng riêng đúng run; không
  ghi đây là demo best C của nhóm. Kiểm app/CLI và input biên ở [verification.json](verification.json).
- Cần C1/C2 để so kiến trúc, đối chiếu ít nhất ba nhóm lỗi cùng ID và chọn demo cuối.
- Cần nhóm khóa danh sách mô hình/cấu hình/ngưỡng và protocol trước đánh giá test cuối.

Hướng dẫn: [DISTILBERT_STUDY.md](../../../docs/DISTILBERT_STUDY.md).
Notebook: [distilbert_huy.ipynb](../../../notebooks/05_distilbert_huy.ipynb).
'''
    (folder/'C3_RESULTS.md').write_text(text,encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for seed,part in history.groupby('seed'):
        axes[0].plot(part.epoch,part.train_loss,marker='o',label=f'Seed {seed}')
        axes[1].plot(part.epoch,part.validation_macro_f1,marker='o',label=f'Seed {seed}')
    for ax in axes:
        ax.set_xticks([1,2,3]); ax.set_xlabel('Epoch'); ax.grid(alpha=.2); ax.legend()
    axes[0].set_ylabel('BCE trung bình trên train'); axes[0].set_title('Loss huấn luyện')
    axes[1].set_ylabel('Macro-F1 validation @0,5'); axes[1].set_title('Đủ 28 nhãn — ba seed C3')
    fig.savefig(folder/'learning_curves.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    x=np.arange(len(protocol['rare_labels_train_defined']))
    for offset,mode,title in [(-.18,'fixed_0.5','Cơ sở @0,5'),(.18,'per_label_tuned_on_validation','Tuned trên cùng validation')]:
        part=rare[rare['mode']==mode].set_index('label').loc[protocol['rare_labels_train_defined']]
        ax.bar(x+offset,part.f1_mean,width=.36,yerr=part.f1_std,capsize=3,label=title)
    ax.set_xticks(x,protocol['rare_labels_train_defined']); ax.set_ylim(0,1)
    ax.set_ylabel('F1 trung bình ± sample std (3 seed)'); ax.legend()
    ax.set_title('Nhãn hiếm C3 — tuned-validation chưa là đánh giá độc lập')
    ax.grid(axis='y',alpha=.2); fig.savefig(folder/'rare_labels_f1.png',dpi=180); plt.close(fig)
    print('PASS: report and plots generated from real run tables.')


if __name__=='__main__':
    main()
